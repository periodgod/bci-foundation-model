"""Bounded PhysioNet intake; real signals, subject-first splits, official hashes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import time

import numpy as np

from .data import audit_splits, sha256


SOURCES = {
    "mi": {
        "dataset": "eegmmidb", "version": "1.0.0", "sample_rate": 160,
        "subjects": {"S001": "train", "S002": "validation", "S003": "test"},
        "labels": ["left_hand_imagery", "right_hand_imagery"],
        "files": [f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in (1, 2, 3) for r in (4, 8, 12)],
    },
    "sleep": {
        "dataset": "sleep-edfx", "version": "1.0.0", "sample_rate": 100,
        "subjects": {"SC400": "train", "SC401": "validation", "SC402": "test"},
        "labels": ["W", "N1", "N2", "N3", "REM"],
        "files": [f"sleep-cassette/{name}" for name in (
            "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf",
            "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf",
            "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf")],
    },
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_spec(kind, scope):
    spec = dict(SOURCES[kind])
    if scope == "development":
        ids = range(1, 11) if kind == "mi" else range(10)
        validation = {2, 9, 10} if kind == "mi" else {1, 8, 9}
        test = 3 if kind == "mi" else 2
        spec["subjects"] = {f"S{i:03d}" if kind == "mi" else f"SC4{i:02d}":
                            "test" if i == test else "validation" if i in validation else "train" for i in ids}
        spec["files"] = [f"S{i:03d}/S{i:03d}R{r:02d}.edf" for i in ids for r in (4, 8, 12)] if kind == "mi" else []
    elif scope != "smoke":
        raise ValueError("Unknown intake scope")
    return spec


def download_data(kind, root="data", workers=3, scope="smoke"):
    """Download only the listed records, validate against source SHA256SUMS."""
    import requests
    spec = source_spec(kind, scope)
    directory = Path(root).resolve() / "raw" / "precheck" / spec["dataset"] / spec["version"]
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://physionet.org/files/{spec['dataset']}/{spec['version']}/"
    transfer_base = f"https://physionet-open.s3.amazonaws.com/{spec['dataset']}/{spec['version']}/"
    page = f"https://physionet.org/content/{spec['dataset']}/{spec['version']}/"
    headers = {"User-Agent": "EEG-precompetition-intake/1.0"}
    response = requests.get(base + "SHA256SUMS.txt", headers=headers, timeout=(15, 45))
    response.raise_for_status()
    (directory / "SHA256SUMS.txt").write_bytes(response.content)
    hashes = {}
    for line in response.text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            hashes[fields[1].lstrip("*")] = fields[0]
    if kind == "sleep" and scope == "development":
        for subject in spec["subjects"]:
            for suffix in ("-PSG.edf", "-Hypnogram.edf"):
                matches = [n for n in hashes if n.startswith(f"sleep-cassette/{subject}1") and n.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError("First-night pairing is not unique in official file list")
                spec["files"].append(matches[0])
    if any(name not in hashes for name in spec["files"]):
        raise ValueError("Official checksum list does not cover all requested files")
    response = requests.get(page, headers=headers, timeout=(15, 45))
    response.raise_for_status()
    (directory / f"source-{scope}.html").write_bytes(response.content)
    if "Open Data Commons Attribution License v1.0" not in response.text:
        raise ValueError("Recheck source license before use")
    links = re.findall(r'href="([^"]*/view-license/[^"]+)"', response.text)
    license_urls = sorted(set("https://physionet.org" + u if u.startswith("/") else u for u in links))
    if not license_urls:
        raise ValueError("License document link not found")
    response = requests.get(license_urls[0], headers=headers, timeout=(15, 45))
    response.raise_for_status()
    license_path = directory / f"license-{scope}.html"
    license_path.write_bytes(response.content)
    # Explicit bounds: 80 MB/file, 1 GB for the expanded diagnostic subset.
    def fetch(name):
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Download target escapes dataset directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != hashes[name]:
                raise ValueError(f"Existing raw file checksum mismatch: {name}")
            return dict(path=name, sha256=hashes[name], bytes=target.stat().st_size, reused=True)
        partial = target.with_suffix(target.suffix + ".part")
        started = time.monotonic()
        digest, size = hashlib.sha256(), 0
        with requests.Session() as session:
            head = session.head(transfer_base + name, headers=headers, timeout=(15, 45))
            head.raise_for_status()
            expected = int(head.headers["Content-Length"])
            if expected > 80_000_000:
                raise ValueError("File exceeds precheck download budget")
            block = 1024 * 1024
            # Keep only complete committed blocks when resuming an interrupted transfer.
            if partial.exists():
                size = min(partial.stat().st_size // block * block, expected // block * block)
                with partial.open("r+b") as f:
                    f.truncate(size)
                with partial.open("rb") as f:
                    for chunk in iter(lambda: f.read(block), b""):
                        digest.update(chunk)
            def get_range(offset):
                end = min(offset + block, expected) - 1
                ranged_headers = dict(headers, Range=f"bytes={offset}-{end}")
                for retry in range(2):
                    try:
                        with requests.get(transfer_base + name, headers=ranged_headers, timeout=(15, 45)) as r:
                            r.raise_for_status()
                            if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {offset}-{end}/{expected}":
                                raise ValueError("Server did not honor exact byte range")
                            if len(r.content) != end - offset + 1:
                                raise ValueError("Incomplete byte range")
                            return r.content
                    except requests.RequestException:
                        if retry:
                            raise
                raise RuntimeError("Byte range failed")
            with partial.open("ab") as f, ThreadPoolExecutor(max_workers=3) as ranges:
                offsets = list(range(size, expected, block))
                for index in range(0, len(offsets), 3):
                    for chunk in ranges.map(get_range, offsets[index:index + 3]):
                        size += len(chunk)
                        if time.monotonic() - started > 1200:
                            raise ValueError("File exceeded bounded download budget")
                        digest.update(chunk)
                        f.write(chunk)
                        f.flush()
                    if kind == "sleep":
                        print(f"download {name}: {size / 1e6:.1f}/{expected / 1e6:.1f} MB", flush=True)
        if size != expected:
            raise ValueError("Incomplete download")
        if digest.hexdigest() != hashes[name]:
            raise ValueError(f"Official SHA256 mismatch: {name}")
        os.replace(partial, target)
        print(f"verified {kind}: {name} ({size / 1e6:.2f} MB)", flush=True)
        return dict(path=name, url=base + name, transfer_url=transfer_base + name,
                    sha256=hashes[name], bytes=size, reused=False)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, name) for name in spec["files"]]
        for future in as_completed(futures):
            records.append(future.result())
    if sum(r["bytes"] for r in records) > (1_000_000_000 if scope == "development" else 300_000_000):
        raise ValueError("Selected dataset exceeds total budget")
    record = dict(dataset=spec["dataset"], version=spec["version"], source=page,
                  retrieved_at=datetime.now(timezone.utc).isoformat(),
                  license="Open Data Commons Attribution License v1.0", license_url=license_urls[0],
                  license_sha256=sha256(license_path),
                  checksum_manifest_sha256=sha256(directory / "SHA256SUMS.txt"),
                  use_scope="local_research_precheck_only; competition permission unresolved",
                  subject_split=spec["subjects"], files=sorted(records, key=lambda r: r["path"]))
    write_json(directory / ("download_record.json" if scope == "smoke" else "download_record-development.json"), record)
    return record


def check_raw(kind, root, scope):
    spec = source_spec(kind, scope)
    directory = Path(root).resolve() / "raw" / "precheck" / spec["dataset"] / spec["version"]
    record = json.loads((directory / ("download_record.json" if scope == "smoke" else "download_record-development.json")).read_text(encoding="utf-8"))
    if record["subject_split"] != spec["subjects"]:
        raise ValueError("Raw intake subject protocol changed")
    for r in record["files"]:
        if sha256(directory / r["path"]) != r["sha256"]:
            raise ValueError("Raw data changed since intake")
    return directory, record


def mi_trial_intervals(annotations, fs, n_times):
    """Use onset + [0.5, 4.0) s; reject windows exceeding event/run boundaries."""
    intervals, excluded = [], []
    for index, (onset, duration, description) in enumerate(zip(
            annotations.onset, annotations.duration, annotations.description)):
        if description not in {"T1", "T2"}:
            continue
        start, stop = int(round((onset + .5) * fs)), int(round((onset + 4.) * fs))
        if duration < 4. or stop > n_times or start < 0:
            excluded.append(dict(annotation=index, reason="event_too_short_or_outside_record"))
            continue
        intervals.append((index, start, stop, 0 if description == "T1" else 1))
    return intervals, excluded


SLEEP_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


def sleep_epoch_intervals(annotations, fs, n_times):
    """30 s epochs; merge R&K 3/4 to N3, exclude unknown/movement explicitly."""
    onset = np.asarray(annotations.onset)
    duration = np.asarray(annotations.duration)
    descriptions = np.asarray(annotations.description)
    asleep = np.isin(descriptions, ["Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage 4", "Sleep stage R"])
    if not asleep.any():
        raise ValueError("No scored sleep in annotation file")
    lower = max(0., float(onset[asleep].min()) - 1800.)
    upper = min(n_times / fs, float((onset + duration)[asleep].max()) + 1800.)
    intervals, excluded = [], []
    previous_end = -1.
    for index, (t, seconds, label) in enumerate(zip(onset, duration, descriptions)):
        if t < previous_end - 1 / fs:
            raise ValueError("Overlapping hypnogram annotations")
        previous_end = float(t + seconds)
        if label not in SLEEP_MAP:
            excluded.append(dict(annotation=index, label=str(label), seconds=float(seconds), reason="unknown_or_movement"))
            continue
        count = int(round(seconds / 30.))
        if abs(count * 30. - seconds) > 1 / fs:
            raise ValueError("Hypnogram interval not divisible by 30 seconds")
        for epoch in range(count):
            beginning = t + epoch * 30.
            ending = beginning + 30.
            if beginning < lower or ending > upper:
                continue
            start, stop = int(round(beginning * fs)), int(round(ending * fs))
            if start < 0 or stop > n_times:
                excluded.append(dict(annotation=index, epoch=epoch, reason="outside_record"))
                continue
            intervals.append((index, epoch, start, stop, SLEEP_MAP[str(label)]))
    return intervals, excluded, [lower, upper]


def signal_summary(x):
    return dict(shape=list(x.shape), finite=bool(np.isfinite(x).all()),
                peak_abs_microvolt=float(np.max(np.abs(x)) * 1e6),
                rms_microvolt=float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) * 1e6))


def ingest_data(kind, root="data", scope="smoke"):
    import mne
    spec = source_spec(kind, scope)
    raw_directory, download = check_raw(kind, root, scope)
    spec["files"] = [r["path"] for r in download["files"]]
    directory = Path(root).resolve() / "processed" / "precheck" / (f"{kind}-v1" if scope == "smoke" else f"{kind}-development-v1")
    directory.mkdir(parents=True, exist_ok=False)
    # Assignment is frozen before loading or windowing; no adaptive split selection.
    write_json(directory / "protocol.json", dict(
        protocol_version=1, kind=kind, subject_split=spec["subjects"],
        use="real_data_pipeline_smoke_not_competition_training", sample_rate=spec["sample_rate"],
        preprocessing="MI: fixed 8-30Hz 4th-order zero-phase Butterworth per independent run; sleep: native EEG unfiltered",
        unit="MNE read_raw_edf returns volt; no extra microvolt conversion",
        reference="preserve acquisition reference; no learned re-reference",
        window="MI: annotation+[0.5,4.0)s; sleep:30s, sleep span+30min wake each side",
        fit="global normalization and any learned baseline only on train",
        split_ratio="1:1:1" if scope == "smoke" else "6:3:1",
        evaluation="local exploratory development, not official 8:1:1", final_test="reserved; no model evaluation"))
    records, audit, vocabulary = [], [], {}
    def save(x, channel_names, name, target, subject, session, trial, start, stop, raw_files):
        if not np.isfinite(x).all() or np.std(x) == 0:
            raise ValueError(f"Invalid signal: {name}")
        for channel in channel_names:
            if channel not in vocabulary:
                vocabulary[channel] = len(vocabulary)
        path = directory / (name + ".npz")
        np.savez(path, x=x.astype(np.float32), channel_ids=np.array([vocabulary[c] for c in channel_names]),
                 coords=np.zeros((len(channel_names), 3), np.float32),
                 coord_valid=np.zeros(len(channel_names), bool), target=np.array(target, dtype=np.int64))
        records.append(dict(path=path.name, sha256=sha256(path), sample_id=name, dataset=spec["dataset"],
                            subject=subject, session=session, trial=trial, split=spec["subjects"][subject],
                            source_files=raw_files, start_sample=start, stop_sample=stop))
    if kind == "mi":
        expected_names = None
        for relative in spec["files"]:
            subject = relative.split("/")[0]
            session = Path(relative).stem[-3:]
            raw = mne.io.read_raw_edf(raw_directory / relative, preload=True, verbose="ERROR")
            if raw.info["sfreq"] != 160 or len(raw.ch_names) != 64:
                raise ValueError("Unexpected MI sample rate/channel count")
            names = [name.strip(".").upper() for name in raw.ch_names]
            if len(set(names)) != 64:
                raise ValueError("MI channel mapping is not unique")
            if expected_names is not None and names != expected_names:
                raise ValueError("MI channel order changed; audit explicit mapping")
            expected_names = names
            original = raw.get_data()
            intervals, excluded = mi_trial_intervals(raw.annotations, 160, raw.n_times)
            raw.filter(8., 30., method="iir", iir_params=dict(order=4, ftype="butter"), phase="zero", verbose="ERROR")
            filtered = raw.get_data()
            for index, start, stop, target in intervals:
                name = f"{subject}_{session}_event{index:03d}"
                save(filtered[:, start:stop], names, name, target, subject, session,
                     str(index), start, stop, [relative])
            audit.append(dict(file=relative, subject=subject, channels=names, native=signal_summary(original),
                              filtered=signal_summary(filtered), retained_trials=len(intervals), excluded=excluded,
                              events=[dict(onset=float(o), duration=float(d), description=str(s)) for o, d, s in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)],
                              unit="volt", physical_dimensions=raw._orig_units))
    else:
        for psg_name, hyp_name in zip(spec["files"][::2], spec["files"][1::2]):
            subject = Path(psg_name).name[:5]
            raw = mne.io.read_raw_edf(raw_directory / psg_name,
                                    include=["EEG Fpz-Cz", "EEG Pz-Oz"], preload=True, verbose="ERROR")
            if raw.info["sfreq"] != 100 or set(raw.ch_names) != {"EEG Fpz-Cz", "EEG Pz-Oz"}:
                raise ValueError("Unexpected sleep EEG channels/sample rate")
            annotations = mne.read_annotations(raw_directory / hyp_name)
            # read_annotations(EDF) can leave orig_time unset; use MNE's native header metadata.
            hyp_header = mne.io.read_raw_edf(raw_directory / hyp_name, preload=False, verbose="ERROR")
            if raw.info["meas_date"] is None or raw.info["meas_date"] != hyp_header.info["meas_date"]:
                raise ValueError("PSG/hypnogram acquisition start dates differ")
            if annotations.orig_time is not None and raw.info["meas_date"] != annotations.orig_time:
                raise ValueError("PSG/hypnogram time origins differ")
            intervals, excluded, crop = sleep_epoch_intervals(annotations, 100, raw.n_times)
            signal = raw.get_data()
            names = [name.removeprefix("EEG ") for name in raw.ch_names]
            for annotation, epoch, start, stop, target in intervals:
                name = f"{subject}_night1_sample{start:08d}"
                save(signal[:, start:stop], names, name, target, subject, "night1",
                     f"{annotation}:{epoch}", start, stop, [psg_name, hyp_name])
            audit.append(dict(file=psg_name, hypnogram=hyp_name, subject=subject, channels=names,
                              native=signal_summary(signal), unit="volt", physical_dimensions=raw._orig_units,
                              retained_epochs=len(intervals), exclusions=excluded, crop_seconds=crop,
                              measurement_date=raw.info["meas_date"].isoformat(),
                              hypnogram_measurement_date=hyp_header.info["meas_date"].isoformat(),
                              annotation_date=annotations.orig_time.isoformat() if annotations.orig_time else None))
    audit_splits(records, "subject")
    manifest = dict(sample_rate=spec["sample_rate"], unit="volt", coordinate_unit="meter",
                    coordinate_frame="unknown_not_used; channel identities documented",
                    channel_vocabulary={str(v): k for k, v in vocabulary.items()},
                    channel_kind="bipolar_derivation" if kind == "sleep" else "scalp_acquisition_channel",
                    use_status="approved", use_scope="local_research_precheck_only",
                    evidence="real_data_pipeline_smoke", group_by="subject", labels=spec["labels"],
                    source=download["source"], source_version=spec["version"],
                    download_record_sha256=sha256(raw_directory / ("download_record.json" if scope == "smoke" else "download_record-development.json")),
                    protocol_sha256=sha256(directory / "protocol.json"), samples=records)
    write_json(directory / "manifest.json", manifest)
    targets = {split: [] for split in ("train", "validation", "test")}
    for record in records:
        with np.load(directory / record["path"], allow_pickle=False) as arrays:
            targets[record["split"]].append(int(arrays["target"]))
    counts = {split: np.bincount(values, minlength=len(spec["labels"])).tolist()
              for split, values in targets.items()}
    write_json(directory / "intake_audit.json", dict(records=audit, class_counts=counts,
                labels=spec["labels"], subjects=spec["subjects"], total_windows=len(records),
                manifest_sha256=sha256(directory / "manifest.json"),
                test_access="input/label integrity audit only; no predictions, metrics, or selection"))
    return dict(manifest=str(directory / "manifest.json"), windows=len(records), class_counts=counts)
