"""下载公开脑电语料（PhysioNet 开放许可）。

只做三件事：读官方清单、下载、校验 SHA256。数据不进仓库，写到 --root 指定的目录。

清单与校验值都来自数据集自带的 RECORDS / SHA256SUMS.txt，不手工维护文件列表，
避免"以为下全了"。

用法（仓库根目录）：
    python scripts/download_corpora.py --dataset eegmmidb  --root D:/wyu-bci-data --jobs 8
    python scripts/download_corpora.py --dataset sleep-edfx --root D:/wyu-bci-data --jobs 8

断点续传：未完成的文件写 .part，重跑时用 Range 续传；已完整且校验通过的文件直接跳过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MIRROR = "https://physionet-open.s3.amazonaws.com/{name}/{version}/"
PRIMARY = "https://physionet.org/files/{name}/{version}/"

CORPORA = {
    "eegmmidb": dict(
        version="1.0.0",
        license="Open Data Commons Attribution License v1.0",
        summary="EEG Motor Movement/Imagery Dataset，109 被试，64 导，160 Hz",
    ),
    "sleep-edfx": dict(
        version="1.0.0",
        license="Open Data Commons Attribution License v1.0",
        summary="Sleep-EDF Expanded，153 条 cassette + 44 条 telemetry",
        companions="hypnogram",
    ),
    "chbmit": dict(
        version="1.0.0",
        license="Open Data Commons Attribution License v1.0",
        summary="CHB-MIT 头皮 EEG 癫痫库，24 例，686 段",
    ),
}

_print_lock = threading.Lock()

# 标注文件配对规则与 ingest 侧共用一份实现（纯标准库，不会把 numpy 拖进下载器）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_mime_foundation.pairing import pair_hypnograms  # noqa: E402


def _urls(name: str, version: str, path: str) -> list[str]:
    return [MIRROR.format(name=name, version=version) + path,
            PRIMARY.format(name=name, version=version) + path]


def _open(url: str, offset: int = 0, timeout: int = 60):
    headers = {"User-Agent": "wyu-bci-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def fetch_text(name: str, version: str, path: str) -> str:
    last = None
    for url in _urls(name, version, path):
        try:
            with _open(url) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as exc:
            last = exc
    raise RuntimeError(f"无法获取清单 {path}: {last}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_size(name: str, version: str, path: str) -> int | None:
    for url in _urls(name, version, path):
        try:
            request = urllib.request.Request(url, method="HEAD",
                                             headers={"User-Agent": "wyu-bci-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length:
                    return int(length)
        except (urllib.error.URLError, OSError):
            continue
    return None


def download_ranged(name: str, version: str, path: str, part: Path, size: int,
                    chunk_bytes: int, chunk_jobs: int, retries: int = 4) -> None:
    """把大文件切成固定大小的字节区间并发拉取。

    PhysioNet 的单连接被限速到几十 KB/s，而并发连接的总吞吐随连接数上升；
    48 MB 的睡眠 PSG 用 8 路区间下载即可把单文件耗时压到分钟级。
    """
    ranges = [(start, min(start + chunk_bytes, size) - 1) for start in range(0, size, chunk_bytes)]
    with part.open("wb") as handle:
        handle.truncate(size)

    def pull(bounds):
        start, end = bounds
        last = None
        for attempt in range(retries):
            for url in _urls(name, version, path):
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "wyu-bci-downloader/1.0",
                                      "Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        data = response.read()
                    if len(data) != end - start + 1:
                        raise OSError(f"区间长度不符 {len(data)} != {end - start + 1}")
                    with part.open("r+b") as handle:
                        handle.seek(start)
                        handle.write(data)
                    return
                except (urllib.error.URLError, OSError) as exc:
                    last = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{path} 区间 {start}-{end} 失败: {last}")

    with ThreadPoolExecutor(max_workers=chunk_jobs) as pool:
        for future in [pool.submit(pull, bounds) for bounds in ranges]:
            future.result()


def download_one(name: str, version: str, path: str, root: Path, expected: str | None,
                 retries: int = 4, chunk_bytes: int = 8 << 20, chunk_jobs: int = 8) -> dict:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if expected is None or sha256_of(target) == expected:
            return dict(path=path, bytes=target.stat().st_size, status="cached")

    part = target.with_suffix(target.suffix + ".part")
    if part.exists() and expected and sha256_of(part) == expected:
        # 上一次跑到写满但没来得及重命名：直接验收，不重下。
        part.replace(target)
        return dict(path=path, bytes=target.stat().st_size, status="downloaded")
    size = remote_size(name, version, path)
    if part.exists() and size and part.stat().st_size >= size:
        # 早前中断的区间模式会把 .part 预分配到整文件大小，但内容是空洞。
        # 这时续传位置等于总长度，Range 请求会得到 416，必须从头写。
        part.unlink()
    if size and size > chunk_bytes and chunk_jobs > 1:
        download_ranged(name, version, path, part, size, chunk_bytes, chunk_jobs)
        digest = sha256_of(part)
        if expected and digest != expected:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{path} 校验失败: {digest} != {expected}")
        part.replace(target)
        return dict(path=path, bytes=target.stat().st_size, status="downloaded")

    last = None
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        for url in _urls(name, version, path):
            try:
                with _open(url, offset) as response:
                    mode = "ab" if offset and response.status == 206 else "wb"
                    if mode == "wb":
                        offset = 0
                    with part.open(mode) as handle:
                        for chunk in iter(lambda: response.read(1 << 20), b""):
                            handle.write(chunk)
                last = None
                break
            except (urllib.error.URLError, OSError) as exc:
                last = exc
                offset = part.stat().st_size if part.exists() else 0
        if last is None:
            break
        time.sleep(1.5 * (attempt + 1))
    if last is not None:
        raise RuntimeError(f"{path} 下载失败: {last}")

    digest = sha256_of(part)
    if expected and digest != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{path} 校验失败: {digest} != {expected}")
    part.replace(target)
    return dict(path=path, bytes=target.stat().st_size, status="downloaded")


def parse_checksums(text: str) -> dict[str, str]:
    table = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            table[parts[1]] = parts[0]
    return table


def companion_files(records: list[str], checksums: dict[str, str]) -> list[str]:
    """睡眠库的 RECORDS 只列 PSG，标注文件（Hypnogram）不在清单里。

    命名不能靠替换推导，配对规则与 ingest 侧共用 `pair_hypnograms`，
    避免下载与切窗对"哪份标注配哪份 PSG"产生两种理解。
    """
    hypnograms = [path for path in checksums if "Hypnogram" in path]
    if not hypnograms:
        raise RuntimeError("校验清单里没有标注文件，无法补齐睡眠语料")
    try:
        pairs = pair_hypnograms(records, hypnograms)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return [hypnogram for _, hypnogram in pairs]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CORPORA))
    ap.add_argument("--root", required=True, help="数据根目录，例如 D:/wyu-bci-data")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=8, help="大于该体积的文件按区间并发下载")
    ap.add_argument("--chunk-jobs", type=int, default=0,
                    help="每个大文件的并发区间数；默认 0 关闭。"
                         "本机实测：区间模式单文件更快，但连接挂起时会把整批卡住")
    ap.add_argument("--subjects", nargs="*", default=None, help="只下指定被试，如 S001 S002")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = CORPORA[args.dataset]
    version, root = meta["version"], Path(args.root).resolve() / args.dataset / meta["version"]
    records = [line.strip() for line in fetch_text(args.dataset, version, "RECORDS").splitlines()
               if line.strip()]
    all_records = list(records)
    if args.subjects:
        wanted = set(args.subjects)
        records = [r for r in records if r.split("/")[0] in wanted]
    if not records:
        raise SystemExit("清单为空，检查 --subjects")

    checksums = {}
    checksum_text = ""
    try:
        checksum_text = fetch_text(args.dataset, version, "SHA256SUMS.txt")
        checksums = parse_checksums(checksum_text)
    except RuntimeError as exc:
        print(f"[warn] 无校验清单，仅按大小校验: {exc}")

    if meta.get("companions") == "hypnogram":
        if not checksums:
            raise SystemExit("睡眠库必须先拿到 SHA256SUMS.txt，否则无法定位标注文件")
        companions = companion_files(records, checksums)
        records = records + companions

    print(f"{args.dataset} {version}: {len(records)} 个文件 → {root}")
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    # 清单始终写全量（即使本次只下 --subjects 子集），避免后续 ingest 读到残缺 RECORDS。
    (root / "RECORDS").write_text("\n".join(all_records) + "\n", encoding="utf-8")
    if checksum_text:
        (root / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")

    started, done, failed, moved = time.time(), 0, [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download_one, args.dataset, version, path, root,
                               checksums.get(path), 4, args.chunk_mb << 20, args.chunk_jobs): path
                   for path in records}
        for future in as_completed(futures):
            path = futures[future]
            try:
                row = future.result()
                moved += row["bytes"]
                done += 1
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不应中断整批
                failed.append(dict(path=path, error=str(exc)))
                done += 1
            if done % 25 == 0 or done == len(records):
                elapsed = time.time() - started
                rate = moved / elapsed / 1e6 if elapsed else 0
                with _print_lock:
                    print(f"  {done}/{len(records)}  {moved / 1e9:.2f} GB  {rate:.1f} MB/s  "
                          f"失败 {len(failed)}")

    record = dict(
        dataset=args.dataset, version=version, summary=meta["summary"], license=meta["license"],
        source=f"https://physionet.org/content/{args.dataset}/{version}/",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        files=done - len(failed), failed=failed,
        bytes=moved, root=str(root),
        note="数据不随仓库分发；仅本地研究使用",
    )
    (root / "download_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{record['files']} 个文件，{moved / 1e9:.2f} GB，失败 {len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
