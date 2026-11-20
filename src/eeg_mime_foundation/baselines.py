"""Train-only task-specific baselines on the same ingested EEG windows."""
from pathlib import Path
import json

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ManifestDataset, sha256
from .engine import code_record, write_json
from .metrics import summarize


def load_arrays(manifest, split):
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    data = ManifestDataset(manifest, split, m["sample_rate"], "classification")
    signals, targets, metadata = [], [], []
    for sample in data:
        signals.append(sample.x.numpy())
        targets.append(int(sample.target))
        metadata.append({k: getattr(sample, k) for k in ("sample_id", "dataset", "subject", "session", "trial")})
    return np.stack(signals), np.array(targets), metadata


def sleep_features(x, fs=100):
    """Fixed Welch features; no cross-window or held-out fitted statistics."""
    frequency, psd = welch(x, fs=fs, nperseg=400, noverlap=200, axis=-1)
    bands = ((.5, 4), (4, 8), (8, 12), (12, 16), (16, 30))
    powers = np.stack([psd[..., (frequency >= lo) & (frequency < hi)].sum(-1) * (frequency[1] - frequency[0])
                       for lo, hi in bands], axis=-1)
    total = np.maximum(powers.sum(-1, keepdims=True), 1e-30)
    relative = powers / total
    log_power = np.log(np.maximum(powers, 1e-30))
    rms = np.log(np.maximum(np.mean(x ** 2, axis=-1, keepdims=True), 1e-30))
    return np.concatenate((log_power, relative, rms), axis=-1).reshape(len(x), -1)


def run_baseline(kind, manifest, directory):
    import joblib
    import mne
    mne.set_log_level("ERROR")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    train, ytrain, _ = load_arrays(manifest, "train")
    validation, yval, metadata = load_arrays(manifest, "validation")
    labels = json.loads(Path(manifest).read_text(encoding="utf-8"))["labels"]
    nclasses = len(labels)
    if kind == "mi":
        from mne.decoding import CSP
        def estimator():
            return make_pipeline(CSP(n_components=4, reg=.1, log=True, norm_trace=False, rank="full"),
                                 LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        name = "CSP4_regularized_covariance_LDA_shrinkage"
    else:
        train, validation = sleep_features(train), sleep_features(validation)
        def estimator():
            return make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=17))
        name = "Welch_bandpower_StandardScaler_balanced_LogisticRegression"
    model = estimator().fit(train, ytrain)
    def logits_for(fitted):
        probability = fitted.predict_proba(validation)
        full = np.full((len(yval), nclasses), 1e-12)
        full[:, fitted.classes_.astype(int)] = probability
        return np.log(full.clip(1e-12))
    logits = logits_for(model)
    shuffled_y = np.random.default_rng(17).permutation(ytrain)
    shuffled = estimator().fit(train, shuffled_y)
    prior = (np.bincount(ytrain, minlength=nclasses) + 1) / (len(ytrain) + nclasses)
    prior_logits = np.tile(np.log(prior), (len(yval), 1))
    report = dict(name=name, evidence="real_data_pipeline_smoke", manifest_sha256=sha256(Path(manifest)),
                  code=code_record(), labels=labels, train_windows=len(ytrain), validation_windows=len(yval),
                  validation=summarize(logits, yval, metadata, "classification"),
                  train_prior_baseline=summarize(prior_logits, yval, metadata, "classification"),
                  shuffled_train_label_control=summarize(logits_for(shuffled), yval, metadata, "classification"),
                  control_seed=17, fit_scope="train only; one predefined estimator; no search",
                  limitation="one train subject and one validation subject; shuffled-label control is one diagnostic draw",
                  test_predictions=0)
    write_json(directory / "metrics.json", report)
    joblib.dump(model, directory / "baseline.joblib")
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for prediction, target, meta in zip(logits, yval, metadata):
            f.write(json.dumps(dict(**meta, prediction=prediction.tolist(), target=int(target))) + "\n")
    return report
