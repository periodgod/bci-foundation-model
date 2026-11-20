"""Metrics from saved predictions, with equal-weight subject summaries."""
import numpy as np


def metrics(predictions, targets, task):
    p, y = np.asarray(predictions), np.asarray(targets)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError("Empty or non-finite predictions")
    if task.endswith("regression"):
        p, y = p.reshape(len(y), -1), y.reshape(len(y), -1)
        if p.shape != y.shape:
            raise ValueError("Regression target/output shape mismatch")
        return dict(mse=float(np.square(p - y).mean()), mae=float(np.abs(p - y).mean()))
    y = y.astype(int).reshape(-1)
    if (y < 0).any() or (y >= p.shape[-1]).any():
        raise ValueError("Class label outside output vocabulary")
    pred = p.argmax(-1)
    matrix = np.zeros((p.shape[1], p.shape[1]), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    tp, support = np.diag(matrix), matrix.sum(1)
    observed = support > 0
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=observed)
    denominator = matrix.sum(0) + support
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    prob = np.exp(p - p.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(recall[observed].mean()),
                macro_f1=float(f1.mean()), nll=float(-np.log(prob[np.arange(len(y)), y].clip(1e-12)).mean()),
                support=support.tolist(), confusion=matrix.tolist())


def summarize(predictions, targets, metadata, task):
    overall = metrics(predictions, targets, task)
    groups = {}
    for i, m in enumerate(metadata):
        groups.setdefault(f"{m['dataset']}:{m['subject']}", []).append(i)
    by_subject = {s: metrics(np.asarray(predictions)[indices], np.asarray(targets)[indices], task)
                  for s, indices in groups.items()}
    scalar = [k for k, v in overall.items() if isinstance(v, float)]
    macro = {k: float(np.mean([v[k] for v in by_subject.values()])) for k in scalar}
    return dict(overall=overall, subject_macro=macro, by_subject=by_subject)


def reject_predictions(logits, threshold):
    """Threshold must be frozen using validation; abstention is not unknown-task discovery."""
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be [0,1]")
    p = np.asarray(logits)
    prob = np.exp(p - p.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    return np.where(prob.max(-1) >= threshold, prob.argmax(-1), -1)
