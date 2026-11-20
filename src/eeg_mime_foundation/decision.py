"""Validation-only branch decisions. Missing evidence stays pending."""
import numpy as np


BRANCHES = {
    "S0": {"architecture": "fusion", "question": "波形与频谱融合是否优于轻量基线？"},
    "S1": {"architecture": "compact", "question": "低资源下是否能维持任务效果？"},
    "S2": {"architecture": "pretrained", "question": "经许可的预训练编码器是否带来迁移收益？"},
    "S3": {"architecture": "distillation", "question": "多教师是否优于单教师与等权对照？"},
    "S4": {"architecture": "latent", "question": "通道压缩是否节约资源且维持效果？"},
    "S5": {"architecture": "hierarchical", "question": "长序列分层是否优于局部窗口？"},
    "S6": {"architecture": "causal", "question": "严格因果输入能否满足实时约束？"},
    "S7": {"architecture": "applicability", "question": "已知任务直连，未知输入是否需要拒识？"},
}


def decide(evidence):
    """Expected paired rows are subject x seed; bootstrap resamples subjects.

    Thresholds belong to a frozen protocol, not a universal benefit criterion.
    Validation-only conclusions are exploratory model selection, not final claims.
    """
    result = {}
    for branch, specification in BRANCHES.items():
        e = evidence.get(branch, {})
        verdict = dict(**specification, status="pending", reason="尚无真实验证证据")
        if e.get("source") in {"test", "leaderboard"}:
            raise ValueError("Final test/leaderboard cannot drive pruning")
        if e.get("rules_allowed") is False:
            verdict.update(status="prune", reason="明确的赛规限制", evidence=e.get("rule_source"))
            if not e.get("rule_source"):
                raise ValueError("Rule pruning needs a source/version")
        elif e.get("resource_measured") and e.get("limits"):
            violations = [k for k, limit in e["limits"].items()
                          if e.get("resource", {}).get(k, -1) > limit]
            if violations:
                verdict.update(status="prune", reason="实测超出资源限制", violations=violations)
        if verdict["status"] == "pending" and e.get("source") == "validation" and e.get("real_data") is True:
            required = {"protocol_id", "paired", "direction", "min_gain", "max_guard_regression", "guards", "frozen_before_runs"}
            if not required <= e.keys() or not e["frozen_before_runs"]:
                verdict["reason"] = "缺少预先冻结的配对协议或门槛"
            else:
                if e["min_gain"] < 0 or e["max_guard_regression"] < 0:
                    raise ValueError("Evidence thresholds must be nonnegative")
                regressions = []
                for guard in e["guards"].values():
                    if guard["direction"] not in {"min", "max"}:
                        raise ValueError("Guard metric direction must be min/max")
                    sign = 1 if guard["direction"] == "max" else -1
                    regression = sign * (guard["baseline"] - guard["candidate"])
                    if not np.isfinite(regression):
                        raise ValueError("Non-finite guard evidence")
                    regressions.append(regression)
                if any(r > e["max_guard_regression"] for r in regressions):
                    verdict.update(status="prune", reason="关键任务回退超过预设阈值")
                    result[branch] = verdict
                    continue
                sign = 1 if e["direction"] == "max" else -1 if e["direction"] == "min" else None
                if sign is None:
                    raise ValueError("Metric direction must be min/max")
                groups, seen = {}, set()
                for row in e["paired"]:
                    key = (row["subject"], row["seed"])
                    if key in seen:
                        raise ValueError("Duplicate subject/seed evidence")
                    seen.add(key)
                    delta = sign * (row["candidate"] - row["baseline"])
                    if not np.isfinite(delta):
                        raise ValueError("Non-finite evidence")
                    groups.setdefault(row["subject"], []).append(delta)
                delta = np.array([np.mean(v) for v in groups.values()])
                if len(delta) < 2:
                    verdict["reason"] = "独立受试者不足，不能估计主体差异"
                else:
                    rng = np.random.default_rng(0)
                    bootstrap = rng.choice(delta, (5000, len(delta))).mean(-1)
                    lo, hi = np.quantile(bootstrap, [.025, .975])
                    verdict.update(mean_gain=float(delta.mean()), interval=[float(lo), float(hi)],
                                   subjects=len(delta), protocol_id=e["protocol_id"])
                    if lo > e["min_gain"]:
                        verdict.update(status="retain", reason="配对验证收益达到预设门槛")
                    elif hi <= e["min_gain"]:
                        verdict.update(status="prune", reason="当前预算下验证收益未达到预设门槛")
                    else:
                        verdict["reason"] = "收益区间跨越门槛，需要补证据"
        result[branch] = verdict
    return result
