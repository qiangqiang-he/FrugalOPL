"""Streaming statistics for the rank of an actually sampled vocabulary token.

No top-k approximation: H = {sample} union {j: logit[j] > logit[sample]}.
Ties other than the sampled token belong to T. All moments are token-weighted.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

METRICS = (
    "head_count", "tail_count", "head_mass", "tail_mass",
    "head_token_mean_probability", "tail_token_mean_probability",
    "sampled_probability", "sampled_surprisal_nats", "entropy_nats",
    "head_entropy_contribution_nats", "tail_entropy_contribution_nats",
    "head_conditional_entropy_nats", "tail_conditional_entropy_nats",
    "top1_probability", "tied_other_count", "head_fraction", "tail_fraction",
    "entropy_effective_tokens",
)
QUANTILES = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1)
POSITION_EDGES = (1, 16, 64, 256, 1024, 2048, 4096, 8192)
LENGTH_EDGES = (16, 64, 256, 1024, 2048, 4096, 8192)


@torch.inference_mode()
def sampled_partition_metrics(logits: torch.Tensor, sampled_ids: torch.Tensor) -> torch.Tensor:
    """Return [rows, len(METRICS)] from the complete, unpadded model vocabulary.

    Caller supplies the unmodified T=1 logits used by sampling, and detached IDs.
    Tail mass is summed directly, never obtained by subtracting from one.
    Empty-tail averages/conditional entropies are NaN and omitted from moments.
    """
    logits = logits.float()
    ids = sampled_ids.long().reshape(-1, 1)
    sample_logits = logits.gather(1, ids)
    head = logits > sample_logits
    head.scatter_(1, ids, True)
    tail = ~head
    logp = logits.log_softmax(-1)
    p = logp.exp()
    hc = head.sum(-1).float()
    tc = logits.shape[-1] - hc
    # FP32 reduction can overshoot one by an ulp; retain physical bounds.
    hm = (p * head).sum(-1).clamp(0, 1)
    tm = (p * tail).sum(-1).clamp(0, 1)
    entropy_terms = torch.where(p > 0, -p * logp, 0.)
    he = (entropy_terms * head).sum(-1)
    te = (entropy_terms * tail).sum(-1)
    nan = torch.full_like(tm, float("nan"))
    hent = he / hm + hm.log()
    tent = torch.where(tm > 0, te / tm + tm.log(), nan)
    result = (
        hc, tc, hm, tm, hm / hc, torch.where(tc > 0, tm / tc, nan),
        p.gather(1, ids).squeeze(1), -logp.gather(1, ids).squeeze(1),
        he + te, he, te, hent.clamp_min(0), tent.clamp_min(0),
        p.amax(-1), (logits == sample_logits).sum(-1).float() - 1,
        hc / logits.shape[-1], tc / logits.shape[-1], (he + te).exp().clamp_max(logits.shape[-1]),
    )
    return torch.stack(result, -1)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bucket(value: int, edges) -> str:
    previous = 0
    for edge in edges:
        if value <= edge:
            return f"{previous + 1}-{edge}"
        previous = edge
    return f"{previous + 1}+"


class Moments:
    def __init__(self):
        self.n = np.zeros(len(METRICS), np.int64)
        self.sum = np.zeros(len(METRICS), np.float64)
        self.sq = self.sum.copy()
        self.lo = np.full(len(METRICS), np.inf)
        self.hi = np.full(len(METRICS), -np.inf)

    def add(self, values):
        values = np.asarray(values, np.float64)
        if not len(values):
            return
        valid = np.isfinite(values)
        self.n += valid.sum(0)
        safe = np.where(valid, values, 0)
        self.sum += safe.sum(0)
        self.sq += np.square(safe).sum(0)
        self.lo = np.minimum(self.lo, np.where(valid, values, np.inf).min(0))
        self.hi = np.maximum(self.hi, np.where(valid, values, -np.inf).max(0))

    def merge(self, other):
        self.n += other.n
        self.sum += other.sum
        self.sq += other.sq
        self.lo = np.minimum(self.lo, other.lo)
        self.hi = np.maximum(self.hi, other.hi)

    def array(self):
        return np.stack([self.n, self.sum, self.sq, self.lo, self.hi])

    @classmethod
    def restore(cls, array):
        out = cls()
        out.n = array[0].astype(np.int64)
        out.sum, out.sq, out.lo, out.hi = (a.copy() for a in array[1:])
        return out

    def summary(self):
        out = {}
        for i, name in enumerate(METRICS):
            n = int(self.n[i])
            mean = float(self.sum[i] / n) if n else None
            out[name] = {"n": n, "mean": mean,
                         "std": math.sqrt(max(0, float(self.sq[i] / n) - mean ** 2)) if n else None,
                         "min": float(self.lo[i]) if n else None,
                         "max": float(self.hi[i]) if n else None}
        return out


class StreamingStats:
    """Bounded-memory moments + exact integer histograms + bounded float quantiles.

    Continuous quantiles use histogram intervals (no claim of exactness). Both
    linear and logarithmic histograms are kept for probabilities, using the
    tighter interval. Per-batch quantiles and retained raw chunks are exact.
    """
    def __init__(self, vocab_size: int, bins: int = 10000):
        self.vocab_size, self.bins = vocab_size, bins
        self.moments = Moments()
        self.strata: dict[str, Moments] = {}
        self.rollout_means = Moments()
        self.lengths = np.zeros(8193, np.int64)
        self.finishes: dict[str, int] = {}
        self.thresholds: dict[str, int] = {}
        self.hist = {}
        self.edges = {}
        for name in METRICS:
            if name.endswith("count"):
                self.hist[name] = np.zeros(vocab_size + 1, np.int64)
                continue
            if "probability" in name or name.endswith("mass") or name.endswith("fraction"):
                self.edges[name] = np.linspace(0, 1, bins + 1)
                if not name.endswith("fraction"):
                    self.edges[name + "__log"] = np.r_[0., np.geomspace(1e-45, 1., 20001)]
            else:
                high = vocab_size if name == "entropy_effective_tokens" else (
                    110 if name == "sampled_surprisal_nats" else math.log(vocab_size) + .01)
                self.edges[name] = np.linspace(0, high, bins + 1)
            self.hist[name] = np.zeros(len(self.edges[name]) - 1, np.int64)
            if name + "__log" in self.edges:
                self.hist[name + "__log"] = np.zeros(len(self.edges[name + "__log"]) - 1, np.int64)

    def add(self, values, positions, request_rows, rollouts):
        values = np.asarray(values, np.float64)
        self.moments.add(values)
        for i, name in enumerate(METRICS):
            x = values[:, i]
            x = x[np.isfinite(x)]
            if name.endswith("count"):
                self.hist[name] += np.bincount(x.astype(np.int64), minlength=self.vocab_size + 1)
            else:
                for key in (name, name + "__log"):
                    if key in self.edges:
                        edges = self.edges[key]
                        # Clip only rounding at valid physical bounds. Record extrema in moments.
                        if len(x) and (x.min() < -2e-5 or x.max() > edges[-1] + 2e-3):
                            raise ValueError(f"Out-of-range {name}: {x.min()}, {x.max()}")
                        self.hist[key] += np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0]
        thresholds = {
            **{f"head_count_le_{v}": values[:, 0] <= v for v in (1, 10, 100, 1000)},
            **{f"head_mass_ge_{v}": values[:, 2] >= v for v in (.9, .95, .99, .999)},
            "tail_empty": values[:, 1] == 0,
            "sampled_probability_lt_0.01": values[:, 6] < .01,
        }
        for name, mask in thresholds.items():
            self.thresholds[name] = self.thresholds.get(name, 0) + int(mask.sum())
        for edge in POSITION_EDGES:
            label = bucket(edge, POSITION_EDGES)
            lower, upper = map(int, label.split("-"))
            mask = (positions >= lower) & (positions <= upper)
            self.strata.setdefault("position/" + label, Moments()).add(values[mask])
        for row, rollout in enumerate(rollouts):
            mask = request_rows == row
            x = values[mask]
            if len(x) != rollout["length"]:
                raise ValueError("Every generated position must have exactly one measurement")
            length = rollout["length"]
            self.lengths[length] += 1
            finish = rollout["finish_reason"]
            self.finishes[finish] = self.finishes.get(finish, 0) + 1
            for key in ("finish/" + finish, "length/" + bucket(length, LENGTH_EDGES)):
                self.strata.setdefault(key, Moments()).add(x)
            n = np.isfinite(x).sum(0)
            mean = np.divide(np.nansum(x, axis=0), n, out=np.full(len(METRICS), np.nan), where=n > 0)
            self.rollout_means.add(mean[None, :])

    def merge(self, other):
        if (self.vocab_size, self.bins) != (other.vocab_size, other.bins):
            raise ValueError("Cannot merge different vocabulary/bin schemas")
        self.moments.merge(other.moments)
        self.rollout_means.merge(other.rollout_means)
        self.lengths += other.lengths
        for name, array in other.hist.items():
            self.hist[name] += array
        for key, value in other.strata.items():
            self.strata.setdefault(key, Moments()).merge(value)
        for attr in ("finishes", "thresholds"):
            target = getattr(self, attr)
            for key, n in getattr(other, attr).items():
                target[key] = target.get(key, 0) + n

    @staticmethod
    def integer_quantiles(hist):
        n = int(hist.sum())
        if not n:
            return {}
        cdf = hist.cumsum()
        result = {}
        for q in QUANTILES:
            rank = q * (n - 1)
            a = np.searchsorted(cdf, math.floor(rank) + 1)
            b = np.searchsorted(cdf, math.ceil(rank) + 1)
            result[str(q)] = float(a + (b - a) * (rank - math.floor(rank)))
        return result

    def float_quantiles(self, name):
        i = METRICS.index(name)
        n = int(self.moments.n[i])
        if not n:
            return {}
        result = {}
        for q in QUANTILES:
            intervals = []
            for key in (name, name + "__log"):
                if key not in self.hist:
                    continue
                cdf = self.hist[key].cumsum()
                rank = q * (n - 1)
                a = min(int(np.searchsorted(cdf, math.floor(rank) + 1)), len(cdf) - 1)
                b = min(int(np.searchsorted(cdf, math.ceil(rank) + 1)), len(cdf) - 1)
                f = rank - math.floor(rank)
                e = self.edges[key]
                lo, hi = (1-f)*e[a]+f*e[b], (1-f)*e[a+1]+f*e[b+1]
                intervals.append((lo, hi))
            lo = max(x[0] for x in intervals)
            hi = min(x[1] for x in intervals)
            lo, hi = max(lo, self.moments.lo[i]), min(hi, self.moments.hi[i])
            result[str(q)] = {"estimate": float((lo+hi)/2), "lower": float(lo), "upper": float(hi)}
        return result

    def summary(self):
        metrics = self.moments.summary()
        for name in METRICS:
            metrics[name]["quantiles"] = (self.integer_quantiles(self.hist[name])
                                          if name.endswith("count") else self.float_quantiles(name))
        n = int(self.moments.n[0])
        return {"positions": n, "rollouts": int(self.lengths.sum()), "vocab_size": self.vocab_size,
                "token_weighted": metrics, "rollout_weighted_means": self.rollout_means.summary(),
                "response_length": {"mean": float(self.lengths @ np.arange(len(self.lengths)) / self.lengths.sum()) if self.lengths.sum() else None,
                                    "quantiles": self.integer_quantiles(self.lengths)},
                "finish_counts": self.finishes,
                "fractions": {k: v/n for k, v in self.thresholds.items()} if n else {},
                "strata_token_weighted": {k: m.summary() for k, m in self.strata.items()},
                "quantile_method": {"counts": "exact numpy-linear quantiles of integer histograms",
                                    "continuous": "bounded histogram estimates; each quantile includes lower/upper bounds",
                                    "probability_linear_bin_width": 1/self.bins,
                                    "probability_log10_bin_width": 45/20000,
                                    "empty_tail": "undefined conditional quantities excluded; valid n reported"}}

    def save(self, path, cursor):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"moments": self.moments.array(), "rollout_means": self.rollout_means.array(), "lengths": self.lengths}
        arrays.update({"hist/" + key: val for key, val in self.hist.items()})
        arrays.update({"strata/" + key: val.array() for key, val in self.strata.items()})
        arrays["metadata"] = np.array(json.dumps({"vocab_size": self.vocab_size, "bins": self.bins,
                                                 "cursor": cursor, "finishes": self.finishes, "thresholds": self.thresholds}))
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            out = cls(meta["vocab_size"], meta["bins"])
            out.moments = Moments.restore(z["moments"])
            out.rollout_means = Moments.restore(z["rollout_means"])
            out.lengths = z["lengths"].copy()
            for name in z.files:
                if name.startswith("hist/"):
                    out.hist[name[5:]] = z[name].copy()
                elif name.startswith("strata/"):
                    out.strata[name[7:]] = Moments.restore(z[name])
            out.finishes, out.thresholds = meta["finishes"], meta["thresholds"]
        return out, meta["cursor"]
