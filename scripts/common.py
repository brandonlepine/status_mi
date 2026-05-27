"""Shared utilities for the identity-geometry and SAE-feature analysis scripts.

Audit issue 5.10 fix. Prior to this module, `cohens_d`, `compute_direction`
(under various names), `evaluate_projection` (under various names),
`residualize`, `normalize`, `cosine`, the Okabe-Ito palette, `save_fig`, and
`CenterOnlyScaler` were re-implemented in 5-8 scripts each. Each copy could
drift independently — silent breaking-change risk on every commit.

This module is the single source of truth. Every analysis and plot script in
`scripts/` should import from here rather than re-defining.

Conventions:
- All effect-size / direction math operates on 2-D arrays `x` of shape
  (n_rows, hidden_dim).
- Masks are boolean numpy arrays the same length as `x`.
- `compute_direction` and `compute_direction_for_pair` always return a
  `ContrastDirection` dataclass; downstream code should unpack `.direction`,
  `.global_mean`, `.sign_flipped` rather than relying on tuple indexing so
  future additions don't break callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Effect sizes and projections
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d for two 1-D score vectors. Returns NaN when
    either group has fewer than two samples or the pooled variance is
    nonpositive."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled = (((n_a - 1) * np.var(a, ddof=1)) + ((n_b - 1) * np.var(b, ddof=1))) / (n_a + n_b - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Vector cosine. Returns NaN when either vector has zero norm."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """Unit-normalize a vector. Returns None if the input has near-zero norm."""
    arr = np.asarray(vec)
    norm = float(np.linalg.norm(arr))
    if norm <= eps or not np.isfinite(norm):
        return None
    return arr / norm


# ---------------------------------------------------------------------------
# Contrast directions
# ---------------------------------------------------------------------------

@dataclass
class ContrastDirection:
    """Result of computing a difference-of-means contrast direction.

    Attributes
    ----------
    direction : np.ndarray
        Unit-normed vector. Sign-flipped so that projecting onto it gives a
        higher mean for group A than group B.
    global_mean : np.ndarray | None
        The mean used to center `x` (shape `(1, hidden_dim)`). `None` when
        the caller passed pre-centered data via `center=False`.
    sign_flipped : bool
        `True` if `mean(scores_a) < mean(scores_b)` before sign correction;
        downstream callers can use this to know whether `direction` points
        from B → A or A → B in the raw activation space.
    """
    direction: np.ndarray
    global_mean: np.ndarray | None
    sign_flipped: bool


def compute_direction(
    x: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    center: bool = True,
    center_mean: np.ndarray | None = None,
) -> ContrastDirection | None:
    """Low-level difference-of-means contrast direction.

    Parameters
    ----------
    x : np.ndarray
        Activations, shape (n_rows, hidden_dim). Can be already-centered;
        if so, pass `center=False`.
    mask_a, mask_b : np.ndarray
        Boolean masks selecting the two identity groups.
    center : bool, default True
        When True, subtract the global mean (over all rows of `x`) before
        computing the difference. When False, `x` is assumed pre-centered.
    center_mean : np.ndarray, optional
        If provided (and `center=True`), use this as the centering mean
        instead of computing `x.mean(axis=0)`. Useful for held-out splits
        where the centering should use train-set statistics. Ignored when
        `center=False`.

    Returns
    -------
    ContrastDirection or None
        `None` when either group is empty or the difference-of-means has
        zero norm.
    """
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None
    if center:
        global_mean = center_mean if center_mean is not None else x.mean(axis=0, keepdims=True)
        centered = x - global_mean
    else:
        global_mean = None
        centered = x
    raw = centered[mask_a].mean(axis=0) - centered[mask_b].mean(axis=0)
    unit = normalize(raw)
    if unit is None:
        return None
    sign_flipped = False
    scores = centered @ unit
    if scores[mask_a].mean() < scores[mask_b].mean():
        unit = -unit
        sign_flipped = True
    return ContrastDirection(direction=unit.astype(np.float32), global_mean=global_mean, sign_flipped=sign_flipped)


def compute_direction_for_pair(
    x: np.ndarray,
    metadata: pd.DataFrame,
    identity_a: str,
    identity_b: str,
    *,
    center: bool = True,
    identity_col: str = "identity_id",
) -> ContrastDirection | None:
    """Convenience wrapper around `compute_direction` that builds the masks
    from a metadata DataFrame and two identity_ids. Equivalent semantics."""
    if identity_col not in metadata.columns:
        raise KeyError(f"metadata is missing column '{identity_col}'")
    mask_a = metadata[identity_col].eq(identity_a).to_numpy()
    mask_b = metadata[identity_col].eq(identity_b).to_numpy()
    return compute_direction(x, mask_a, mask_b, center=center)


# ---------------------------------------------------------------------------
# Projection evaluation
# ---------------------------------------------------------------------------

def evaluate_projection(
    scores: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate already-computed projection scores against two group masks.

    Returns the same dict shape used by `evaluate_component` in
    `analyze_shared_social_subspace.py`:

        {
            'auc': float (NaN if degenerate),
            'cohens_d': float (NaN if degenerate),
            'accuracy_midpoint': float (NaN if degenerate),
            'mean_a': float, 'mean_b': float,
            'sd_a': float, 'sd_b': float,
            'n_a': int, 'n_b': int,
        }
    """
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    base: dict[str, float | int] = {
        "auc": float("nan"),
        "cohens_d": float("nan"),
        "accuracy_midpoint": float("nan"),
        "mean_a": float("nan"),
        "mean_b": float("nan"),
        "sd_a": float("nan"),
        "sd_b": float("nan"),
        "n_a": n_a,
        "n_b": n_b,
    }
    if n_a == 0 or n_b == 0:
        return base
    scores = np.asarray(scores).ravel()
    scores_a = scores[mask_a]
    scores_b = scores[mask_b]
    pair_scores = np.concatenate([scores_a, scores_b])
    labels = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
    auc = float(roc_auc_score(labels, pair_scores)) if len(np.unique(labels)) == 2 else float("nan")
    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())
    midpoint = (mean_a + mean_b) / 2
    accuracy = float(np.mean(np.concatenate([scores_a >= midpoint, scores_b < midpoint])))
    base.update({
        "auc": auc,
        "cohens_d": cohens_d(scores_a, scores_b),
        "accuracy_midpoint": accuracy,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sd_a": float(scores_a.std(ddof=1)) if len(scores_a) > 1 else 0.0,
        "sd_b": float(scores_b.std(ddof=1)) if len(scores_b) > 1 else 0.0,
    })
    return base


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------

def residualize(
    x: np.ndarray,
    metadata: pd.DataFrame,
    group_col: str | None,
) -> np.ndarray:
    """Within-group centering, re-adding the global mean so the output has
    the same overall mean as the input. `group_col=None` is identity (no
    residualization)."""
    if group_col is None:
        return x
    global_mean = x.mean(axis=0, keepdims=True)
    out = x.copy()
    for _, idx in metadata.groupby(group_col, sort=True).groups.items():
        idx_array = np.fromiter(idx, dtype=int)
        out[idx_array] = x[idx_array] - x[idx_array].mean(axis=0, keepdims=True) + global_mean
    return out


# ---------------------------------------------------------------------------
# Scalers (--scaling flag; audit 5.9)
# ---------------------------------------------------------------------------

SCALING_MODES = ("standardize", "center_only")
DEFAULT_SCALING = "center_only"


class CenterOnlyScaler:
    """Drop-in replacement for sklearn StandardScaler that subtracts the
    per-dim mean only (no z-scoring).

    Audit 5.9: Llama residual-stream dimensions carry meaningfully unequal
    scale (rogue / high-norm dimensions carry real signal); z-scoring
    upweights low-variance dimensions and changes what
    `explained_variance_ratio` is measuring. `center_only` preserves the
    activation-space variance structure.
    """
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "CenterOnlyScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("CenterOnlyScaler.transform called before fit.")
        return x - self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def make_scaler(mode: str):
    """Factory for the --scaling flag. Audit 5.9."""
    if mode == "standardize":
        return StandardScaler()
    if mode == "center_only":
        return CenterOnlyScaler()
    raise ValueError(f"Unknown scaling mode: {mode!r}; expected one of {SCALING_MODES}.")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

OKABE_ITO: list[str] = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]


def okabe_ito_palette(labels: Iterable[str]) -> dict[str, str]:
    """Map an ordered iterable of categorical labels to Okabe-Ito colors,
    cycling once the palette is exhausted."""
    return {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(labels)}


def save_fig(
    fig,
    path_no_suffix: Path,
    *,
    dpi: int = 200,
    bbox_inches: str | None = "tight",
    tight_layout: bool = False,
) -> None:
    """Save a matplotlib figure as both PNG and PDF next to one another,
    using the same basename. Closes the figure after writing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_no_suffix : Path | str
        Basename without an extension. Any `.png` or `.pdf` suffix on the
        input is stripped before adding both.
    dpi : int, default 200
    bbox_inches : str or None, default "tight"
        Pass through to `fig.savefig`. Set to `None` to suppress.
    tight_layout : bool, default False
        If True, call `fig.tight_layout()` before saving. Some callers
        prefer this over `bbox_inches="tight"` for cleaner panel spacing.
    """
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required for save_fig.")
    if tight_layout:
        fig.tight_layout()
    path_no_suffix = Path(str(path_no_suffix))
    if path_no_suffix.suffix.lower() in {".png", ".pdf"}:
        path_no_suffix = path_no_suffix.with_suffix("")
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"dpi": dpi}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    fig.savefig(path_no_suffix.with_suffix(".png"), **kwargs)
    save_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    fig.savefig(path_no_suffix.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
