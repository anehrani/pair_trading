from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from copulae import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentCopula


@dataclass(frozen=True)
class FittedMarginal:
    dist: object
    params: tuple

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        d = self.dist
        return np.asarray(d.cdf(x, *self.params), dtype=float)


@dataclass(frozen=True)
class FittedCopula:
    name: str
    copula: object
    aic: float


class RotatedCopula:
    """Wrapper implementing rotated copulas (paper Eq. 16).

    C90(u1,u2)  := C(1-u2, u1)
    C180(u1,u2) := C(1-u1, 1-u2)
    C270(u1,u2) := C(u2, 1-u1)

    This wrapper exposes cdf/pdf that accept an (n,2) array.
    """

    def __init__(self, base: object, rotation: int):
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be one of {0,90,180,270}")
        self.base = base
        self.rotation = rotation

    def _transform(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be shape (n,2)")
        u1 = u[:, 0]
        u2 = u[:, 1]
        if self.rotation == 0:
            out = np.column_stack([u1, u2])
        elif self.rotation == 90:
            out = np.column_stack([1 - u2, u1])
        elif self.rotation == 180:
            out = np.column_stack([1 - u1, 1 - u2])
        else:  # 270
            out = np.column_stack([u2, 1 - u1])
        return _safe_unit_interval(out)

    def cdf(self, u: np.ndarray) -> np.ndarray:
        return self.base.cdf(self._transform(u))

    def pdf(self, u: np.ndarray) -> np.ndarray:
        return self.base.pdf(self._transform(u))


def fit_best_marginal(spread: np.ndarray) -> FittedMarginal:
    """Fit Gaussian / Student-t / Cauchy and choose by AIC (paper's approach)."""
    candidates = [stats.norm, stats.t, stats.cauchy]
    best_aic = float("inf")
    best = None

    spread = np.asarray(spread, dtype=float)
    spread = spread[np.isfinite(spread)]
    if spread.size < 50:
        raise ValueError("Not enough spread samples to fit marginals")

    for d in candidates:
        params = d.fit(spread)
        pdf = d.pdf(spread, *params)
        # Guard log(0)
        pdf = np.clip(pdf, 1e-300, None)
        log_lik = float(np.sum(np.log(pdf)))
        k = len(params)
        aic = 2 * k - 2 * log_lik
        if aic < best_aic:
            best_aic = aic
            best = (d, params)

    assert best is not None
    return FittedMarginal(dist=best[0], params=tuple(best[1]))


def _safe_unit_interval(u: np.ndarray) -> np.ndarray:
    # Copulas can be numerically unhappy at exactly 0 or 1.
    eps = 1e-10
    return np.clip(u, eps, 1 - eps)


def fit_copula_candidates(u: np.ndarray) -> list[FittedCopula]:
    """Fit a small set of copulas available in copulae and compute AIC.

    The paper evaluates many families (incl. BB*, Tawn). Here we implement a
    practical subset we can fit robustly with copulae out of the box.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be an array of shape (n, 2)")
    u = _safe_unit_interval(u)

    base_candidates: list[tuple[str, object]] = [
        ("gaussian", GaussianCopula(dim=2)),
        ("student", StudentCopula(dim=2)),
        ("clayton", ClaytonCopula(dim=2)),
        ("gumbel", GumbelCopula(dim=2)),
        ("frank", FrankCopula(dim=2)),
    ]

    # Rotations can capture negative dependence / different tail structure.
    rotations = (0, 90, 180, 270)

    candidates: list[tuple[str, object]] = []
    for name, base in base_candidates:
        for rot in rotations:
            if rot == 0:
                candidates.append((name, base.__class__(dim=2)))
            else:
                candidates.append((f"{name}_rot{rot}", RotatedCopula(base.__class__(dim=2), rot)))

    fitted: list[FittedCopula] = []

    def _log_lik_value(cop: object, u_: np.ndarray) -> float:
        """Return a scalar log-likelihood for a fitted copula.

        copulae sometimes exposes `log_lik` as a scalar or a vector; in either
        case we convert to a scalar by summing.
        """
        ll_attr = getattr(cop, "log_lik", None)
        ll = None
        if ll_attr is not None:
            if callable(ll_attr):
                try:
                    ll = ll_attr()
                except Exception:
                    ll = None
            else:
                ll = ll_attr

        if ll is not None:
            arr = np.asarray(ll, dtype=float)
            return float(arr) if arr.ndim == 0 else float(np.nansum(arr))

        pdf = np.asarray(cop.pdf(u_), dtype=float)
        pdf = np.clip(pdf, 1e-300, None)
        return float(np.sum(np.log(pdf)))

    for name, c in candidates:
        try:
            # RotatedCopula wraps an underlying copula that supports fit()
            if isinstance(c, RotatedCopula):
                c.base.fit(u)
            else:
                c.fit(u)

            # Many copulae objects expose log_lik; fall back to log(pdf) sum.
            if isinstance(c, RotatedCopula):
                log_lik = _log_lik_value(c.base, c._transform(u))
            else:
                log_lik = _log_lik_value(c, u)

            # Parameter count: use len(params) if present, else 1.
            k = 1
            params_obj = None
            if isinstance(c, RotatedCopula):
                params_obj = getattr(c.base, "params", None)
            else:
                params_obj = getattr(c, "params", None)

            if params_obj is not None:
                try:
                    k = int(np.size(params_obj))
                except Exception:
                    k = 1

            aic = 2 * k - 2 * log_lik
            fitted.append(FittedCopula(name=name, copula=c, aic=aic))
        except Exception:
            continue

    if not fitted:
        raise ValueError("No copula candidates could be fit")
    return sorted(fitted, key=lambda x: x.aic)


def h_functions_numerical(copula: object, u1: float, u2: float, *, delta: float = 1e-5) -> tuple[float, float]:
    """Numerically approximate h_{1|2} and h_{2|1} from Eq. (4).

    h_{1|2} = ∂C(u1,u2)/∂u2
    h_{2|1} = ∂C(u1,u2)/∂u1

    Uses central differences on the copula CDF.
    """
    u1 = float(np.clip(u1, 1e-10, 1 - 1e-10))
    u2 = float(np.clip(u2, 1e-10, 1 - 1e-10))

    def cdf(a: float, b: float) -> float:
        arr = np.array([[a, b]], dtype=float)
        val = copula.cdf(arr)
        # copulae returns array-like
        return float(np.asarray(val).reshape(-1)[0])

    u2p = min(1 - 1e-10, u2 + delta)
    u2m = max(1e-10, u2 - delta)
    u1p = min(1 - 1e-10, u1 + delta)
    u1m = max(1e-10, u1 - delta)

    h1_2 = (cdf(u1, u2p) - cdf(u1, u2m)) / (u2p - u2m)
    h2_1 = (cdf(u1p, u2) - cdf(u1m, u2)) / (u1p - u1m)

    # h-functions are conditional CDF values, should be in [0,1]
    h1_2 = float(np.clip(h1_2, 0.0, 1.0))
    h2_1 = float(np.clip(h2_1, 0.0, 1.0))
    return h1_2, h2_1
