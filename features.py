"""
features.py — виділення ознак за методикою статті
«Online ECG Biometrics via Hadamard Code» (Wang et al., ICASSP 2022).

Етапи:
  1) детекція R-піків алгоритмом Пана–Томпкінса [19];
  2) сегментація серцевих циклів фіксованою кількістю відліків з кожного боку
     R-піку (у статті — 260 відліків для MIT-BIH);
  3) базові ознаки 1DMRLBP — одновимірні багатороздільні локальні бінарні
     шаблони [7] (гістограми кодів у часових вікнах для кількох радіусів);
  4) онлайн-навчання дискримінативних представлень: матрична факторизація +
     коди Адамара (OEB-HC), рівняння (1)–(10) статті, з інкрементним
     оновленням накопичувачів C1, C2, C3.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import signal as sps
from scipy.linalg import hadamard

# ---------------------------------------------------------------------------
# 1. Детектор R-піків Пана–Томпкінса
# ---------------------------------------------------------------------------


def pan_tompkins(x: np.ndarray, fs: float, return_stages: bool = False):
    """Детекція QRS за Pan & Tompkins (1985).

    Етапи: смуговий фільтр 5–15 Гц → похідна → піднесення до квадрату →
    ковзне інтегрування (150 мс) → адаптивні пороги SPKI/NPKI з
    пошуком назад (search-back) та відсіюванням T-хвиль.
    """
    x = np.asarray(x, dtype=float)
    sos = sps.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band", output="sos")
    bp = sps.sosfiltfilt(sos, x)
    deriv = np.gradient(bp) * fs
    sq = deriv ** 2
    win = max(int(0.150 * fs), 1)
    mwi = np.convolve(sq, np.ones(win) / win, mode="same")

    cand, _ = sps.find_peaks(mwi, distance=int(0.2 * fs))
    if len(cand) == 0:
        return (np.array([], dtype=int), {}) if return_stages else np.array([], dtype=int)

    # Фаза навчання: перші 2 с
    init = mwi[: int(2 * fs)]
    spki, npki = 0.25 * np.max(init), 0.5 * np.mean(init)
    thr1 = npki + 0.25 * (spki - npki)

    qrs: list[int] = []
    slopes: list[float] = []
    rr_hist: list[int] = []
    for p in cand:
        v = mwi[p]
        # Пошук назад, якщо довго не було QRS (> 1.66 середнього RR)
        if qrs and rr_hist:
            rr_avg = np.mean(rr_hist[-8:])
            if p - qrs[-1] > 1.66 * rr_avg:
                lo, hi = qrs[-1] + int(0.2 * fs), p - int(0.2 * fs)
                sb = [c for c in cand if lo < c < hi and mwi[c] > 0.5 * thr1]
                if sb:
                    best = max(sb, key=lambda c: mwi[c])
                    rr_hist.append(best - qrs[-1])
                    qrs.append(best)
                    slopes.append(np.max(np.abs(deriv[max(best - win, 0):best + 1])))
                    spki = 0.25 * mwi[best] + 0.75 * spki
        if v > thr1:
            slope = np.max(np.abs(deriv[max(p - win, 0):p + 1]))
            # Відсіювання T-хвилі: близько до попереднього QRS і менший нахил
            if qrs and (p - qrs[-1]) < 0.36 * fs and slopes and slope < 0.5 * slopes[-1]:
                npki = 0.125 * v + 0.875 * npki
            else:
                if qrs:
                    rr_hist.append(p - qrs[-1])
                qrs.append(p)
                slopes.append(slope)
                spki = 0.125 * v + 0.875 * spki
        else:
            npki = 0.125 * v + 0.875 * npki
        thr1 = npki + 0.25 * (spki - npki)

    # Уточнення положення R-піку на вхідному сигналі (±75 мс)
    half = int(0.075 * fs)
    refined = []
    for p in qrs:
        lo, hi = max(p - half, 0), min(p + half + 1, len(x))
        seg = x[lo:hi]
        refined.append(lo + int(np.argmax(np.abs(seg - np.median(seg)))))
    peaks = np.unique(np.array(refined, dtype=int))
    # прибрати дублікати ближчі за 200 мс
    if len(peaks) > 1:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= int(0.2 * fs):
                keep.append(p)
        peaks = np.array(keep, dtype=int)
    if return_stages:
        return peaks, {"bandpass": bp, "derivative": deriv, "squared": sq, "integrated": mwi}
    return peaks


# ---------------------------------------------------------------------------
# 2. Сегментація серцевих циклів
# ---------------------------------------------------------------------------


def segment_heartbeats(x: np.ndarray, r_peaks: np.ndarray, pre: int, post: int):
    """Вирізати цикли [R - pre, R + post). Повертає (матриця циклів, R-піки)."""
    beats, used = [], []
    for r in r_peaks:
        if r - pre >= 0 and r + post <= len(x):
            beats.append(x[r - pre:r + post])
            used.append(r)
    if not beats:
        return np.zeros((0, pre + post)), np.array([], dtype=int)
    return np.vstack(beats), np.array(used, dtype=int)


# ---------------------------------------------------------------------------
# 3. Базові ознаки 1DMRLBP
# ---------------------------------------------------------------------------

FEATURE_TYPES = {
    "1dmrlbp": "1DMRLBP (як у статті)",
    "waveform": "Форма циклу (амплітуди)",
    "both": "1DMRLBP + форма циклу",
}


@dataclass
class FeatureParams:
    """Параметри сегментації та ознак."""
    pre: int = 90                    # відліків до R-піку
    post: int = 170                  # відліків після R-піку (разом 260, як у статті)
    feature_type: str = "1dmrlbp"
    lbp_p: int = 2                   # сусідів з кожного боку (код має 2P біт)
    radii: tuple = (1, 3, 6, 12)     # радіуси (багатороздільність)
    n_windows: int = 10              # часових вікон для гістограм
    threshold: float = 0.03          # поріг порівняння, мВ (стійкість до шуму)
    hellinger: bool = True           # корінь із гістограм (ядро Хеллінгера)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["radii"] = ", ".join(map(str, self.radii))
        return d


def lbp_1d_codes(beat: np.ndarray, p: int, radius: int, threshold: float = 0.0) -> np.ndarray:
    """Коди 1D-LBP: для кожного відліку порівнюємо p сусідів зліва та справа
    (крок = radius) з центральним значенням; біт = 1, якщо сусід ≥ центр + поріг."""
    beat = np.asarray(beat, dtype=float)
    reach = p * radius
    n = len(beat)
    if n <= 2 * reach:
        return np.zeros(0, dtype=int)
    idx = np.arange(reach, n - reach)
    center = beat[idx]
    code = np.zeros(len(idx), dtype=int)
    bit = 0
    for k in range(1, p + 1):
        for side in (-1, 1):
            code |= ((beat[idx + side * k * radius] - center) >= threshold).astype(int) << bit
            bit += 1
    return code


def mrlbp_1d(beat: np.ndarray, fp: FeatureParams) -> np.ndarray:
    """Вектор 1DMRLBP одного циклу: конкатенація нормованих гістограм кодів
    у n_windows часових вікнах для кожного радіуса."""
    n_bins = 2 ** (2 * fp.lbp_p)
    feats = []
    for r in fp.radii:
        codes = lbp_1d_codes(beat, fp.lbp_p, int(r), fp.threshold)
        for part in np.array_split(codes, fp.n_windows):
            h = np.bincount(part, minlength=n_bins).astype(float)
            feats.append(h / max(len(part), 1))
    v = np.concatenate(feats)
    return np.sqrt(v) if fp.hellinger else v


def _l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def extract_features(beats: np.ndarray, fp: FeatureParams) -> np.ndarray:
    """Матриця базових ознак (рядок = цикл), кожен рядок L2-нормовано."""
    if len(beats) == 0:
        return np.zeros((0, feature_dim(fp)))
    parts = []
    if fp.feature_type in ("1dmrlbp", "both"):
        parts.append(_l2(np.vstack([mrlbp_1d(b, fp) for b in beats])))
    if fp.feature_type in ("waveform", "both"):
        wf = beats - beats.mean(axis=1, keepdims=True)
        parts.append(_l2(wf))
    return _l2(np.hstack(parts))


def feature_dim(fp: FeatureParams) -> int:
    d = 0
    if fp.feature_type in ("1dmrlbp", "both"):
        d += 2 ** (2 * fp.lbp_p) * len(fp.radii) * fp.n_windows
    if fp.feature_type in ("waveform", "both"):
        d += fp.pre + fp.post
    return d


# ---------------------------------------------------------------------------
# 4. Онлайн-модель з кодами Адамара (OEB-HC)
# ---------------------------------------------------------------------------


@dataclass
class ModelParams:
    """Параметри моделі та протоколу (типові значення — зі статті)."""
    r: int = 128            # розмірність представлення = порядок матриці Адамара (2^7)
    alpha: float = 1.0
    beta: float = 1.0
    theta: float = 1.0
    delta: float = 1.0
    n_iter: int = 6         # кількість ітерацій на раунд
    n_chunks: int = 5       # кількість порцій даних (раундів)
    n_test: int = 5         # тестових циклів на особу
    n_train: int = 7        # навчальних циклів на особу (≈329/47 у статті); 0 — усі
    seed: int = 0
    enroll_mode: str = "mean_feature"  # шаблон: середнє ознак або ознаки середнього циклу

    def to_dict(self) -> dict:
        return asdict(self)


class OnlineHadamardMF:
    """Online ECG Biometrics via Hadamard Code.

    Позначення як у статті: X ∈ R^{m×n} (стовпці — зразки), V ∈ R^{n×r},
    W, G ∈ R^{m×r}, H — коди Адамара (рядок = код зразка).
    Накопичувачі між раундами:
        C1 = Σ X Xᵀ,  C2 = Σ X V,  C3 = Σ Vᵀ V.
    """

    def __init__(self, m: int, r: int = 128, alpha: float = 1.0, beta: float = 1.0,
                 theta: float = 1.0, delta: float = 1.0, n_iter: int = 6, seed: int = 0):
        if r & (r - 1):
            raise ValueError("r має бути степенем двійки (порядок матриці Адамара).")
        self.m, self.r = m, r
        self.alpha, self.beta, self.theta, self.delta = alpha, beta, theta, delta
        self.n_iter = n_iter
        # Матриця Адамара (конструкція Сильвестра); стовпці ортогональні
        self.H = hadamard(r).astype(float)
        rng = np.random.default_rng(seed)
        self._free_cols = list(rng.permutation(r))   # випадковий порядок без повторів
        self.code_of: dict = {}
        self.C1 = np.zeros((m, m))
        self.C2 = np.zeros((m, r))
        self.C3 = np.zeros((r, r))
        self.W = np.zeros((m, r))
        self.G = np.zeros((m, r))
        self.n_seen = 0
        self.loss_history: list[list[float]] = []

    # --- коди Адамара ---
    def codes_for(self, labels) -> np.ndarray:
        """Призначити (за потреби) нові стовпці матриці Адамара новим особам."""
        rows = []
        for lab in labels:
            if lab not in self.code_of:
                if not self._free_cols:
                    raise ValueError(f"Осіб більше, ніж r={self.r}: збільште r.")
                self.code_of[lab] = self._free_cols.pop(0)
            rows.append(self.H[:, self.code_of[lab]])
        return np.vstack(rows)

    def _objective(self, X, V, Hn, W, G) -> float:
        o1 = self.beta * np.linalg.norm(X - G @ V.T) ** 2
        o2 = self.alpha * np.linalg.norm(V - X.T @ W) ** 2
        o3 = self.theta * np.linalg.norm(Hn - V) ** 2
        reg = self.delta * (np.linalg.norm(W) ** 2 + np.linalg.norm(G) ** 2)
        return float(o1 + o2 + o3 + reg)

    def partial_fit(self, samples: np.ndarray, labels) -> "OnlineHadamardMF":
        """Один раунд онлайн-навчання на новій порції (рядки samples — зразки)."""
        X = np.asarray(samples, dtype=float).T          # m × n_t
        Hn = self.codes_for(labels)                      # n_t × r
        a, b, th, d = self.alpha, self.beta, self.theta, self.delta
        I_m, I_r = np.eye(self.m), np.eye(self.r)

        C1 = self.C1 + X @ X.T                           # рівняння (7)
        V = Hn.copy()                                    # ініціалізація V кодами
        losses = []
        for _ in range(self.n_iter):
            C2 = self.C2 + X @ V
            W = np.linalg.solve(C1 + (d / a) * I_m, C2)                  # (6)
            C3 = self.C3 + V.T @ V
            G = np.linalg.solve((C3 + (d / b) * I_r).T, C2.T).T          # (8)
            rhs = a * X.T @ W + b * X.T @ G + th * Hn
            V = np.linalg.solve(((a + th) * I_r + b * G.T @ G).T, rhs.T).T  # (10)
            losses.append(self._objective(X, V, Hn, W, G))
        # Фіксація знань раунду в накопичувачах
        self.C1 = C1
        self.C2 = self.C2 + X @ V
        self.C3 = self.C3 + V.T @ V
        self.W = np.linalg.solve(self.C1 + (d / a) * I_m, self.C2)
        self.G = self.C2 @ np.linalg.inv(self.C3 + (d / b) * I_r)
        self.n_seen += X.shape[1]
        self.loss_history.append(losses)
        return self

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Проєкція зразків у латентний простір: Xᵀ W."""
        return np.asarray(samples, dtype=float) @ self.W


def pca_2d(z: np.ndarray) -> np.ndarray:
    """Проєкція на дві головні компоненти (для візуалізації)."""
    zc = z - z.mean(axis=0)
    u, s, _ = np.linalg.svd(zc, full_matrices=False)
    return u[:, :2] * s[:2]
