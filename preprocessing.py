"""
preprocessing.py — попередня обробка ЕКГ: видалення дрейфу ізолінії,
режекторний фільтр мережевої наводки та смуговий фільтр Баттерворта.

У статті (Wang et al., ICASSP 2022) окремий етап фільтрації детально не
описано, тому тут реалізовано стандартний ланцюжок для MIT-BIH:
  1) видалення базової лінії (двоступеневий медіанний фільтр 200/600 мс
     або ФВЧ Баттерворта);
  2) режекторний фільтр 50/60 Гц;
  3) смуговий фільтр Баттерворта (типово 0.5–40 Гц) з нульовою фазою.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal as sps

BASELINE_METHODS = {
    "median": "Двоступеневий медіанний фільтр",
    "highpass": "ФВЧ Баттерворта",
    "none": "Без видалення",
}


@dataclass
class PreprocessParams:
    """Параметри попередньої обробки."""
    baseline_method: str = "median"
    median_win1_ms: int = 200     # перше вікно медіанного фільтра (QRS, P)
    median_win2_ms: int = 600     # друге вікно (T-хвиля)
    hp_cutoff: float = 0.5        # частота зрізу ФВЧ, Гц
    band_low: float = 0.5         # нижня межа смугового фільтра, Гц
    band_high: float = 40.0       # верхня межа смугового фільтра, Гц
    order: int = 4                # порядок фільтра Баттерворта
    notch: bool = True            # вмикати режекторний фільтр
    notch_freq: float = 60.0      # частота мережі, Гц
    notch_q: float = 30.0         # добротність режекторного фільтра

    def to_dict(self) -> dict:
        return asdict(self)


def _odd(n: int) -> int:
    n = max(int(n), 3)
    return n if n % 2 else n + 1


def estimate_baseline(x: np.ndarray, fs: float, p: PreprocessParams) -> np.ndarray:
    """Оцінка дрейфу ізолінії."""
    if p.baseline_method == "median":
        b1 = sps.medfilt(x, _odd(p.median_win1_ms * fs / 1000))
        return sps.medfilt(b1, _odd(p.median_win2_ms * fs / 1000))
    if p.baseline_method == "highpass":
        sos = sps.butter(2, p.hp_cutoff / (fs / 2), btype="low", output="sos")
        return sps.sosfiltfilt(sos, x)
    return np.zeros_like(x)


def remove_baseline(x: np.ndarray, fs: float, p: PreprocessParams) -> tuple[np.ndarray, np.ndarray]:
    """Відняти базову лінію. Повертає (сигнал без дрейфу, оцінка дрейфу)."""
    base = estimate_baseline(x, fs, p)
    return x - base, base


def notch_filter(x: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    """Режекторний IIR-фільтр (нульова фаза)."""
    if f0 >= fs / 2:
        return x
    b, a = sps.iirnotch(f0, q, fs)
    return sps.filtfilt(b, a, x)


def _band_sos(fs: float, p: PreprocessParams):
    nyq = fs / 2
    low = max(p.band_low, 0.01)
    high = min(p.band_high, nyq * 0.99)
    if low >= high:
        low = high / 2
    return sps.butter(p.order, [low / nyq, high / nyq], btype="band", output="sos")


def bandpass_filter(x: np.ndarray, fs: float, p: PreprocessParams) -> np.ndarray:
    """Смуговий фільтр Баттерворта (прямий і зворотний прохід — нульова фаза)."""
    return sps.sosfiltfilt(_band_sos(fs, p), x)


def preprocess(x: np.ndarray, fs: float, p: PreprocessParams) -> dict:
    """Повний ланцюжок обробки. Повертає словник з проміжними сигналами."""
    x = np.asarray(x, dtype=float)
    no_base, base = remove_baseline(x, fs, p)
    y = notch_filter(no_base, fs, p.notch_freq, p.notch_q) if p.notch else no_base
    y = bandpass_filter(y, fs, p)
    return {"raw": x, "baseline": base, "no_baseline": no_base, "clean": y}


def filter_frequency_response(fs: float, p: PreprocessParams, n: int = 2048):
    """АЧХ лінійної частини ланцюжка (смуговий + режекторний), дБ."""
    w, h = sps.sosfreqz(_band_sos(fs, p), worN=n, fs=fs)
    h_total = np.abs(h) ** 2  # квадрат, бо фільтр застосовано двічі (filtfilt)
    if p.notch and p.notch_freq < fs / 2:
        b, a = sps.iirnotch(p.notch_freq, p.notch_q, fs)
        _, hn = sps.freqz(b, a, worN=w, fs=fs)
        h_total *= np.abs(hn) ** 2
    return w, 10 * np.log10(np.maximum(h_total, 1e-12))


def power_spectrum(x: np.ndarray, fs: float):
    """Спектральна щільність потужності методом Велча, дБ."""
    f, pxx = sps.welch(x - np.mean(x), fs=fs, nperseg=min(len(x), int(4 * fs)))
    return f, 10 * np.log10(np.maximum(pxx, 1e-15))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """SNR (дБ) сигналу test відносно чистого еталону reference."""
    ref = reference - np.mean(reference)
    err = (test - np.mean(test)) - ref
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum(err ** 2) + 1e-12)))
