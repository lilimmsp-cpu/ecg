"""
data.py — тестові дані для дашборду.

Два джерела:
1. Записи бази MIT-BIH Arrhythmia (PhysioNet) — завантажуються автоматично
   через бібліотеку wfdb і кешуються в папці ./data/mitdb.
2. Синтетичні ЕКГ — реалістична модель (сума гаусових хвиль P, Q, R, S, T,
   як у ECGSYN) з індивідуальною морфологією для кожної «особи», варіабельністю
   серцевого ритму та шумами: дрейф ізолінії, мережева наводка, ЕМГ-шум,
   артефакти руху. Використовується, якщо інтернету / wfdb немає.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sps

# Частота дискретизації MIT-BIH (Гц)
FS_MITBIH = 360

# 48 записів MIT-BIH (47 осіб: записи 201 та 202 належать одній людині)
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
MITBIH_SUBJECT = {r: r for r in MITBIH_RECORDS}
MITBIH_SUBJECT["202"] = "201"  # одна особа

# Символи анотацій, що позначають серцеві скорочення (QRS)
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

DATA_DIR = Path(__file__).resolve().parent / "data" / "mitdb"

N_SYNTHETIC = 47  # як кількість осіб у MIT-BIH


@dataclass
class ECGRecord:
    """Один запис ЕКГ із метаданими."""
    name: str                       # ідентифікатор запису
    subject: str                    # ідентифікатор особи (клас для біометрії)
    fs: float                       # частота дискретизації, Гц
    signal: np.ndarray              # «сирий» сигнал, мВ
    source: str                     # "MIT-BIH" або "Синтетика"
    lead: str = "MLII"              # відведення
    clean: Optional[np.ndarray] = None    # чистий сигнал (лише для синтетики)
    r_peaks: Optional[np.ndarray] = None  # еталонні R-піки (анотації / модель)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MIT-BIH через wfdb
# ---------------------------------------------------------------------------

def wfdb_available() -> bool:
    """Чи встановлено бібліотеку wfdb."""
    try:
        import wfdb  # noqa: F401
        return True
    except Exception:
        return False


def load_mitbih_record(name: str, duration_s: float = 120.0, channel: int = 0) -> ECGRecord:
    """Завантажити запис MIT-BIH (з локального кешу або з PhysioNet)."""
    import wfdb

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = DATA_DIR / name
    if not (DATA_DIR / f"{name}.hea").exists():
        # Автоматичне завантаження .hea/.dat/.atr з PhysioNet
        wfdb.dl_database("mitdb", dl_dir=str(DATA_DIR), records=[name], annotators=["atr"])
    sampto = int(duration_s * FS_MITBIH)
    rec = wfdb.rdrecord(str(local), sampto=sampto, channels=[channel])
    ann = wfdb.rdann(str(local), "atr", sampto=sampto)
    sig = rec.p_signal[:, 0].astype(float)
    peaks = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS], dtype=int)
    return ECGRecord(
        name=name, subject=MITBIH_SUBJECT.get(name, name), fs=float(rec.fs),
        signal=sig, source="MIT-BIH", lead=rec.sig_name[0], r_peaks=peaks,
        info={"Коментар": "; ".join(rec.comments) if rec.comments else ""},
    )


# ---------------------------------------------------------------------------
# Синтетичні ЕКГ
# ---------------------------------------------------------------------------

# Базова морфологія хвиль: (зсув відносно R, с; амплітуда, мВ; ширина, с)
_BASE_WAVES = {
    "P": (-0.200, 0.15, 0.025),
    "Q": (-0.035, -0.12, 0.010),
    "R": (0.000, 1.20, 0.012),
    "S": (0.035, -0.28, 0.012),
    "T": (0.280, 0.32, 0.050),
}


def _subject_morphology(subject_idx: int) -> dict:
    """Індивідуальні параметри хвиль для «особи» (детерміновано за номером)."""
    rng = np.random.default_rng(1000 + subject_idx)
    morph = {}
    for wave, (mu, a, w) in _BASE_WAVES.items():
        amp_scale = rng.normal(1.0, 0.30 if wave != "R" else 0.20)
        if wave in ("Q", "S"):
            amp_scale = abs(amp_scale)
        morph[wave] = (
            mu * rng.normal(1.0, 0.12) + (0 if wave == "R" else rng.normal(0, 0.006)),
            a * amp_scale,
            w * float(np.clip(rng.normal(1.0, 0.18), 0.6, 1.6)),
        )
    # Іноді — двофазна (додаткова) компонента T-хвилі
    if rng.random() < 0.3:
        mu, a, w = morph["T"]
        morph["T2"] = (mu + 0.06, -0.35 * a, 0.6 * w)
    hr = float(np.clip(rng.normal(72, 9), 52, 100))
    return {"waves": morph, "hr": hr}


def generate_synthetic_ecg(subject_idx: int, duration_s: float = 60.0, fs: float = FS_MITBIH,
                           noise_level: float = 1.0, seed: Optional[int] = None) -> ECGRecord:
    """Згенерувати реалістичний зашумлений ЕКГ-запис для особи subject_idx."""
    rng = np.random.default_rng(seed if seed is not None else 5000 + subject_idx)
    params = _subject_morphology(subject_idx)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Ритм: середня ЧСС + дихальна синусова аритмія + випадкова варіабельність
    rr_mean = 60.0 / params["hr"]
    beat_times, tk = [], 0.35
    while tk < duration_s - 0.5:
        beat_times.append(tk)
        rr = rr_mean * (1 + 0.04 * np.sin(2 * np.pi * 0.25 * tk) + rng.normal(0, 0.025))
        tk += max(rr, 0.35)
    beat_times = np.array(beat_times)

    clean = np.zeros(n)
    for k, tb in enumerate(beat_times):
        rr = (beat_times[k + 1] - tb) if k + 1 < len(beat_times) else rr_mean
        qt_scale = np.sqrt(rr / rr_mean)  # формула Базетта для інтервалу QT
        lo, hi = int(max(0, (tb - 0.4) * fs)), int(min(n, (tb + 0.6) * fs))
        tt = t[lo:hi] - tb
        for wave, (mu, a, w) in params["waves"].items():
            mu_k = mu * qt_scale if wave.startswith("T") else mu
            a_k = a * rng.normal(1.0, 0.03)  # мала міжциклова варіабельність
            clean[lo:hi] += a_k * np.exp(-((tt - mu_k) ** 2) / (2 * w ** 2))

    # --- Шуми ---
    nl = noise_level
    # 1) дрейф ізолінії (дихання + повільний випадковий дрейф)
    walk = np.cumsum(rng.normal(0, 1, n))
    b, a = sps.butter(2, 0.3 / (fs / 2))
    walk = sps.filtfilt(b, a, walk)
    walk = walk / (np.std(walk) + 1e-12)
    baseline = 0.25 * np.sin(2 * np.pi * 0.22 * t + rng.uniform(0, 6)) + 0.15 * walk
    # 2) мережева наводка 60 Гц (як у США, де записано MIT-BIH)
    powerline = 0.06 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6))
    # 3) ЕМГ-шум (широкосмуговий, переважно > 20 Гц)
    b, a = sps.butter(2, 20 / (fs / 2), btype="high")
    emg = sps.filtfilt(b, a, rng.normal(0, 0.05, n))
    # 4) рідкісні артефакти руху
    motion = np.zeros(n)
    for _ in range(int(duration_s // 20)):
        c = rng.uniform(0, duration_s)
        motion += rng.normal(0, 0.35) * np.exp(-((t - c) ** 2) / (2 * 0.15 ** 2))

    noisy = clean + nl * (baseline + powerline + emg + motion)
    r_peaks = np.round(beat_times * fs).astype(int)
    return ECGRecord(
        name=f"SYN-{subject_idx + 1:02d}", subject=f"SYN-{subject_idx + 1:02d}", fs=fs,
        signal=noisy, clean=clean, r_peaks=r_peaks, source="Синтетика", lead="модель",
        info={"ЧСС, уд/хв": round(params["hr"], 1), "Рівень шуму": noise_level},
    )


def synthetic_record_names(n: int = N_SYNTHETIC) -> list[str]:
    return [f"SYN-{i + 1:02d}" for i in range(n)]


def load_record(name: str, duration_s: float = 60.0, noise_level: float = 1.0) -> ECGRecord:
    """Універсальне завантаження запису за назвою."""
    if name.startswith("SYN-"):
        return generate_synthetic_ecg(int(name.split("-")[1]) - 1, duration_s, noise_level=noise_level)
    return load_mitbih_record(name, duration_s)


def detect_available_source() -> str:
    """Перевірити, чи можна отримати MIT-BIH (wfdb + мережа або кеш)."""
    if not wfdb_available():
        return "Синтетика"
    try:
        load_mitbih_record("100", duration_s=5)
        return "MIT-BIH"
    except Exception:
        return "Синтетика"
