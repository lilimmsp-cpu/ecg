"""
evaluation.py — протокол експерименту та метрики.

Протокол статті (розд. 3.1):
  * для кожної особи випадково відбирається 5 циклів у тестову вибірку,
    решта — навчальна;
  * навчальна вибірка ділиться на порції (для MIT-BIH — 5 раундів) і
    подається моделі послідовно (онлайн-сценарій);
  * шаблон реєстрації (enroll) особи — середнє її навчальних циклів;
  * ідентифікація — найменша евклідова відстань до шаблону (Accuracy);
  * верифікація — рівна похибка EER за відстанями «запит–шаблон».
Додатково рахуються метрики фільтрації (SNR) і детекції R-піків (Se, +P).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from preprocessing import PreprocessParams, preprocess, snr_db
from features import (FeatureParams, ModelParams, OnlineHadamardMF, extract_features,
                      pan_tompkins, segment_heartbeats, mrlbp_1d, _l2)

# ---------------------------------------------------------------------------
# Еталонні результати авторів (таблиці 1–3 статті)
# ---------------------------------------------------------------------------

PAPER_RESULTS = {
    "MIT-BIH": {"Accuracy": 99.15, "EER": 0.64},
    "CYBHiDB T1": {"Accuracy": 98.73, "EER": 1.58},
    "CYBHiDB T2": {"Accuracy": 97.78, "EER": 1.71},
}

PAPER_TABLE1 = pd.DataFrame({
    "Метод": ["[21] DWT (Dar et al.)", "[11] MSDF+CMF (Wang et al.)", "[22] Fusion (Hejazi et al.)",
              "[23] Two-stage (Wu et al.), deep", "[1] Spectral corr.+DL, deep", "OEB-HC (стаття)"],
    "EER, %": [None, 2.73, None, 0.02, None, 0.64],
    "Accuracy, %": [93.1, 94.68, 98.2, 99.7, 96.5, 99.15],
})

PAPER_SETUP = {
    "Довжина циклу (MIT-BIH)": "260 відліків",
    "Детектор R-піків": "Pan–Tompkins",
    "Базові ознаки": "1DMRLBP",
    "Порядок матриці Адамара / r": "2^7 = 128",
    "Ітерацій": "6",
    "α, β, θ, δ": "1, 1, 1, 1",
    "Тест на особу": "5 циклів",
    "Порції (MIT-BIH)": "4 × 70 + 49 = 329 навчальних зразків",
}


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def detection_metrics(true_peaks, pred_peaks, fs: float, tol_s: float = 0.15) -> dict:
    """Чутливість (Se), позитивна прогностичність (+P) і F1 детекції R-піків."""
    true_peaks = np.asarray(true_peaks if true_peaks is not None else [], dtype=int)
    pred_peaks = np.asarray(pred_peaks, dtype=int)
    if len(true_peaks) == 0:
        return {"Se, %": np.nan, "+P, %": np.nan, "F1, %": np.nan, "TP": 0, "FP": len(pred_peaks), "FN": 0}
    tol = int(tol_s * fs)
    used = np.zeros(len(pred_peaks), dtype=bool)
    tp = 0
    for t in true_peaks:
        if len(pred_peaks) == 0:
            break
        j = int(np.argmin(np.abs(pred_peaks - t)))
        if abs(pred_peaks[j] - t) <= tol and not used[j]:
            used[j] = True
            tp += 1
    fn, fp = len(true_peaks) - tp, len(pred_peaks) - tp
    se = 100 * tp / max(tp + fn, 1)
    ppv = 100 * tp / max(tp + fp, 1)
    f1 = 2 * se * ppv / max(se + ppv, 1e-9)
    return {"Se, %": se, "+P, %": ppv, "F1, %": f1, "TP": tp, "FP": fp, "FN": fn}


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матриця евклідових відстаней між рядками a та b."""
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def identification_accuracy(dist: np.ndarray, q_labels, e_labels) -> float:
    """Частка запитів, для яких найближчий шаблон належить тій самій особі, %."""
    pred = np.asarray(e_labels)[np.argmin(dist, axis=1)]
    return float(100 * np.mean(pred == np.asarray(q_labels)))


def verification_scores(dist: np.ndarray, q_labels, e_labels):
    """Відстані «свій» (genuine) та «чужий» (impostor)."""
    same = np.asarray(q_labels)[:, None] == np.asarray(e_labels)[None, :]
    return dist[same], dist[~same]


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """EER (%) та криві FAR/FRR. Прийняття, якщо відстань ≤ порогу."""
    if len(genuine) == 0 or len(impostor) == 0:
        return {"EER": np.nan, "thr": np.array([]), "FAR": np.array([]), "FRR": np.array([])}
    thr = np.unique(np.concatenate([genuine, impostor]))
    g, im = np.sort(genuine), np.sort(impostor)
    frr = 1 - np.searchsorted(g, thr, side="right") / len(g)
    far = np.searchsorted(im, thr, side="right") / len(im)
    diff = far - frr
    i = int(np.argmax(diff >= 0))
    if i > 0 and diff[i] != diff[i - 1]:
        w = -diff[i - 1] / (diff[i] - diff[i - 1])  # лінійна інтерполяція точки перетину
        eer = (far[i - 1] + w * (far[i] - far[i - 1]) + frr[i - 1] + w * (frr[i] - frr[i - 1])) / 2
    else:
        eer = (far[i] + frr[i]) / 2
    return {"EER": float(100 * eer), "thr": thr, "FAR": far, "FRR": frr}


# ---------------------------------------------------------------------------
# Обробка одного запису
# ---------------------------------------------------------------------------

def process_record(rec, prep: PreprocessParams, fp: FeatureParams) -> dict:
    """Фільтрація → Pan–Tompkins → сегментація → ознаки для одного запису."""
    out = preprocess(rec.signal, rec.fs, prep)
    peaks, stages = pan_tompkins(out["clean"], rec.fs, return_stages=True)
    beats, used = segment_heartbeats(out["clean"], peaks, fp.pre, fp.post)
    feats = extract_features(beats, fp)
    det = detection_metrics(rec.r_peaks, peaks, rec.fs)
    res = {"rec": rec, **out, "peaks": peaks, "stages": stages, "beats": beats,
           "beat_peaks": used, "features": feats, "detection": det}
    if rec.clean is not None:
        res["snr_before"] = snr_db(rec.clean, rec.signal)
        res["snr_after"] = snr_db(rec.clean, out["clean"])
    hr = 60 * rec.fs / np.median(np.diff(peaks)) if len(peaks) > 2 else np.nan
    res["hr"] = float(hr)
    return res


# ---------------------------------------------------------------------------
# Пакетний експеримент
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    per_record: pd.DataFrame
    rounds: pd.DataFrame
    final: dict
    batch_mode: dict
    genuine: np.ndarray
    impostor: np.ndarray
    eer_curve: dict
    query_emb: np.ndarray
    query_labels: np.ndarray
    enroll_emb: np.ndarray
    enroll_labels: np.ndarray
    loss_history: list
    n_subjects: int
    n_train: int
    n_test: int
    feature_dim: int
    sensitivity: Optional[pd.DataFrame] = None
    params: dict = field(default_factory=dict)


def _split(labels: np.ndarray, n_test: int, n_train: int, rng):
    """Для кожної особи: n_test випадкових циклів у тест, до n_train — у навчання."""
    tr, te = [], []
    for s in np.unique(labels):
        idx = rng.permutation(np.where(labels == s)[0])
        if len(idx) < n_test + 1:
            continue
        te.extend(idx[:n_test])
        rest = idx[n_test:]
        tr.extend(rest[:n_train] if n_train > 0 else rest)
    return np.array(tr, dtype=int), np.array(te, dtype=int)


def _enroll(feats, beats, labels, idx, fp: FeatureParams, mode: str):
    """Шаблони реєстрації: середнє «гомологічних» циклів особи з навчальних даних."""
    labs = labels[idx]
    uniq = np.unique(labs)
    rows = []
    for s in uniq:
        sel = idx[labs == s]
        if mode == "mean_beat":
            rows.append(extract_features(beats[sel].mean(axis=0, keepdims=True), fp)[0])
        else:
            rows.append(_l2(feats[sel].mean(axis=0)))
    return np.vstack(rows), uniq


def _run_stream(feats, beats, labels, tr, te, mp: ModelParams, fp: FeatureParams, n_chunks: int, rng):
    """Онлайн-прогін: послідовні раунди, оцінка після кожного."""
    model = OnlineHadamardMF(feats.shape[1], mp.r, mp.alpha, mp.beta, mp.theta, mp.delta,
                             mp.n_iter, seed=mp.seed)
    order = rng.permutation(tr)                      # потік у випадковому порядку
    chunks = np.array_split(order, n_chunks)
    rounds, seen = [], np.array([], dtype=int)
    t0 = time.perf_counter()
    for t, ch in enumerate(chunks, start=1):
        tr_t = time.perf_counter()
        model.partial_fit(feats[ch], labels[ch])
        dt = time.perf_counter() - tr_t
        seen = np.concatenate([seen, ch])
        e_feat, e_lab = _enroll(feats, beats, labels, seen, fp, mp.enroll_mode)
        q = model.transform(feats[te])
        e = model.transform(e_feat)
        dist = pairwise_distances(q, e)
        acc = identification_accuracy(dist, labels[te], e_lab)
        gen, imp = verification_scores(dist, labels[te], e_lab)
        eer = compute_eer(gen, imp)["EER"]
        rounds.append({"Раунд": t, "Нових зразків": len(ch), "Усього зразків": len(seen),
                       "Відомих осіб": len(model.code_of), "Accuracy, %": acc, "EER, %": eer,
                       "Час раунду, мс": 1000 * dt})
    total = time.perf_counter() - t0
    return model, pd.DataFrame(rounds), dist, e, e_lab, q, total


def run_experiment(records: list, prep: PreprocessParams, fp: FeatureParams, mp: ModelParams,
                   progress: Optional[Callable[[float, str], None]] = None,
                   with_sensitivity: bool = False) -> ExperimentResult:
    """Повний пакетний прогін за протоколом статті."""
    rng = np.random.default_rng(mp.seed)
    feats_all, beats_all, labs_all, rows = [], [], [], []
    for i, rec in enumerate(records):
        if progress:
            progress(0.6 * i / max(len(records), 1), f"Обробка запису {rec.name}…")
        pr = process_record(rec, prep, fp)
        feats_all.append(pr["features"])
        beats_all.append(pr["beats"])
        labs_all.extend([rec.subject] * len(pr["beats"]))
        row = {"Запис": rec.name, "Особа": rec.subject, "Джерело": rec.source,
               "Тривалість, с": len(rec.signal) / rec.fs, "R-піків": len(pr["peaks"]),
               "Циклів": len(pr["beats"]), "ЧСС, уд/хв": pr["hr"], **pr["detection"]}
        if "snr_after" in pr:
            row["SNR до, дБ"] = pr["snr_before"]
            row["SNR після, дБ"] = pr["snr_after"]
        rows.append(row)
    feats = np.vstack(feats_all)
    beats = np.vstack(beats_all)
    labels = np.array(labs_all)
    tr, te = _split(labels, mp.n_test, mp.n_train, rng)
    n_subj = len(np.unique(labels[te]))
    if n_subj > mp.r:
        raise ValueError(f"Осіб ({n_subj}) більше, ніж r = {mp.r}. Збільште r.")

    if progress:
        progress(0.65, "Онлайн-навчання за раундами…")
    model, rounds, dist, e, e_lab, q, t_online = _run_stream(
        feats, beats, labels, tr, te, mp, fp, mp.n_chunks, np.random.default_rng(mp.seed + 1))
    gen, imp = verification_scores(dist, labels[te], e_lab)
    eer = compute_eer(gen, imp)

    if progress:
        progress(0.8, "Порівняння з пакетним (offline) навчанням…")
    _, rounds_b, _, _, _, _, t_batch = _run_stream(
        feats, beats, labels, tr, te, mp, fp, 1, np.random.default_rng(mp.seed + 1))

    sens = None
    if with_sensitivity:
        sens_rows = []
        grid = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
        for k, name in enumerate(["alpha", "beta", "theta", "delta"]):
            for v in grid:
                if progress:
                    progress(0.82 + 0.17 * k / 4, f"Аналіз чутливості: {name} = {v:g}")
                mp_s = ModelParams(**{**mp.to_dict(), name: v})
                _, rr, *_ = _run_stream(feats, beats, labels, tr, te, mp_s, fp, mp.n_chunks,
                                        np.random.default_rng(mp.seed + 1))
                sens_rows.append({"Параметр": name, "Значення": v,
                                  "Accuracy, %": rr["Accuracy, %"].iloc[-1],
                                  "EER, %": rr["EER, %"].iloc[-1]})
        sens = pd.DataFrame(sens_rows)
    if progress:
        progress(1.0, "Готово")

    final = {"Accuracy, %": float(rounds["Accuracy, %"].iloc[-1]), "EER, %": float(eer["EER"]),
             "Час онлайн-навчання, с": t_online}
    batch = {"Accuracy, %": float(rounds_b["Accuracy, %"].iloc[-1]),
             "EER, %": float(rounds_b["EER, %"].iloc[-1]), "Час, с": t_batch}
    return ExperimentResult(
        per_record=pd.DataFrame(rows), rounds=rounds, final=final, batch_mode=batch,
        genuine=gen, impostor=imp, eer_curve=eer, query_emb=q, query_labels=labels[te],
        enroll_emb=e, enroll_labels=e_lab, loss_history=model.loss_history,
        n_subjects=n_subj, n_train=len(tr), n_test=len(te), feature_dim=feats.shape[1],
        sensitivity=sens,
        params={"Попередня обробка": prep.to_dict(), "Ознаки": fp.to_dict(), "Модель": mp.to_dict()},
    )


# ---------------------------------------------------------------------------
# Звіт Excel
# ---------------------------------------------------------------------------

def build_excel_report(result: Optional[ExperimentResult], single: Optional[dict] = None,
                       params: Optional[dict] = None, dataset_name: str = "MIT-BIH") -> bytes:
    """Сформувати звіт .xlsx (кілька аркушів) і повернути його як байти."""
    buf = io.BytesIO()
    params = params or (result.params if result else {})
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        info = [("Стаття", "Wang K. et al. Online ECG Biometrics via Hadamard Code, ICASSP 2022"),
                ("DOI", "10.1109/ICASSP43922.2022.9747753"),
                ("Дата звіту", time.strftime("%Y-%m-%d %H:%M"))]
        pd.DataFrame(info, columns=["Поле", "Значення"]).to_excel(xw, sheet_name="Опис", index=False)

        prow = [(grp, k, str(v)) for grp, d in params.items() for k, v in d.items()]
        pd.DataFrame(prow, columns=["Група", "Параметр", "Значення"]).to_excel(xw, sheet_name="Параметри", index=False)

        if result is not None:
            ref = PAPER_RESULTS.get(dataset_name, PAPER_RESULTS["MIT-BIH"])
            cmp = pd.DataFrame([
                {"Метрика": "Accuracy, %", "Відтворено (онлайн)": result.final["Accuracy, %"],
                 "Пакетне навчання": result.batch_mode["Accuracy, %"], "Стаття (OEB-HC)": ref["Accuracy"],
                 "Різниця з статтею": result.final["Accuracy, %"] - ref["Accuracy"]},
                {"Метрика": "EER, %", "Відтворено (онлайн)": result.final["EER, %"],
                 "Пакетне навчання": result.batch_mode["EER, %"], "Стаття (OEB-HC)": ref["EER"],
                 "Різниця з статтею": result.final["EER, %"] - ref["EER"]},
            ])
            cmp.to_excel(xw, sheet_name="Порівняння зі статтею", index=False)
            pd.DataFrame([{"Осіб": result.n_subjects, "Навчальних зразків": result.n_train,
                           "Тестових зразків": result.n_test, "Розмірність ознак m": result.feature_dim,
                           **result.final}]).to_excel(xw, sheet_name="Підсумок", index=False)
            result.rounds.to_excel(xw, sheet_name="Раунди", index=False)
            result.per_record.to_excel(xw, sheet_name="Записи", index=False)
            if result.sensitivity is not None:
                result.sensitivity.to_excel(xw, sheet_name="Чутливість", index=False)
        if single is not None:
            pd.DataFrame([single]).to_excel(xw, sheet_name="Поточний запис", index=False)
        PAPER_TABLE1.to_excel(xw, sheet_name="Таблиця 1 статті", index=False)

        # Оформлення: жирні заголовки, ширина стовпців
        from openpyxl.styles import Font, PatternFill
        for ws in xw.book.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="2A78D6")
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            ws.freeze_panes = "A2"
    return buf.getvalue()
