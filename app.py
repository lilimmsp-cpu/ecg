"""
app.py — інтерактивний науковий дашборд (Streamlit + Plotly), що відтворює
експеримент статті «Online ECG Biometrics via Hadamard Code» (ICASSP 2022).

Запуск:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import (MITBIH_RECORDS, detect_available_source, load_record,
                  synthetic_record_names)
from preprocessing import (BASELINE_METHODS, PreprocessParams, filter_frequency_response,
                           power_spectrum)
from features import (FEATURE_TYPES, FeatureParams, ModelParams, lbp_1d_codes,
                      feature_dim, pca_2d)
from evaluation import (PAPER_RESULTS, PAPER_SETUP, PAPER_TABLE1, build_excel_report,
                        process_record, run_experiment)

# ---------------------------------------------------------------------------
# Оформлення
# ---------------------------------------------------------------------------
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_GRAY, C_MUTED, C_GRID = "#9a9893", "#52514e", "rgba(128,128,128,0.18)"
_trapz = getattr(np, "trapezoid", None) or np.trapz  # сумісність з NumPy 1.x
LOG_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

st.set_page_config(page_title="ЕКГ-біометрія: OEB-HC", page_icon="🫀", layout="wide")


def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    """Єдиний стиль графіків: стримана сітка, тонкі лінії, легенда зверху."""
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified", template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Кешовані обчислення
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_source() -> str:
    return detect_available_source()


@st.cache_data(show_spinner=False)
def cached_record(name: str, duration: float, noise: float):
    return load_record(name, duration, noise)


@st.cache_data(show_spinner=False)
def cached_process(name: str, duration: float, noise: float, prep: dict, feat: dict):
    rec = cached_record(name, duration, noise)
    fp = FeatureParams(**{**feat, "radii": tuple(int(r) for r in feat["radii"].split(", "))})
    return process_record(rec, PreprocessParams(**prep), fp)


# ---------------------------------------------------------------------------
# Бічна панель керування
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.header("🎛️ Панель керування")

auto_src = cached_source()
src_options = ["Синтетичні ЕКГ (без інтернету)", "MIT-BIH (wfdb, PhysioNet)"]
src = sb.radio("Джерело даних", src_options, index=1 if auto_src == "MIT-BIH" else 0,
               help="MIT-BIH завантажується автоматично через wfdb. Якщо бібліотека чи "
                    "інтернет недоступні, використовуються синтетичні ЕКГ.")
use_mitbih = src.startswith("MIT-BIH")
if use_mitbih and auto_src != "MIT-BIH":
    sb.warning("MIT-BIH недоступна (немає wfdb або інтернету). Використано синтетичні ЕКГ.")
    use_mitbih = False

all_names = MITBIH_RECORDS if use_mitbih else synthetic_record_names()
record = sb.selectbox("Запис", all_names, index=0)
duration = sb.slider("Тривалість запису, с", 20, 300 if use_mitbih else 180, 60, step=10)
noise = 1.0
if not use_mitbih:
    noise = sb.slider("Рівень шуму (синтетика)", 0.0, 3.0, 1.0, 0.1,
                      help="Множник для дрейфу ізолінії, наводки 60 Гц, ЕМГ та артефактів руху.")

with sb.expander("① Попередня обробка", expanded=True):
    bmethod = st.selectbox("Видалення базової лінії", list(BASELINE_METHODS),
                           format_func=BASELINE_METHODS.get)
    w1, w2, hp = 200, 600, 0.5
    if bmethod == "median":
        w1 = st.slider("Вікно медіани 1, мс", 50, 400, 200, 10)
        w2 = st.slider("Вікно медіани 2, мс", 200, 1200, 600, 50)
    elif bmethod == "highpass":
        hp = st.slider("Частота зрізу ФВЧ, Гц", 0.05, 2.0, 0.5, 0.05)
    band = st.slider("Смуга пропускання, Гц", 0.1, 100.0, (0.5, 40.0), 0.1)
    order = st.slider("Порядок фільтра Баттерворта", 1, 8, 4)
    notch = st.toggle("Режекторний фільтр мережі", value=True)
    notch_f = st.radio("Частота мережі, Гц", [60, 50], horizontal=True, disabled=not notch)
prep = PreprocessParams(bmethod, w1, w2, hp, band[0], band[1], order, notch, float(notch_f))

with sb.expander("② Сегментація та ознаки 1DMRLBP"):
    pre = st.slider("Відліків до R-піку", 40, 200, 90, 5)
    post = st.slider("Відліків після R-піку", 80, 400, 170, 5)
    st.caption(f"Довжина циклу: **{pre + post}** відліків (у статті — 260).")
    ftype = st.selectbox("Базові ознаки", list(FEATURE_TYPES), format_func=FEATURE_TYPES.get)
    lbp_p = st.slider("Сусідів з кожного боку P (код = 2P біт)", 1, 3, 2)
    radii = st.multiselect("Радіуси (роздільність)", [1, 2, 3, 4, 6, 8, 12, 16], default=[1, 3, 6, 12])
    n_win = st.slider("Кількість часових вікон", 1, 16, 10)
    thr = st.slider("Поріг порівняння LBP, мВ", 0.0, 0.1, 0.03, 0.005)
    hell = st.checkbox("Корінь із гістограм (ядро Хеллінгера)", value=True)
feat = FeatureParams(pre, post, ftype, lbp_p, tuple(sorted(radii)) or (1,), n_win, thr, hell)

with sb.expander("③ Модель OEB-HC (коди Адамара)"):
    r_dim = st.select_slider("Розмірність r = порядок матриці Адамара", [64, 128, 256], value=128)
    fmt = lambda v: f"{v:g}"
    alpha = st.select_slider("α (проєкція W)", LOG_GRID, value=1.0, format_func=fmt)
    beta = st.select_slider("β (факторизація G)", LOG_GRID, value=1.0, format_func=fmt)
    theta = st.select_slider("θ (коди Адамара)", LOG_GRID, value=1.0, format_func=fmt)
    delta = st.select_slider("δ (регуляризація)", LOG_GRID, value=1.0, format_func=fmt)
    n_iter = st.slider("Ітерацій на раунд", 1, 20, 6)
    n_chunks = st.slider("Кількість порцій (раундів)", 1, 10, 5)
    n_test = st.slider("Тестових циклів на особу", 1, 10, 5)
    n_train = st.slider("Навчальних циклів на особу (0 — усі)", 0, 60, 7,
                        help="У статті 329 навчальних зразків на 47 осіб ≈ 7 на особу.")
    enroll = st.radio("Шаблон реєстрації", ["mean_feature", "mean_beat"],
                      format_func={"mean_feature": "Середнє ознак", "mean_beat": "Ознаки середнього циклу"}.get)
    seed = st.number_input("Зерно випадковості", 0, 9999, 0)
model_p = ModelParams(r_dim, alpha, beta, theta, delta, n_iter, n_chunks, n_test, n_train, int(seed), enroll)

sb.divider()
sb.subheader("🚀 Пакетний прогін")
batch_names = sb.multiselect("Записи для експерименту", all_names, default=all_names,
                             help="Кожен запис — окрема особа (MIT-BIH 201/202 — одна особа).")
with_sens = sb.checkbox("+ аналіз чутливості α, β, θ, δ (рис. 2 статті)", value=False)
run_batch = sb.button("Запустити пакетний прогін", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------
st.title("🫀 Онлайн ЕКГ-біометрія з кодами Адамара")
st.caption("Відтворення методики: K. Wang, G. Yang, Y. Huang, L. Yang, Y. Yin. "
           "*Online ECG Biometrics via Hadamard Code*, ICASSP 2022, DOI 10.1109/ICASSP43922.2022.9747753")
dataset_label = "MIT-BIH" if use_mitbih else "Синтетика"
if not use_mitbih:
    st.info("Працюємо з **синтетичними ЕКГ** (47 «осіб», 360 Гц, як у MIT-BIH). "
            "Еталони статті наведено для MIT-BIH — порівняння орієнтовне.", icon="ℹ️")

# Обробка поточного запису (оновлюється в реальному часі)
try:
    pr = cached_process(record, float(duration), float(noise), prep.to_dict(), feat.to_dict())
except Exception as exc:  # наприклад, збій мережі під час завантаження
    st.error(f"Не вдалося обробити запис {record}: {exc}")
    st.stop()
rec = pr["rec"]
fs = rec.fs
t = np.arange(len(rec.signal)) / fs

# Ключові показники поточного запису
k = st.columns(5)
k[0].metric("Знайдено R-піків", len(pr["peaks"]))
k[1].metric("ЧСС (медіана)", f"{pr['hr']:.0f} уд/хв" if np.isfinite(pr["hr"]) else "—")
det = pr["detection"]
k[2].metric("Чутливість Se", f"{det['Se, %']:.2f} %" if np.isfinite(det["Se, %"]) else "—")
k[3].metric("Прогностичність +P", f"{det['+P, %']:.2f} %" if np.isfinite(det["+P, %"]) else "—")
if "snr_after" in pr:
    k[4].metric("SNR після очищення", f"{pr['snr_after']:.1f} дБ",
                delta=f"{pr['snr_after'] - pr['snr_before']:+.1f} дБ")
else:
    k[4].metric("Циклів для ознак", len(pr["beats"]))

tabs = st.tabs(["📈 Сигнал", "🌈 Спектр", "🧬 Ознаки", "🏁 Результати експерименту",
                "🎚️ Чутливість", "📄 Методика"])

# ---------------------------------------------------------------------------
# Вкладка «Сигнал»
# ---------------------------------------------------------------------------
with tabs[0]:
    win = st.slider("Вікно перегляду, с", 0.0, float(max(t[-1] - 2, 1)), (0.0, min(10.0, float(t[-1]))),
                    0.5, key="view")
    m = (t >= win[0]) & (t <= win[1])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Сирий сигнал ЕКГ", "Очищений сигнал"))
    fig.add_trace(go.Scatter(x=t[m], y=pr["raw"][m], name="Сирий сигнал",
                             line=dict(color=C_GRAY, width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=t[m], y=pr["baseline"][m], name="Оцінка базової лінії",
                             line=dict(color=C_ORANGE, width=2, dash="dot")), 1, 1)
    if rec.clean is not None:
        fig.add_trace(go.Scatter(x=t[m], y=rec.clean[m], name="Еталон (без шуму)",
                                 line=dict(color=C_AQUA, width=1.5), opacity=0.8), 2, 1)
    fig.add_trace(go.Scatter(x=t[m], y=pr["clean"][m], name="Очищений сигнал",
                             line=dict(color=C_BLUE, width=1.6)), 2, 1)
    pk = pr["peaks"][(pr["peaks"] >= m.argmax()) & (t[pr["peaks"]] <= win[1])]
    fig.add_trace(go.Scatter(x=t[pk], y=pr["clean"][pk], mode="markers", name="R-піки (Pan–Tompkins)",
                             marker=dict(color=C_ORANGE, size=9, symbol="triangle-down",
                                         line=dict(color="white", width=2))), 2, 1)
    fig.update_yaxes(title_text="мВ", row=1, col=1)
    fig.update_yaxes(title_text="мВ", row=2, col=1)
    fig.update_xaxes(title_text="Час, с", row=2, col=1)
    st.plotly_chart(style(fig, 560), width="stretch")

    with st.expander("Етапи алгоритму Пана–Томпкінса"):
        stg = pr["stages"]
        f2 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("Смуговий фільтр 5–15 Гц", "Похідна",
                                           "Квадрат", "Ковзне інтегрування 150 мс"))
        for i, key in enumerate(["bandpass", "derivative", "squared", "integrated"], start=1):
            f2.add_trace(go.Scatter(x=t[m], y=stg[key][m], showlegend=False,
                                    line=dict(color=C_BLUE, width=1.2)), i, 1)
        f2.update_xaxes(title_text="Час, с", row=4, col=1)
        st.plotly_chart(style(f2, 640), width="stretch")

# ---------------------------------------------------------------------------
# Вкладка «Спектр»
# ---------------------------------------------------------------------------
with tabs[1]:
    f_raw, p_raw = power_spectrum(pr["raw"], fs)
    f_cl, p_cl = power_spectrum(pr["clean"], fs)
    fw, hw = filter_frequency_response(fs, prep)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f_raw, y=p_raw, name="Сирий сигнал", line=dict(color=C_GRAY, width=1.5)))
        fig.add_trace(go.Scatter(x=f_cl, y=p_cl, name="Очищений сигнал", line=dict(color=C_BLUE, width=2)))
        fig.add_vrect(x0=prep.band_low, x1=prep.band_high, fillcolor=C_BLUE, opacity=0.06, line_width=0)
        if prep.notch:
            fig.add_vline(x=prep.notch_freq, line=dict(color=C_ORANGE, dash="dot", width=1.5),
                          annotation_text=f"Режекція {prep.notch_freq:g} Гц")
        fig.update_xaxes(title_text="Частота, Гц", range=[0, min(fs / 2, 100)])
        fig.update_yaxes(title_text="СЩП, дБ (мВ²/Гц)")
        st.plotly_chart(style(fig, 420, "Спектральна щільність потужності (метод Велча)"),
                        width="stretch")
    with c2:
        fig = go.Figure(go.Scatter(x=fw, y=hw, name="АЧХ", line=dict(color=C_BLUE, width=2)))
        fig.update_xaxes(title_text="Частота, Гц", range=[0, min(fs / 2, 100)])
        fig.update_yaxes(title_text="Підсилення, дБ", range=[-80, 5])
        st.plotly_chart(style(fig, 420, "АЧХ ланцюжка фільтрів (нульова фаза)"),
                        width="stretch")
    def band_power(f, p, lo, hi):
        sel = (f >= lo) & (f < hi)
        return 10 * np.log10(_trapz(10 ** (p[sel] / 10), f[sel]) + 1e-15) if sel.any() else np.nan
    bands = [("Дрейф ізолінії (< 0.5 Гц)", 0, 0.5), ("Корисна ЕКГ (0.5–40 Гц)", 0.5, 40),
             (f"Мережа ({prep.notch_freq:g} ± 1 Гц)", prep.notch_freq - 1, prep.notch_freq + 1),
             ("ВЧ-шум / ЕМГ (> 40 Гц)", 40, fs / 2)]
    st.dataframe(pd.DataFrame([{"Діапазон": n, "Потужність до, дБ": band_power(f_raw, p_raw, a, b),
                                "Потужність після, дБ": band_power(f_cl, p_cl, a, b)} for n, a, b in bands]),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Вкладка «Ознаки»
# ---------------------------------------------------------------------------
with tabs[2]:
    beats = pr["beats"]
    if len(beats) == 0:
        st.warning("Не вдалося сегментувати жодного циклу — змініть параметри.")
    else:
        tb = (np.arange(beats.shape[1]) - feat.pre) / fs * 1000
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for b in beats[: min(len(beats), 60)]:
                fig.add_trace(go.Scatter(x=tb, y=b, mode="lines", line=dict(color=C_GRAY, width=0.8),
                                         opacity=0.35, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=tb, y=beats.mean(0), name="Середній цикл",
                                     line=dict(color=C_BLUE, width=2.5)))
            fig.add_vline(x=0, line=dict(color=C_ORANGE, dash="dot", width=1))
            fig.update_xaxes(title_text="Час відносно R-піку, мс")
            fig.update_yaxes(title_text="мВ")
            st.plotly_chart(style(fig, 400, f"Сегментовані серцеві цикли ({len(beats)} шт.)"),
                            width="stretch")
        with c2:
            sel_r = (st.select_slider("Радіус для карти кодів LBP", list(feat.radii), value=feat.radii[0])
                     if len(feat.radii) > 1 else feat.radii[0])
            codes = np.vstack([lbp_1d_codes(b, feat.lbp_p, sel_r, feat.threshold) for b in beats[:60]])
            n_bins = 2 ** (2 * feat.lbp_p)
            hist = np.vstack([np.bincount(part.ravel(), minlength=n_bins) / part.size
                              for part in np.array_split(codes, feat.n_windows, axis=1)])
            fig = go.Figure(go.Heatmap(z=hist.T, x=[f"В{i + 1}" for i in range(feat.n_windows)],
                                       y=list(range(n_bins)), colorscale="Blues",
                                       colorbar=dict(title="Частка"),
                                       hovertemplate="Вікно %{x}<br>Код %{y}<br>Частка %{z:.3f}<extra></extra>"))
            fig.update_xaxes(title_text="Часове вікно циклу")
            fig.update_yaxes(title_text="Код 1D-LBP")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(style(fig, 400, f"Гістограми 1DMRLBP за вікнами (радіус {sel_r})"),
                            width="stretch")
        fvec = pr["features"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=fvec.mean(0), name="Середнє", line=dict(color=C_BLUE, width=2)))
        fig.add_trace(go.Scatter(y=fvec.mean(0) + fvec.std(0), name="+1σ",
                                 line=dict(color=C_BLUE, width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(y=np.maximum(fvec.mean(0) - fvec.std(0), 0), name="±1σ", fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.15)", line=dict(width=0)))
        fig.update_xaxes(title_text=f"Індекс ознаки (m = {feature_dim(feat)})")
        fig.update_yaxes(title_text="Значення (L2-нормовано)")
        st.plotly_chart(style(fig, 320, "Базовий вектор ознак запису"), width="stretch")

# ---------------------------------------------------------------------------
# Пакетний прогін
# ---------------------------------------------------------------------------
if run_batch:
    if len(batch_names) < 2:
        st.sidebar.error("Оберіть щонайменше два записи.")
    else:
        bar = st.progress(0.0, text="Завантаження записів…")
        try:
            recs = []
            for i, n in enumerate(batch_names):
                bar.progress(0.3 * i / len(batch_names), text=f"Завантаження {n}…")
                recs.append(cached_record(n, float(duration), float(noise)))
            res = run_experiment(recs, prep, feat, model_p,
                                 progress=lambda p, msg: bar.progress(0.3 + 0.7 * p, text=msg),
                                 with_sensitivity=with_sens)
            st.session_state["result"] = res
            st.session_state["result_ds"] = dataset_label
            bar.empty()
            st.toast("Пакетний прогін завершено", icon="✅")
        except Exception as exc:
            bar.empty()
            st.error(f"Помилка пакетного прогону: {exc}")

res = st.session_state.get("result")
ref = PAPER_RESULTS["MIT-BIH"]

# ---------------------------------------------------------------------------
# Вкладка «Результати»
# ---------------------------------------------------------------------------
with tabs[3]:
    if res is None:
        st.info("Натисніть **«Запустити пакетний прогін»** на панелі зліва, щоб відтворити "
                "експеримент статті (онлайн-навчання за порціями, Accuracy та EER).", icon="👈")
    else:
        st.subheader(f"Порівняння з еталоном авторів (MIT-BIH, табл. 1) · дані: {st.session_state['result_ds']}")
        c = st.columns(4)
        acc, eer = res.final["Accuracy, %"], res.final["EER, %"]
        c[0].metric("Accuracy (онлайн), %", f"{acc:.2f}", delta=f"{acc - ref['Accuracy']:+.2f} п.п. до статті")
        c[1].metric("EER (онлайн), %", f"{eer:.2f}", delta=f"{eer - ref['EER']:+.2f} п.п. до статті",
                    delta_color="inverse")
        c[2].metric("Стаття: Accuracy / EER", f"{ref['Accuracy']:.2f} / {ref['EER']:.2f} %")
        c[3].metric("Пакетне навчання (1 порція)", f"{res.batch_mode['Accuracy, %']:.2f} %",
                    delta=f"EER {res.batch_mode['EER, %']:.2f} %", delta_color="off")
        st.caption(f"Осіб: **{res.n_subjects}** · навчальних зразків: **{res.n_train}** · тестових: "
                   f"**{res.n_test}** · розмірність ознак m = **{res.feature_dim}** · r = **{model_p.r}** · "
                   f"час онлайн-навчання: **{res.final['Час онлайн-навчання, с']:.2f} с**")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.rounds["Раунд"], y=res.rounds["Accuracy, %"], mode="lines+markers",
                                     name="Відтворено", line=dict(color=C_BLUE, width=2),
                                     marker=dict(size=9, line=dict(color="white", width=2))))
            fig.add_hline(y=ref["Accuracy"], line=dict(color=C_ORANGE, dash="dash", width=2),
                          annotation_text=f"Стаття: {ref['Accuracy']} %")
            fig.update_xaxes(title_text="Раунд (порція даних)", dtick=1)
            fig.update_yaxes(title_text="Accuracy, %")
            st.plotly_chart(style(fig, 380, "Точність ідентифікації за раундами (рис. 1 статті)"),
                            width="stretch")
        with c2:
            ec = res.eer_curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ec["thr"], y=100 * ec["FAR"], name="FAR (хибне прийняття)",
                                     line=dict(color=C_ORANGE, width=2)))
            fig.add_trace(go.Scatter(x=ec["thr"], y=100 * ec["FRR"], name="FRR (хибне відхилення)",
                                     line=dict(color=C_BLUE, width=2)))
            fig.add_hline(y=eer, line=dict(color=C_MUTED, dash="dot", width=1),
                          annotation_text=f"EER = {eer:.2f} %")
            fig.update_xaxes(title_text="Поріг евклідової відстані")
            fig.update_yaxes(title_text="Похибка, %")
            st.plotly_chart(style(fig, 380, "Режим верифікації: криві FAR/FRR"), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=res.impostor, name="Чужі (impostor)", marker_color=C_ORANGE,
                                       opacity=0.65, histnorm="probability density", nbinsx=60))
            fig.add_trace(go.Histogram(x=res.genuine, name="Свої (genuine)", marker_color=C_BLUE,
                                       opacity=0.75, histnorm="probability density", nbinsx=40))
            fig.update_layout(barmode="overlay", hovermode="closest")
            fig.update_xaxes(title_text="Відстань «запит–шаблон»")
            fig.update_yaxes(title_text="Щільність")
            st.plotly_chart(style(fig, 380, "Розподіли відстаней"), width="stretch")
        with c2:
            z = np.vstack([res.query_emb, res.enroll_emb])
            xy = pca_2d(z)
            nq = len(res.query_emb)
            subj = sorted(set(res.enroll_labels))
            focus = st.selectbox("Виділити особу", subj, key="focus")
            qm = res.query_labels == focus
            em = res.enroll_labels == focus
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xy[:nq][~qm, 0], y=xy[:nq][~qm, 1], mode="markers", name="Інші запити",
                                     marker=dict(color=C_GRAY, size=8, opacity=0.45),
                                     text=res.query_labels[~qm], hovertemplate="Особа %{text}<extra></extra>"))
            fig.add_trace(go.Scatter(x=xy[:nq][qm, 0], y=xy[:nq][qm, 1], mode="markers",
                                     name=f"Запити {focus}",
                                     marker=dict(color=C_BLUE, size=11, line=dict(color="white", width=2))))
            fig.add_trace(go.Scatter(x=xy[nq:][em, 0], y=xy[nq:][em, 1], mode="markers",
                                     name=f"Шаблон {focus}",
                                     marker=dict(color=C_ORANGE, size=16, symbol="star",
                                                 line=dict(color="white", width=2))))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title_text="ГК 1")
            fig.update_yaxes(title_text="ГК 2")
            st.plotly_chart(style(fig, 380, "Латентні представлення XᵀW (PCA)"), width="stretch")

        c1, c2 = st.columns([3, 2])
        with c1:
            tb1 = PAPER_TABLE1.copy()
            tb1.loc[len(tb1)] = [f"Відтворення ({st.session_state['result_ds']})", round(eer, 2), round(acc, 2)]
            colors = [C_GRAY] * (len(tb1) - 2) + [C_ORANGE, C_BLUE]
            fig = go.Figure(go.Bar(x=tb1["Accuracy, %"], y=tb1["Метод"], orientation="h",
                                   marker_color=colors, text=[f"{v:.2f}" for v in tb1["Accuracy, %"]],
                                   textposition="outside"))
            fig.update_xaxes(title_text="Accuracy, %", range=[min(80, tb1["Accuracy, %"].min() - 2), 101])
            fig.update_layout(hovermode="closest", showlegend=False)
            st.plotly_chart(style(fig, 360, "Порівняння з методами з табл. 1 (MIT-BIH)"),
                            width="stretch")
        with c2:
            st.markdown("**Таблиця 1 статті + відтворення**")
            st.dataframe(tb1, hide_index=True, width="stretch")

        st.markdown("**Раунди онлайн-навчання**")
        st.dataframe(res.rounds.round(3), hide_index=True, width="stretch")
        with st.expander("Збіжність цільової функції (рівн. 4) за ітераціями"):
            fig = go.Figure()
            for i, lh in enumerate(res.loss_history, start=1):
                fig.add_trace(go.Scatter(x=list(range(1, len(lh) + 1)), y=lh, mode="lines+markers",
                                         name=f"Раунд {i}", line=dict(width=2),
                                         marker=dict(size=8)))
            fig.update_xaxes(title_text="Ітерація", dtick=1)
            fig.update_yaxes(title_text="Значення цільової функції", type="log")
            st.plotly_chart(style(fig, 340), width="stretch")
        with st.expander("Показники за записами (детекція R-піків, SNR)"):
            st.dataframe(res.per_record.round(2), hide_index=True, width="stretch")

    # Звіт Excel (доступний завжди: з пакетним прогоном або лише для поточного запису)
    single = {"Запис": rec.name, "Джерело": rec.source, "R-піків": len(pr["peaks"]),
              "Циклів": len(pr["beats"]), "ЧСС, уд/хв": round(pr["hr"], 1),
              **{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in det.items()}}
    if "snr_after" in pr:
        single["SNR до, дБ"] = round(pr["snr_before"], 2)
        single["SNR після, дБ"] = round(pr["snr_after"], 2)
    xlsx = build_excel_report(res, single,
                              params=res.params if res else {"Попередня обробка": prep.to_dict(),
                                                             "Ознаки": feat.to_dict(),
                                                             "Модель": model_p.to_dict()})
    st.download_button("⬇️ Завантажити звіт (.xlsx)", data=xlsx, file_name="zvit_ecg_oeb_hc.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")

# ---------------------------------------------------------------------------
# Вкладка «Чутливість»
# ---------------------------------------------------------------------------
with tabs[4]:
    if res is None or res.sensitivity is None:
        st.info("Увімкніть «+ аналіз чутливості» на панелі зліва та запустіть пакетний прогін. "
                "Кожен параметр змінюється в діапазоні 10⁻³…10³ при решті = 1 (як на рис. 2).", icon="🎚️")
    else:
        s = res.sensitivity
        names = {"alpha": "α", "beta": "β", "theta": "θ", "delta": "δ"}
        cols = {"alpha": C_BLUE, "beta": C_ORANGE, "theta": C_AQUA, "delta": "#eda100"}
        fig = go.Figure()
        for p_, g in s.groupby("Параметр", sort=False):
            fig.add_trace(go.Scatter(x=g["Значення"], y=g["Accuracy, %"], mode="lines+markers",
                                     name=names[p_], line=dict(color=cols[p_], width=2),
                                     marker=dict(size=9, line=dict(color="white", width=2))))
        fig.update_xaxes(type="log", title_text="Значення параметра")
        fig.update_yaxes(title_text="Accuracy, %")
        st.plotly_chart(style(fig, 420, "Чутливість до параметрів балансу (рис. 2 статті)"),
                        width="stretch")
        st.dataframe(s.pivot(index="Значення", columns="Параметр", values="Accuracy, %").round(2),
                     width="stretch")

# ---------------------------------------------------------------------------
# Вкладка «Методика»
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown(r"""
### Конвеєр обробки
1. **Попередня обробка** — видалення дрейфу ізолінії (двоступенева медіана 200/600 мс або ФВЧ),
   режекція 50/60 Гц, смуговий фільтр Баттерворта з нульовою фазою.
2. **Детекція R-піків** — алгоритм Пана–Томпкінса [19].
3. **Сегментація** — фіксована кількість відліків з обох боків R-піку (260 для MIT-BIH).
4. **Базові ознаки 1DMRLBP** [7] — гістограми 1D-LBP кодів у часових вікнах для кількох радіусів.
5. **OEB-HC** — онлайн-матрична факторизація, кероване кодами Адамара:

$$\min_{W,G,\vec V}\; \beta\|X - G V^\top\|_F^2 + \alpha\|V - X^\top W\|_F^2 + \theta\|H - V\|_F^2 + \delta(\|W\|_F^2+\|G\|_F^2)$$

   Розв'язки в замкненій формі: $W=(C_1+\tfrac{\delta}{\alpha}I)^{-1}C_2$,
   $G=C_2(C_3+\tfrac{\delta}{\beta}I)^{-1}$,
   $\vec V=(\alpha\vec X^\top W+\beta\vec X^\top G+\theta\vec H)((\alpha+\theta)I+\beta G^\top G)^{-1}$,
   де $C_1=\sum XX^\top$, $C_2=\sum XV$, $C_3=\sum V^\top V$ накопичуються між раундами.
6. **Зіставлення** — евклідова відстань між $X_q^\top W$ і шаблонами $X_e^\top W$; метрики Accuracy та EER.
""")
    st.markdown("**Налаштування експерименту в статті**")
    st.dataframe(pd.DataFrame(PAPER_SETUP.items(), columns=["Параметр", "Значення"]),
                 hide_index=True, width="stretch")
    st.markdown("""
**Відмінності відтворення від оригіналу**
* Стаття не описує фільтрацію — використано стандартний ланцюжок для MIT-BIH.
* 1DMRLBP реалізовано в спрощеному варіанті (без часової динаміки між кількома циклами).
* Шаблони реєстрації будуються з навчальних зразків, що вже надійшли до поточного раунду.
* Матриця Адамара будується за Сильвестром (визначення $(-1)^{(i-1)(j-1)}$ у статті не є ортогональним).
""")
