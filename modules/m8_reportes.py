"""Módulo 8. Reportes: el usuario elige qué incluir en Excel, Word y PDF."""
import io
import os
import re
from datetime import date
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st

from modules import m2_variables as m2
from modules.m1_datos import resumen_por_subgrupo
from modules.m3b_calidad import EJEMPLO_ISHIKAWA, EJEMPLO_PARETO

ORDEN = ["datos", "estabilidad", "supuestos", "capacidad", "atributos", "arl", "aceptacion",
         "pareto", "ishikawa", "causas"]
ETQ_SUP = {"n": "Observaciones evaluadas", "shapiro_p": "Shapiro-Wilk (p-valor)",
           "ad_rechaza": "Anderson-Darling rechaza normalidad", "levene_p": "Levene (p-valor)",
           "bartlett_p": "Bartlett (p-valor)", "ljung_p": "Ljung-Box (p-valor)",
           "dw": "Durbin-Watson", "rachas_p": "Rachas (p-valor)",
           "spearman_p": "Tendencia, Spearman (p-valor)", "senales": "Señales de falta de control"}
AZ, RJ, VD = "#2F6FDE", "#E5484D", "#2BAE66"


# ---------------------------------------------------------------- disponibilidad
def _f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def _causas_df():
    c = st.session_state.get("causas_tabla")
    if c is None:
        return None
    c = c.replace("", np.nan).dropna(how="all").fillna("")
    return c if len(c) else None


def _pareto_real():
    t = st.session_state.get("pareto")
    if t is None:
        return False
    actual = sorted(zip(t["Tipo de defecto"], t["Frecuencia"]))
    ejemplo = sorted(zip(EJEMPLO_PARETO["Tipo de defecto"], EJEMPLO_PARETO["Frecuencia"]))
    return actual != ejemplo


def _ishikawa_real():
    ish = st.session_state.get("ishikawa")
    if not ish:
        return False
    causas = {c for lista in ish["causas"].values() for c in lista}
    return bool(causas) and causas != set(EJEMPLO_ISHIKAWA["Causa"])


def disponibles():
    """Secciones con resultados disponibles en esta sesión (id -> nombre)."""
    ss = st.session_state
    todo = {
        "datos": ("Datos y resumen estadístico", ss.get("df") is not None or ss.get("datos_attr") is not None),
        "estabilidad": ("Gráfico de control de variables", bool(ss.get("limites"))),
        "supuestos": ("Validación de supuestos", bool(ss.get("supuestos"))),
        "capacidad": ("Capacidad del proceso", bool(ss.get("capacidad"))),
        "atributos": ("Gráficos de control por atributos", bool(ss.get("atributos"))),
        "arl": ("Diseño del gráfico: ARL0, ARL1 y ATS", bool(ss.get("arl_params"))),
        "aceptacion": ("Muestreo de aceptación", bool(ss.get("plan_aceptacion"))),
        "pareto": ("Diagrama de Pareto", _pareto_real()),
        "ishikawa": ("Diagrama de Ishikawa", _ishikawa_real()),
        "causas": ("Registro de causas especiales", _causas_df() is not None),
    }
    return {k: todo[k][0] for k in ORDEN if todo[k][1]}


# ---------------------------------------------------------------- tablas
def tabla_supuestos(sup):
    filas = []
    for k, etq in ETQ_SUP.items():
        v = sup.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, bool):
            val = "Sí" if v else "No"
        elif isinstance(v, (float, np.floating)):
            val = f"{v:.4f}"
        else:
            val = str(v)
        filas.append({"Prueba o dato": etq, "Valor": val})
    return pd.DataFrame(filas)


def tabla_limites(lim):
    return pd.DataFrame({
        "Parámetro": ["Gráfico", "n", "Línea central", "LCI", "LCS", "σ estimada", "Puntos excluidos",
                      "Señales", "Señales en Fase II"],
        "Valor": [lim["tipo"], lim["n"], round(lim["media"], 4), round(lim["lcl"], 4), round(lim["ucl"], 4),
                  round(lim["sigma"], 4), ", ".join(map(str, lim["excluidos"])) or "ninguno",
                  lim["n_senales"], lim["alarmas_ii"]]}).astype(str)


def tabla_plan(pl):
    return pd.DataFrame([(k, str(v)) for k, v in pl.items()], columns=["Parámetro", "Valor"])


def tabla_ishikawa():
    ish = st.session_state["ishikawa"]
    return pd.DataFrame([(cat, c) for cat, lista in ish["causas"].items() for c in lista],
                        columns=["Categoría (6M)", "Causa"])

def _desempeno_control(lim):
    """Texto con α, ARL y ATS del gráfico de control (si ya fueron calculados en el Módulo 2)."""
    if lim.get("arl0") is None:
        return ""
    u = lim.get("unidad", "")
    return (f" Desempeño del gráfico (límites a 3σ): α = {lim['alfa']:.5f}, ARL0 = {lim['arl0']:.1f} muestras "
            f"(ATS0 = {lim['ats0']:.1f} {u}); para un desplazamiento de {lim['delta']:g}σ, "
            f"ARL1 = {lim['arl1']:.2f} muestras (ATS1 = {lim['ats1']:.2f} {u}).")


def _cap_en_control(cap):
    """Texto con Cp y Cpk si el módulo de capacidad ya se ejecutó."""
    if not cap or cap.get("Cpk") is None:
        return ""
    return f" Capacidad del proceso: Cp = {_f(cap.get('Cp'))} y Cpk = {_f(cap['Cpk'])}."
# ---------------------------------------------------------------- conclusiones
def conclusiones():
    """Lista de (id, título, texto) con conclusiones preliminares de lo calculado."""
    ss = st.session_state
    out = []
    partes = []
    if ss.get("df") is not None:
        df, col = ss["df"], ss["col_valor"]
        partes.append(f"Se analizaron {len(df)} observaciones de la variable {col}, organizadas en "
                      f"{df['_subgrupo'].nunique()} subgrupos (media = {df[col].mean():.4f}, "
                      f"desviación estándar = {df[col].std():.4f}).")
    if ss.get("datos_attr") is not None:
        partes.append(f"Se cargaron {len(ss['datos_attr'])} muestras de atributos.")
    if partes:
        out.append(("datos", "1. Datos analizados", " ".join(partes)))
    lim = ss.get("limites")
    if lim:
        ex = lim["excluidos"]
        out.append(("estabilidad", "2. Estabilidad (gráficos de control)",
                    f"Se construyó un gráfico {lim['tipo']} (n = {lim['n']}). Con la Fase I depurada, la línea "
                    f"central es {lim['media']:.4f}, los límites van de {lim['lcl']:.4f} a {lim['ucl']:.4f} y "
                    f"σ estimada = {lim['sigma']:.4f}. Se excluyeron {len(ex)} punto(s) en Fase I"
                    f"{' (' + ', '.join(map(str, ex)) + ')' if ex else ''}. Se detectaron {lim['n_senales']} "
                    f"señal(es) en total, de las cuales {lim['alarmas_ii']} están en Fase II."))
    sup = ss.get("supuestos")
    if sup and not np.isnan(sup.get("shapiro_p", np.nan)):
        alfa_sup = sup.get("alfa", 0.05)
        norm = "se rechaza" if sup["shapiro_p"] < alfa_sup else "no se rechaza"
        txt = (f"Con {sup['n']} observaciones, según Shapiro-Wilk (p = {sup['shapiro_p']:.4f}) {norm} la "
               "normalidad.")
        if not np.isnan(sup.get("ad_p", np.nan)):
            ad_txt = "también rechaza" if sup.get("ad_rechaza") else "no rechaza"
            txt += f" Anderson-Darling (p = {sup['ad_p']:.4f}) {ad_txt} la normalidad."
        if not np.isnan(sup.get("ljung_p", np.nan)):
            ind = "hay evidencia de autocorrelación" if sup["ljung_p"] < alfa_sup else "no hay evidencia de autocorrelación"
            txt += f" Según Ljung-Box (p = {sup['ljung_p']:.4f}) {ind}."
        if sup.get("levene_p") is not None:
            hom = "hay evidencia de varianzas distintas" if sup["levene_p"] < alfa_sup else "no hay evidencia de varianzas distintas"
            txt += f" Según Levene (p = {sup['levene_p']:.4f}) {hom}."
        txt += f" La revisión de estabilidad encontró {sup['senales']} señal(es)."
        out.append(("supuestos", "3. Validación de supuestos", txt))
    cap = ss.get("capacidad")
    if cap:
        out.append(("capacidad", "4. Capacidad del proceso",
                    f"Especificaciones: LEI = {_f(cap['lei'], 4)}, LES = {_f(cap['les'], 4)}. "
                    f"Cp = {_f(cap['Cp'])}, Cpk = {_f(cap['Cpk'])}, Pp = {_f(cap['Pp'])}, Ppk = {_f(cap['Ppk'])}. "
                    f"{cap['texto']} La fracción no conforme esperada con σ global es "
                    f"{cap['ppm_global']:.1f} PPM. {cap.get('conclusion_tecnica', '')}"
                    f"{' Los índices se calcularon con transformación Box-Cox.' if cap['transformada'] else ''}"))
    at = ss.get("atributos")
    if at:
        out.append(("atributos", "5. Gráficos por atributos",
                    f"Se construyó un gráfico {at['tipo']} con {at['n_puntos']} muestras; parámetro central "
                    f"= {at['par']:.5f}. Se excluyeron {len(at['excluidos'])} punto(s) en Fase I y se "
                    f"detectaron {at['n_senales']} señal(es), {at['alarmas_ii']} en Fase II."))
    ar = ss.get("arl_params")
    if ar:
        out.append(("arl", "6. Diseño del gráfico (ARL/ATS)",
                    f"Con L = {ar['L']:.3f}, n = {ar['n']} y h = {ar['h']} {ar['unidad']}: ARL0 = {ar['arl0']:.0f} "
                    f"muestras (ATS0 = {ar['ats0']:.1f} {ar['unidad']}). Un desplazamiento de 1σ se detecta en "
                    f"promedio en {ar['arl1_d1']:.1f} muestras."))
    pl = ss.get("plan_aceptacion")
    if pl:
        ltpd = f", Pa(NRC) = {pl['pa_ltpd']:.3f}" if pl["pa_ltpd"] is not None else ""
        out.append(("aceptacion", "7. Muestreo de aceptación",
                    f"Plan simple n = {pl['n']}, c = {pl['c']} (lote N = {pl['N']}). Pa(NAC) = "
                    f"{pl['pa_aql']:.3f}{ltpd}; AOQL = {pl['aoql'] * 100:.2f} %. "
                    f"Riesgo real del productor = {1-pl['pa_aql']:.3f} frente a α objetivo = {pl['alfa']:.3f}; "
                    f"riesgo real del consumidor = {pl['pa_ltpd']:.3f} frente a β objetivo = {pl['beta']:.3f}. "
                    f"{'El plan cumple los dos riesgos objetivo.' if (1-pl['pa_aql']) <= pl['alfa'] and pl['pa_ltpd'] <= pl['beta'] else 'El plan no cumple al menos uno de los riesgos objetivo y requiere ajuste.'}"))
    return out


# ---------------------------------------------------------------- figuras (matplotlib → PNG)
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _png(fig):
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _leyenda(ax, n=4):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=n, frameon=False, fontsize=8)


def fig_boxplot():
    plt = _plt()
    ss = st.session_state
    x = ss["df"][ss["col_valor"]].dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.boxplot(x, vert=False, showmeans=True, widths=0.5,
               boxprops=dict(color=AZ), medianprops=dict(color=RJ), meanprops=dict(markerfacecolor=VD))
    ax.scatter(x, 1 + np.random.default_rng(1).uniform(-0.18, 0.18, len(x)), s=8, color=AZ, alpha=0.5,
               label="Observaciones")
    ax.set_title("Diagrama de caja de la variable de calidad", fontweight="bold")
    ax.set_xlabel(ss["col_valor"])
    ax.set_yticks([])
    ax.grid(alpha=0.3)
    _leyenda(ax, 1)
    return _png(fig)


def fig_control():
    plt = _plt()
    ss = st.session_state
    lim = ss["limites"]
    principal, _, _ = m2.construir_series(ss["df"], ss["col_valor"], lim["tipo"])
    y = principal.values.astype(float)
    x = np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, y, "-o", color=AZ, ms=4, lw=1, label="Valor del subgrupo")
    ax.axhline(lim["ucl"], color=RJ, ls="--", label="LCS")
    ax.axhline(lim["media"], color=VD, label="Línea central")
    ax.axhline(lim["lcl"], color=RJ, ls="--", label="LCI")
    fuera = (y > lim["ucl"]) | (y < lim["lcl"])
    ax.plot(x[fuera], y[fuera], "o", color=RJ, ms=8, label="Fuera de límites")
    if lim["n_f1"] < len(y):
        ax.axvline(lim["n_f1"] + 0.5, color="gray", ls=":", label="Inicio de Fase II")
    ax.set_title(f"Gráfico de control {lim['tipo']}", fontweight="bold")
    ax.set_xlabel("Subgrupo / observación")
    ax.set_ylabel(ss["col_valor"])
    ax.grid(alpha=0.3)
    _leyenda(ax, 5)
    return _png(fig)


def fig_capacidad():
    from scipy import stats
    plt = _plt()
    cap = st.session_state["capacidad"]
    x = cap["x"]
    xs = np.linspace(x.min() - 3 * cap["sigma_global"], x.max() + 3 * cap["sigma_global"], 300)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(x, bins=20, density=True, color="#9DB8F2", edgecolor="white", label="Datos")
    ax.plot(xs, stats.norm.pdf(xs, cap["mu"], cap["sigma_global"]), color=AZ, lw=2, label="Normal (σ global)")
    for v, nom in ((cap["lei"], "LEI"), (cap["les"], "LES")):
        if v is not None:
            ax.axvline(v, color=RJ, ls="--", label=nom)
    ax.axvline(cap["mu"], color=VD, label="Media")
    ax.set_title("Distribución del proceso frente a las especificaciones", fontweight="bold")
    ax.set_xlabel("Valor de la variable")
    ax.set_ylabel("Densidad")
    ax.grid(alpha=0.3)
    _leyenda(ax, 5)
    return _png(fig)


def fig_atributos():
    plt = _plt()
    at = st.session_state["atributos"]
    t = at["tabla"]
    x = np.arange(1, len(t) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, t["Valor"], "-o", color=AZ, ms=4, lw=1, label="Valor de la muestra")
    ax.step(x, t["LCS"], where="mid", color=RJ, ls="--", label="LCS")
    ax.step(x, t["LC"], where="mid", color=VD, label="Línea central")
    ax.step(x, t["LCI"], where="mid", color=RJ, ls="--", label="LCI")
    fuera = (t["Valor"] > t["LCS"]) | (t["Valor"] < t["LCI"])
    ax.plot(x[fuera.to_numpy()], t["Valor"][fuera], "o", color=RJ, ms=8, label="Fuera de límites")
    if at["n_f1"] < len(t):
        ax.axvline(at["n_f1"] + 0.5, color="gray", ls=":", label="Inicio de Fase II")
    ax.set_title(f"Gráfico de control {at['tipo']}", fontweight="bold")
    ax.set_xlabel("Número de muestra")
    ax.grid(alpha=0.3)
    _leyenda(ax, 5)
    return _png(fig)


def fig_pareto():
    plt = _plt()
    t = st.session_state["pareto"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(t["Tipo de defecto"], t["Frecuencia"], color=AZ, label="Frecuencia")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    ax2.plot(t["Tipo de defecto"], t["% acumulado"], "-o", color=RJ, label="% acumulado")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% acumulado")
    ax.set_title("Diagrama de Pareto", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False,
              fontsize=8)
    return _png(fig)


# ---------------------------------------------------------------- Excel
def hojas_excel():
    ss = st.session_state
    h = {}
    if ss.get("df") is not None:
        df, col = ss["df"], ss["col_valor"]
        h["Datos"] = df
        if ss.get("resumen_est") is not None:
            h["Resumen_estadistico"] = ss["resumen_est"]
        h["Resumen_subgrupos"] = resumen_por_subgrupo(df, col).reset_index()
    if ss.get("datos_attr") is not None:
        h["Datos_atributos"] = ss["datos_attr"].rename(columns={"etq": "Muestra", "x": "Conteo"})
    if ss.get("limites"):
        h["Limites_control"] = tabla_limites(ss["limites"])
    if ss.get("supuestos"):
        h["Supuestos"] = tabla_supuestos(ss["supuestos"])
    if ss.get("capacidad"):
        h["Capacidad_indices"] = ss["capacidad"]["tabla_indices"]
        h["Capacidad_FNC"] = ss["capacidad"]["tabla_fnc"]
    if ss.get("atributos"):
        h["Atributos_control"] = ss["atributos"]["tabla"]
    if ss.get("arl_params"):
        h["ARL_ATS"] = ss["arl_params"]["tabla"]
    if ss.get("arl_comp") is not None:
        h["ARL_n_vs_h"] = ss["arl_comp"]
    if ss.get("plan_aceptacion"):
        h["Muestreo_aceptacion"] = tabla_plan(ss["plan_aceptacion"])
    if _pareto_real():
        h["Pareto"] = ss["pareto"]
    if _ishikawa_real():
        h["Ishikawa"] = tabla_ishikawa()
    if _causas_df() is not None:
        h["Causas_especiales"] = _causas_df()
    return h


def construir_excel(hojas):
    from openpyxl.utils import get_column_letter
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nombre, d in hojas.items():
            d.to_excel(w, sheet_name=nombre[:31], index=False)
            ws = w.sheets[nombre[:31]]
            for i, c in enumerate(d.columns, start=1):
                ws.column_dimensions[get_column_letter(i)].width = min(45, max(12, len(str(c)) + 3))
    return buf.getvalue()


# ---------------------------------------------------------------- bloques comunes de Word y PDF
def bloques(secs, textos, notas, con_tablas, con_figs):
    """Contenido del reporte como lista de bloques: h1, p, caption, tabla, img."""
    ss = st.session_state
    B, nt, nf = [], [0], [0]

    def tabla(tit, df, d=4):
        if not con_tablas or df is None or len(df) == 0:
            return
        nt[0] += 1
        B.append(("caption", f"Tabla {nt[0]}. {tit}"))
        B.append(("tabla", df, d))

    def figura(tit, fn):
        if not con_figs:
            return
        try:
            png = fn()
        except Exception as e:  # noqa: BLE001
            B.append(("p", f"(No se pudo generar la figura «{tit}»: {e})"))
            return
        nf[0] += 1
        B.append(("img", png))
        B.append(("caption", f"Figura {nf[0]}. {tit}"))

    def seccion(id_, titulo):
        B.append(("h1", titulo))
        if textos.get(id_, "").strip():
            B.append(("p", textos[id_]))

    if "datos" in secs:
        seccion("datos", "Datos analizados")
        if ss.get("df") is not None:
            tabla("Resumen estadístico de " + ss["col_valor"], ss.get("resumen_est"))
            figura("Diagrama de caja de la variable de calidad", fig_boxplot)
            rs = resumen_por_subgrupo(ss["df"], ss["col_valor"]).reset_index()
            if len(rs) <= 60:
                tabla("Resumen por subgrupo", rs)
        if ss.get("datos_attr") is not None:
            tabla("Datos de atributos (primeras 40 muestras)",
                  ss["datos_attr"].head(40).rename(columns={"etq": "Muestra", "x": "Conteo"}), 3)
    if "estabilidad" in secs:
        seccion("estabilidad", "Estabilidad del proceso")
        tabla("Límites de control y señales", tabla_limites(ss["limites"]))
        figura(f"Gráfico de control {ss['limites']['tipo']} (Fase I y Fase II)", fig_control)
    if "supuestos" in secs:
        seccion("supuestos", "Validación de supuestos")
        tabla("Resultados de la validación de supuestos", tabla_supuestos(ss["supuestos"]))
    if "capacidad" in secs:
        seccion("capacidad", "Capacidad del proceso")
        tabla("Índices de capacidad", ss["capacidad"]["tabla_indices"], 3)
        tabla("Fracción no conforme (PPM)", ss["capacidad"]["tabla_fnc"], 1)
        figura("Distribución del proceso frente a las especificaciones", fig_capacidad)
    if "atributos" in secs:
        seccion("atributos", "Gráficos de control por atributos")
        figura(f"Gráfico de control {ss['atributos']['tipo']}", fig_atributos)
        tabla("Valores y límites del gráfico", ss["atributos"]["tabla"], 5)
    if "arl" in secs:
        seccion("arl", "Diseño del gráfico: ARL0, ARL1 y ATS")
        tabla("ARL1 y ATS1 según el desplazamiento", ss["arl_params"]["tabla"], 2)
    if "aceptacion" in secs:
        seccion("aceptacion", "Muestreo de aceptación")
        tabla("Plan de muestreo evaluado", tabla_plan(ss["plan_aceptacion"]))
    if "pareto" in secs:
        B.append(("h1", "Diagrama de Pareto"))
        top = ss["pareto"]["Tipo de defecto"].iloc[0]
        B.append(("p", f"El defecto con mayor frecuencia es «{top}»."))
        tabla("Frecuencias y porcentaje acumulado de defectos", ss["pareto"].round(2), 2)
        figura("Diagrama de Pareto de los defectos", fig_pareto)
    if "ishikawa" in secs:
        B.append(("h1", "Diagrama de Ishikawa (causa-efecto)"))
        B.append(("p", f"Efecto analizado: {ss['ishikawa']['efecto']}."))
        tabla("Causas identificadas por categoría (6M)", tabla_ishikawa())
    if "causas" in secs:
        B.append(("h1", "Registro de causas especiales"))
        tabla("Causas especiales y acciones correctivas", _causas_df())
    resumenes = [txt for ident, titulo, txt in conclusiones() if ident in secs]
    if resumenes:
        B.append(("h1", "Conclusión general integrada"))
        B.append(("p", "A partir de los módulos ejecutados, se obtiene la siguiente síntesis de los hallazgos: " + " ".join(resumenes)))
    if notas.strip():
        B.append(("h1", "Observaciones del equipo"))
        B.append(("p", notas))
    return B


def _fmt(v, d):
    if isinstance(v, (float, np.floating)):
        return "—" if np.isnan(v) else f"{v:.{d}f}"
    return str(v)


# ---------------------------------------------------------------- Word
def a_word(meta, B):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading(meta["titulo"], 0)
    for etq, v in (("Empresa / proceso", meta["empresa"]), ("Autores", meta["autores"])):
        if v.strip():
            doc.add_paragraph(f"{etq}: {v}")
    doc.add_paragraph(f"Fecha: {date.today():%d/%m/%Y}")
    doc.add_paragraph("Síntesis de resultados y conclusiones estadísticas derivadas de los análisis ejecutados.")
    for b in B:
        if b[0] == "h1":
            doc.add_heading(b[1], 1)
        elif b[0] == "p":
            doc.add_paragraph(b[1])
        elif b[0] == "caption":
            doc.add_paragraph().add_run(b[1]).italic = True
        elif b[0] == "tabla":
            df, d = b[1], b[2]
            t = doc.add_table(rows=1, cols=len(df.columns))
            t.style = "Table Grid"
            for i, c in enumerate(df.columns):
                t.rows[0].cells[i].text = str(c)
                for r in t.rows[0].cells[i].paragraphs[0].runs:
                    r.bold = True
            for fila in df.itertuples(index=False):
                celdas = t.add_row().cells
                for i, v in enumerate(fila):
                    celdas[i].text = _fmt(v, d)
            doc.add_paragraph()
        elif b[0] == "img":
            doc.add_picture(io.BytesIO(b[1]), width=Inches(6))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- PDF
_FUENTES = {"listas": False}


def _registrar_fuentes():
    """Usa las fuentes DejaVu que trae matplotlib, para que se vean σ, ≥, α, etc."""
    if _FUENTES["listas"]:
        return
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    ruta = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for nombre, archivo in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                            ("DV-I", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(nombre, os.path.join(ruta, archivo)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
    _FUENTES["listas"] = True


def _limpia(t):
    """Escapa el texto para reportlab y reemplaza la barra combinada (X̄ → X-barra)."""
    t = re.sub("(.)\u0304", r"\1-barra", str(t))
    return escape(t).replace("\n", "<br/>")


def a_pdf(meta, B):
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    _registrar_fuentes()
    s_tit = ParagraphStyle("t", fontName="DV-B", fontSize=20, leading=26, alignment=1, spaceAfter=10)
    s_sub = ParagraphStyle("s", fontName="DV", fontSize=10, leading=14, alignment=1, textColor=colors.grey)
    s_h1 = ParagraphStyle("h", fontName="DV-B", fontSize=14, leading=18, spaceBefore=14, spaceAfter=6,
                          textColor=colors.HexColor("#3E2FC0"))
    s_p = ParagraphStyle("p", fontName="DV", fontSize=9.5, leading=14, spaceAfter=6)
    s_cap = ParagraphStyle("c", fontName="DV-I", fontSize=8.5, leading=11, textColor=colors.HexColor("#5F6488"),
                           spaceBefore=4, spaceAfter=4)
    s_th = ParagraphStyle("th", fontName="DV-B", fontSize=7.5, leading=9)
    s_td = ParagraphStyle("td", fontName="DV", fontSize=7.5, leading=9)

    story = [Spacer(1, 1.2 * cm), Paragraph(_limpia(meta["titulo"]), s_tit)]
    for etq, v in (("Empresa / proceso", meta["empresa"]), ("Autores", meta["autores"])):
        if v.strip():
            story.append(Paragraph(_limpia(f"{etq}: {v}"), s_sub))
    story.append(Paragraph(f"Fecha: {date.today():%d/%m/%Y}", s_sub))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Síntesis de resultados y conclusiones estadísticas calculadas a partir de los módulos ejecutados.", s_cap))
    for b in B:
        if b[0] == "h1":
            story.append(Paragraph(_limpia(b[1]), s_h1))
        elif b[0] == "p":
            story.append(Paragraph(_limpia(b[1]), s_p))
        elif b[0] == "caption":
            story.append(Paragraph(_limpia(b[1]), s_cap))
        elif b[0] == "tabla":
            df, d = b[1], b[2]
            datos = [[Paragraph(_limpia(c), s_th) for c in df.columns]]
            for fila in df.itertuples(index=False):
                datos.append([Paragraph(_limpia(_fmt(v, d)), s_td) for v in fila])
            t = Table(datos, colWidths=[17 * cm / len(df.columns)] * len(df.columns), repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E2FF")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8BCD6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7FD")])]))
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))
        elif b[0] == "img":
            iw, ih = PILImage.open(io.BytesIO(b[1])).size
            ancho = 16 * cm
            story.append(Image(io.BytesIO(b[1]), width=ancho, height=ancho * ih / iw))

    def pie(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title=meta["titulo"]).build(story, onFirstPage=pie,
                                                                       onLaterPages=pie)
    return buf.getvalue()


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Reportes")
    disp = disponibles()
    if not disp:
        st.warning("Todavía no hay resultados. Cargue datos en **Datos** y ejecute los análisis en los "
                   "demás módulos; aquí podrá elegir qué incluir en el reporte.")
        return
    st.caption("El informe PDF consolida los resultados, tablas, figuras y conclusiones de los módulos ejecutados. "
               "Los apartados sin análisis disponible no se incorporan. Los diagramas de Pareto e Ishikawa "
               "solo aparecen cuando se reemplazan los datos de ejemplo por datos reales.")

    # ---- 1. Contenido ----
    st.subheader("1. Informe técnico consolidado en PDF")
    with st.container(border=True):
        c = st.columns(3)
        titulo = c[0].text_input("Título", "Reporte de análisis de Control Estadístico de Procesos", key="r_tit")
        empresa = c[1].text_input("Empresa / proceso analizado", key="r_emp")
        autores = c[2].text_input("Autores", key="r_aut")
        st.markdown("**Secciones a incluir**")
        cols = st.columns(3)
        secs = [k for i, k in enumerate(disp)
                if cols[i % 3].checkbox(disp[k], value=True, key=f"r_sec_{k}")]
        o = st.columns(2)
        con_tablas = o[0].checkbox("Incluir tablas", True, key="r_tab")
        con_figs = o[1].checkbox("Incluir figuras", True, key="r_fig")

    textos = {}
    concl = [c_ for c_ in conclusiones() if c_[0] in secs]
    if concl:
        with st.expander("Editar las conclusiones antes de exportar", expanded=False):
            if st.button("Regenerar conclusiones automáticas"):
                for k in [k for k in st.session_state if k.startswith("concl_")]:
                    del st.session_state[k]
            for id_, tit, txt in concl:
                textos[id_] = st.text_area(tit, value=txt, key=f"concl_{id_}", height=110)
    notas = st.text_area("Observaciones del equipo (opcional)", key="notas_equipo",
                         height=110)

    # ---- 2. Excel ----
    st.subheader("2. Datos complementarios para exportación Excel")
    hojas = hojas_excel()
    previo = st.session_state.get("r_hojas")
    st.session_state["r_hojas"] = list(hojas) if previo is None else [h for h in previo if h in hojas]
    sel_h = st.multiselect("Hojas a incluir", list(hojas), key="r_hojas")

    # ---- 3. Descargas ----
    st.subheader("3. Exportación de resultados")
    d = st.columns(2)
    with d[0]:
        st.markdown("**Excel**")
        if sel_h:
            st.download_button("⬇️ Descargar Excel", construir_excel({h: hojas[h] for h in sel_h}),
                               file_name="resultados_cep.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               width="stretch")
        else:
            st.info("Seleccione al menos una hoja.")

    meta = {"titulo": titulo, "empresa": empresa, "autores": autores}
    B = None
    if secs or notas.strip():
        try:
            B = bloques(secs, textos, notas, con_tablas, con_figs)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo armar el reporte: {e}")
    with d[1]:
        st.markdown("**PDF**")
        if B is None:
            st.info("Seleccione al menos una sección.")
        else:
            try:
                st.download_button("⬇️ Descargar PDF", a_pdf(meta, B), file_name="reporte_cep.pdf",
                                   mime="application/pdf", width="stretch")
            except ImportError:
                st.error("Falta reportlab: ejecute  pip install reportlab")
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo generar el PDF: {e}")
    st.caption("El PDF integra las conclusiones disponibles de los módulos ejecutados, junto con las tablas y figuras seleccionadas. Las observaciones del equipo se incorporan como apartado independiente.")
