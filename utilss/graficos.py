"""Estilo común de los gráficos: marco, título centrado y leyenda automática."""
import numpy as np
import plotly.graph_objects as go

AZUL, ROJO, VERDE, NARANJA, GRIS = "#287FC4", "#D95B64", "#2BAE66", "#C58A36", "#9A8790"
TEXTO = "#3B1724"

# Significado de los colores de punto en los gráficos de control
NOMBRES_COLOR = {
    AZUL: ("Dentro de control", "circle"),
    ROJO: ("Señal / fuera de límites", "circle"),
    NARANJA: ("Alerta (regla de zonas / falsa alarma)", "diamond"),
    GRIS: ("Excluido de Fase I", "x"),
}


def _leyenda_automatica(fig):
    """Agrega a la leyenda las líneas de referencia (LCS, LC, LCI, NAC...) y el significado
    de los colores de los puntos en los gráficos de control."""
    nombres = {t.name for t in fig.data if t.name}
    anotaciones = list(fig.layout.annotations)
    lineas = []
    for s in fig.layout.shapes:
        if s.type != "line" or not (s.line and s.line.color):
            continue
        texto = None
        for a in anotaciones:
            if not a.text:
                continue
            if s.x0 == s.x1 and a.x == s.x0 and a.xref == s.xref:      # línea vertical
                texto = a.text
                break
            if s.y0 == s.y1 and a.y == s.y0 and a.yref == s.yref:      # línea horizontal
                texto = a.text
                break
        if texto:
            nom = (texto.replace("<b>", "").replace("</b>", "")
                   .split("=")[0].replace("→", "").strip())
            lineas.append((nom, s.line.color, s.line.dash or "solid", s.line.width or 1.5))

    es_control = any(n in ("LCS", "LCI") for n in nombres | {l[0] for l in lineas})
    colores = []
    if es_control:
        for t in fig.data:
            modo = getattr(t, "mode", None)
            if t.showlegend is False and modo and "markers" in modo:
                col = t.marker.color
                if isinstance(col, (list, tuple, np.ndarray)):
                    for c in col:
                        if c in NOMBRES_COLOR and c not in colores:
                            colores.append(c)

    hechos = set()
    for nom, color, dash, ancho in lineas:
        if nom in nombres or nom in hechos:
            continue
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom, hoverinfo="skip",
                                 line=dict(color=color, dash=dash, width=ancho)))
        hechos.add(nom)
    for c in NOMBRES_COLOR:
        if c in colores:
            nom, simbolo = NOMBRES_COLOR[c]
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nom, hoverinfo="skip",
                                     marker=dict(color=c, symbol=simbolo, size=9)))


def estilo(fig, titulo=None, alto=360, derecha=10):
    """Aplica un aspecto uniforme: fondo blanco, marco, título centrado y leyenda abajo."""
    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b>", x=0.5, xanchor="center", y=0.97, yanchor="top",
                   font=dict(size=16, color=TEXTO)) if titulo else None,
        height=alto + 50, template="plotly_white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=TEXTO),
        margin=dict(l=35, r=derecha, t=70 if titulo else 40, b=78),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", xref="container", x=0.5, xanchor="center",
                    yref="container", y=0.01, yanchor="bottom",
                    font=dict(size=11, color=TEXTO), bgcolor="rgba(244,245,252,.9)",
                    bordercolor="#E5D3D9", borderwidth=1, title=dict(text="")),
        hoverlabel=dict(bgcolor="white", font=dict(color=TEXTO)))
    eje = dict(showgrid=True, gridcolor="#F0E5E9", zeroline=False, showline=True,
               linecolor="#A88E98", linewidth=1.3, mirror=True, ticks="outside",
               tickcolor="#A88E98", automargin=True,
               title_font=dict(size=12, color=TEXTO), tickfont=dict(color=TEXTO))
    fig.update_xaxes(**eje)
    fig.update_yaxes(**eje)
    _leyenda_automatica(fig)
    return fig