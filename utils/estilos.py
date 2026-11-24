"""Sistema visual de VVM Analytics: bienvenida clara + dashboard profesional."""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st


@lru_cache(maxsize=1)
def logo_uri():
    """Devuelve el logo como data-URI para incrustarlo en HTML (busca logo.* en la raíz del proyecto)."""
    raiz = Path(__file__).resolve().parent.parent
    tipos = {".png": "png", ".jpeg": "jpeg", ".jpg": "jpeg", ".webp": "webp"}
    for ext, mime in tipos.items():
        ruta = raiz / f"logo{ext}"
        if ruta.exists():
            return f"data:image/{mime};base64," + base64.b64encode(ruta.read_bytes()).decode()
    return ""
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#35101D;--muted:#806A72;--purple:#741B36;--purple2:#A33A58;--line:#EADDE1;--panel:#fff;--bg:#FAF5F6;}
html,body,.stApp{font-family:'DM Sans',sans-serif;color:var(--ink);background:var(--bg)}
.stApp{background:linear-gradient(135deg,#FCF8F9 0%,#F5EAEE 100%)}
#MainMenu,footer{visibility:hidden} header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;padding-bottom:2.5rem;max-width:1500px}
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3{font-family:'Manrope',sans-serif;color:var(--ink);letter-spacing:-.035em}
[data-testid="stAppViewContainer"] p,[data-testid="stAppViewContainer"] li,[data-testid="stAppViewContainer"] label{color:#533943}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{color:var(--muted)!important}
[data-testid="stWidgetLabel"] p{font-weight:600;color:#533943}
input,textarea{color:var(--ink)!important;background:#fff!important}
[data-baseweb="select"]>div{background:#fff;color:var(--ink)}
/* Navegación */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#4A1024 0%,#310A18 100%);border-right:0}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.42rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label{color:#F8EAF0}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#F8EAF0}
.brand-lockup{display:flex;align-items:center;gap:.75rem;padding:.5rem .35rem 1.4rem;border-bottom:1px solid rgba(255,255,255,.13);margin-bottom:1rem}
.brand-mark{display:grid;place-items:center;width:43px;height:43px;border-radius:14px;background:linear-gradient(135deg,#B85A75,#741B36);font-size:1.45rem}
.brand-name{font-family:'Manrope',sans-serif;font-size:1.2rem;font-weight:800;color:#fff;line-height:1.1}
.brand-caption{font-size:.58rem;letter-spacing:.09em;color:#DABCC6;margin-top:.3rem;font-weight:700}
.sidebar-section-label{font-size:.65rem;letter-spacing:.12em;color:#C99EAD;padding:.25rem .55rem .45rem;font-weight:800}
[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:.28rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.56rem .65rem;border-radius:11px;transition:background .15s}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,.09)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(90deg,#8A2947,#5B1830)}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-size:.86rem;font-weight:600;color:#FFF8FA!important}
[data-testid="stSidebar"] [data-testid="stRadio"] input{accent-color:#DDA9BA}
.sidebar-note{margin-top:1.3rem;padding:1rem .7rem;border-radius:15px;text-align:center;background:linear-gradient(135deg,rgba(163,58,88,.38),rgba(91,24,48,.35));color:#F9EAF0;font-size:.82rem;font-weight:600;line-height:1.5}
.sidebar-note span{font-size:1.35rem;color:#F0C8D5}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{padding:.7rem .4rem;color:#D9B9C4!important}
/* Hero de bienvenida */
.hero{display:grid;grid-template-columns:1.05fr .95fr;min-height:260px;overflow:hidden;border-radius:22px;border:1px solid #E8D5DC;background:linear-gradient(115deg,#fff 0%,#F5E7EC 62%,#E8C6D2 100%);box-shadow:0 9px 28px rgba(43,48,105,.08);margin-bottom:1rem}
.hero-copy{padding:2.25rem 2.4rem;display:flex;flex-direction:column;justify-content:center}
.hero-kicker{font-size:.72rem;letter-spacing:.13em;font-weight:800;color:#842440;margin-bottom:.7rem}
.hero-copy h1{font-family:'Manrope',sans-serif;font-size:clamp(2rem,3.1vw,3rem);line-height:1.07;letter-spacing:-.055em;margin:0 0 .85rem;color:#35101D}
.hero-copy p{font-size:1rem;line-height:1.55;color:#725660;margin:0;max-width:570px}
.hero-visual{min-height:260px;display:flex;align-items:center;justify-content:center;position:relative;background:radial-gradient(circle at 70% 35%,rgba(255,255,255,.9),transparent 30%),linear-gradient(135deg,#DDAEBE,#A94D69 58%,#54172D)}
.hero-visual:before{content:'CEP';position:absolute;font-family:'Manrope';font-size:8rem;font-weight:800;color:rgba(255,255,255,.13);letter-spacing:-.08em}
.hero-graphic{position:relative;width:78%;max-width:390px;padding:1.25rem;border:1px solid rgba(255,255,255,.55);border-radius:18px;background:rgba(255,255,255,.83);box-shadow:0 18px 40px rgba(26,38,84,.2);backdrop-filter:blur(8px)}
.hero-graphic-title{font-weight:800;color:#4A1D2C;font-size:.85rem;margin-bottom:.65rem}
.hero-chart{height:105px;display:flex;align-items:flex-end;gap:7px;border-bottom:1px solid #DCC9D0;padding:0 7px;position:relative}
.hero-chart:before,.hero-chart:after{content:'';position:absolute;left:0;right:0;border-top:1px dashed #B65A73}
.hero-chart:before{top:22%}.hero-chart:after{top:78%}
.hero-chart i{display:block;flex:1;border-radius:5px 5px 0 0;background:linear-gradient(180deg,#B85A75,#741B36);height:var(--h)}
.hero-chart-label{display:flex;justify-content:space-between;color:#8D737C;font-size:.65rem;margin-top:.5rem}
/* Cards y métricas */
.section-heading{font-family:'Manrope';font-size:1.12rem;font-weight:800;color:#3B1724;margin:.5rem 0 .8rem}
.kpi{border-radius:16px;padding:1rem 1.1rem;border:1px solid rgba(30,45,90,.05);min-height:105px;box-shadow:0 4px 14px rgba(39,48,94,.045)}
.kpi .titulo{font-size:.78rem;font-weight:700;margin-bottom:.4rem}.kpi .valor{font-size:1.8rem;line-height:1.1;font-weight:800;color:#3B1724}.kpi .sub{font-size:.68rem;color:#806A72;margin-top:.35rem}
.kpi.azul{background:#e8efff}.kpi.azul .titulo{color:#7A2944}.kpi.verde{background:#e2f5ee}.kpi.verde .titulo{color:#18815d}.kpi.morado{background:#eeeaff}.kpi.morado .titulo{color:#842440}.kpi.naranja{background:#fff0e5}.kpi.naranja .titulo{color:#bd682e}
.module-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem .9rem;min-height:145px;box-shadow:0 4px 14px rgba(39,48,94,.045);text-align:center}
.module-card .ico{font-size:1.8rem;margin-bottom:.35rem}.module-card .name{font-weight:800;color:#3B1724;font-size:.9rem}.module-card .desc{font-size:.74rem;color:#806A72;margin-top:.3rem;line-height:1.35}
/* Gráficos y controles */
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #E8D5DC;border-radius:16px;padding:8px;box-shadow:0 4px 14px rgba(39,48,94,.05)}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem;box-shadow:0 3px 10px rgba(39,48,94,.04)}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#806A72!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#4A1D2C!important}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:15px;background:#fff;border-color:#E8D5DC}
[data-testid="stExpander"]{background:#fff;border:1px solid #E8D5DC;border-radius:13px}
[data-testid="stDataFrame"]{border:1px solid #E8D5DC;border-radius:12px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid #E8D5DC}
.stTabs [data-baseweb="tab"]{padding:.55rem 1rem;border-radius:9px 9px 0 0;color:#725660}
.stTabs [aria-selected="true"]{color:#741B36!important;font-weight:800}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid #E3CDD5;background:#fff;padding:.52rem 1rem;transition:all .15s}
.stButton>button p,.stDownloadButton>button p{color:#741B36!important;font-weight:700}
.stButton>button:hover{border-color:#741B36;background:#F8EDF1}
.stButton button[kind="primary"],.stButton [data-testid="stBaseButton-primary"]{background:#741B36!important;border-color:#741B36!important}
.stButton button[kind="primary"] p,.stButton [data-testid="stBaseButton-primary"] p{color:#fff!important}
@media(max-width:800px){.hero{grid-template-columns:1fr}.hero-copy{padding:1.6rem}.hero-visual{min-height:180px}.hero-copy h1{font-size:2rem}}

/* VVM Analytics — identidad visual dark navy / electric blue */
:root{color-scheme:dark;--ink:#EAF2FF;--muted:#9BAEC8;--line:#243B59;--panel:#101D2D;--bg:#07111F}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#07111F!important;color:#EAF2FF!important}
.stApp{background:radial-gradient(ellipse at 82% 0%,rgba(22,91,166,.18),transparent 38%),#07111F!important}
.block-container{max-width:1580px;padding-top:1.8rem}
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3{color:#F2F7FF!important}
[data-testid="stAppViewContainer"] p,[data-testid="stAppViewContainer"] li,[data-testid="stAppViewContainer"] label,[data-testid="stWidgetLabel"] p{color:#C5D3E5!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:#91A8C5!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0C1B2D 0%,#081321 100%)!important;border-right:1px solid #1D3858}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#DCEAFF!important}
.brand-lockup{border-bottom:1px solid #24415F}.brand-mark{background:linear-gradient(135deg,#168BFF,#0752A4);box-shadow:0 8px 24px #0875d544}.brand-caption{color:#80BFFF}
.sidebar-section-label{color:#70B6FF}.sidebar-note{background:linear-gradient(135deg,#123D68,#10243B);border:1px solid #24517B;color:#D8EBFF}.sidebar-note span{color:#65B6FF}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(90deg,#126CC0,#10477E)}
[data-testid="stSidebar"] [data-testid="stRadio"] input{accent-color:#3DA0FF}
/* superficies */
div[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stDataFrame"],[data-testid="stPlotlyChart"]{background:#0E1C2C!important;border:1px solid #233D5B!important;border-radius:15px!important;color:#EAF2FF!important;box-shadow:0 8px 24px rgba(0,0,0,.18)}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:#9EB4CE!important}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#F1F7FF!important}
input,textarea,[data-baseweb="select"]>div{color:#EAF2FF!important;background:#102238!important;border-color:#2B4A6B!important}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"]{background:#102238!important}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid #294663}
.stTabs [data-baseweb="tab"]{color:#AFC3DC!important}.stTabs [aria-selected="true"]{color:#58AEFF!important;border-bottom-color:#238BEE!important}
.stButton>button,.stDownloadButton>button{background:#122943!important;border:1px solid #2D5A85!important;border-radius:10px!important}
.stButton>button p,.stDownloadButton>button p{color:#D9ECFF!important}.stButton>button:hover,.stDownloadButton>button:hover{background:#17416B!important;border-color:#48A5F8!important}
.stButton button[kind="primary"],.stButton [data-testid="stBaseButton-primary"]{background:#0876D8!important;border-color:#168BFF!important}
.stButton button[kind="primary"] p,.stButton [data-testid="stBaseButton-primary"] p{color:white!important}
[data-testid="stAlert"]{border-radius:12px}
.hero{background:linear-gradient(115deg,#0D2035,#102D4A 65%,#124D7C)!important;border:1px solid #24517A!important;box-shadow:0 14px 36px #0005!important}
.hero-copy h1{color:#F4F9FF!important}.hero-copy p{color:#B9CDE3!important}.hero-kicker{color:#69B8FF!important}
.hero-visual{background:radial-gradient(circle at 70% 35%,rgba(67,164,255,.35),transparent 30%),linear-gradient(135deg,#164A78,#0B2440 58%,#071728)!important}
.hero-graphic{background:rgba(8,24,41,.88)!important;border-color:#4778A3!important}.hero-graphic-title{color:#D8EBFF!important}.hero-chart{border-color:#315474}.hero-chart i{background:linear-gradient(180deg,#46B4FF,#0868C2)}
.section-heading{color:#EAF4FF!important}.kpi{background:#102238!important;border:1px solid #24415F!important;box-shadow:none!important}.kpi .valor{color:#F3F8FF!important}.kpi .sub{color:#9CB1CA!important}.kpi.azul .titulo,.kpi.verde .titulo,.kpi.morado .titulo,.kpi.naranja .titulo{color:#72BFFF!important}
.module-card{background:#102238!important;border:1px solid #294967!important}.module-card .name{color:#EAF4FF!important}.module-card .desc{color:#9EB4CE!important}
@media(max-width:800px){.hero{grid-template-columns:1fr}.hero-copy{padding:1.6rem}.hero-visual{min-height:180px}}

</style>
"""

CSS_LOGO = """<style>
.hero-logo{position:relative;width:min(380px,80%);height:auto;border-radius:20px;box-shadow:0 18px 44px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.14)}
.brand-lockup-logo{display:block;padding:.2rem 0 1.1rem}
.brand-logo-full{display:block;width:100%;height:auto;border-radius:14px}
</style>"""


CSS_LOGO = """<style>
.hero-logo{position:relative;width:min(380px,80%);height:auto;border-radius:20px;box-shadow:0 18px 44px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.14)}
.brand-lockup-logo{display:block;padding:.2rem 0 1.1rem}
.brand-logo-full{display:block;width:100%;height:auto;border-radius:14px}
</style>"""


CSS_LOGO = """<style>
.hero-logo{position:relative;width:min(380px,80%);height:auto;border-radius:20px;box-shadow:0 18px 44px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.14)}
.brand-lockup-logo{display:block;padding:.2rem 0 1.1rem}
.brand-logo-full{display:block;width:100%;height:auto;border-radius:14px}
</style>"""


def aplicar_estilos():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(CSS_LOGO, unsafe_allow_html=True)