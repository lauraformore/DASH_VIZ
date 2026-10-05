import math
import os
import textwrap
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table, Input, Output, State, ctx, ALL, no_update

# ---------- Configuración ----------
CSV = "data/data.csv"      # pon aquí tu dataset
TARGET = "Bankrupt?"       # columna 0/1 de bancarrota
TITLE = "Predicción de bancarrota"

# ---------- Datos ----------
def load():
    if os.path.exists(CSV):
        d = pd.read_csv(CSV)
        d.columns = d.columns.str.strip()
        return d
    # Datos de demostración si no hay CSV (para que la app corra desde ya)
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.08).astype(int)
    d = pd.DataFrame({
        "ROA": rng.normal(.05, .05, n) - .08 * y,
        "Debt ratio": rng.normal(.4, .1, n) + .15 * y,
        "Current ratio": rng.lognormal(.3, .4, n) - .3 * y,
        "Net profit margin": rng.normal(.06, .05, n) - .09 * y,
        "Asset turnover": rng.gamma(2, .4, n),
        "Equity / liability": rng.normal(.5, .15, n) - .15 * y,
    })
    d.insert(0, TARGET, y)
    return d

df = load()
df[TARGET] = df[TARGET].astype(int)
CAT = [c for c in ["Liability-Assets Flag", "Net Income Flag"] if c in df.columns]   # cualitativas (flags 0/1)
FEATS = [c for c in df.select_dtypes("number").columns if c not in CAT + [TARGET]]   # cuantitativas (ratios)
df["Estado"] = df[TARGET].map({0: "No bancarrota", 1: "Bancarrota"})
ACCENT, ROSE, AMBER = "#7c6bff", "#ff5d7a", "#fdab3d"      # identidad gráfica (violeta = sana, rosa = quiebra)
COLORS = {"No bancarrota": ACCENT, "Bancarrota": ROSE}
pio.templates.default = "none"
px.defaults.color_discrete_sequence = [ACCENT, ROSE, AMBER, "#2dd4bf", "#60a5fa", "#c084fc"]
DENSITY = [[0, "rgba(124,107,255,0.04)"], [.5, ACCENT], [1, "#f0abfc"]]
CORR = df[FEATS].corrwith(df[TARGET]).dropna()
BY_ABS = CORR.abs().sort_values(ascending=False)

# ---------- Sistema de ayuda ----------
# Cada entrada: (título, qué muestra, cómo interpretarlo, [ten en cuenta]). Tooltips (?) para filtros e indicadores; ventana modal (i) para gráficos y secciones.
HELP = {
    "p_overview": ("Overview", "Resumen del conjunto de datos: cuántas empresas hay, cuántas variables y qué proporción quebró.",
                   "Empieza por las tarjetas superiores para dimensionar los datos y sigue con el gráfico de desbalance: condiciona cómo debe evaluarse cualquier modelo."),
    "p_uni": ("Univariado", "Análisis de una sola variable a la vez: su forma, centro, dispersión y valores extremos.",
              "Elige el tipo de variable y la variable en el filtro. Al final, el panorama compara todas las variables cuantitativas para ver cuáles tienen más outliers o asimetría."),
    "p_bi": ("Bivariado", "Comparación de una variable entre empresas quebradas y no quebradas.",
             "Si las distribuciones de ambos grupos se separan, la variable ayuda a distinguirlos; si se superponen, aporta poco por sí sola.",
             "Son asociaciones descriptivas: no prueban causalidad."),
    "p_multi": ("Multivariado", "Exploración conjunta mediante matriz de correlación, dispersión, densidad, pares correlacionados, PCA y matriz de dispersión.",
                "Ajusta el slider y los ejes para explorar. Busca grupos de variables muy correlacionadas y zonas donde se concentran las empresas quebradas."),
    "p_insights": ("Insights", "Síntesis de los hallazgos principales del análisis exploratorio.",
                   "Compara los perfiles del radar, consulta la tabla de síntesis por aspectos y lee las ocho conclusiones numeradas. Las cifras se calculan sobre los datos cargados.",
                   "Los hallazgos son asociaciones exploratorias; su utilidad predictiva debe validarse fuera de la muestra."),
    "eda_summary": ("Síntesis por aspectos del EDA", "Tabla de estructura, target, flags, extremos IQR, asociaciones cuantitativas y categóricas y multicolinealidad.",
                    "Cada fila relaciona un hallazgo calculado sobre la muestra completa con una recomendación preliminar para modelado.",
                    "Las pruebas usan α = 0.05. Los p-valores de Mann-Whitney no están ajustados por multiplicidad; no equivalen a desempeño predictivo."),
    "eda_conclusions": ("Conclusiones del EDA", "Ocho conclusiones que reúnen estadísticos observados, limitaciones y decisiones metodológicas.",
                        "Contrasta el resumen con Univariado, Bivariado y Multivariado. r biserial positivo indica valores mayores en solventes; negativo, en quebradas.",
                        "Las medias ROA se muestran en la escala del CSV. Ajusta preprocesamiento y selección exclusivamente en entrenamiento dentro de cada fold."),
    "p_explorer": ("Data Explorer", "Tabla completa de datos para investigar por tu cuenta.",
                   "Elige columnas, ordena con las flechas del encabezado, filtra escribiendo en la fila bajo el encabezado (por ejemplo >0.5) y descarga lo que ves."),
    "balance": ("Balance de clases", "Proporción de empresas quebradas y no quebradas.",
                "Cuanto más pequeña la porción rosa, más desbalanceado el problema. Con clases tan desiguales la exactitud (accuracy) engaña.",
                "Para evaluar modelos conviene usar recall, F1 o AUC-PR."),
    "sample": ("Muestra de datos", "Primeras filas del dataset con la variable objetivo y algunas variables.",
               "Sirve para comprobar el formato: 0 = no bancarrota, 1 = bancarrota."),
    "uv_hist": ("Distribución y boxplot", "Histograma (frecuencia de valores) con un boxplot encima (mediana, cuartiles y outliers).",
                "La altura de las barras indica cuántas empresas hay en cada rango. La caja contiene el 50 % central, la línea interna es la mediana y los puntos son outliers.",
                "Se recorta el 1 % extremo de cada lado para que la forma sea visible; las tarjetas usan todos los datos."),
    "uv_bar": ("Frecuencia por categoría", "Número de empresas en cada categoría.", "Compara la altura de las barras; el texto sobre cada una muestra el porcentaje."),
    "pan_out": ("Outliers por variable", "Porcentaje de empresas fuera del rango habitual (regla IQR) en las 15 variables con más outliers.",
                "Barras largas indican muchos valores extremos, que conviene tratar (winsorizing, transformaciones) antes de modelar.",
                "Regla IQR: valores por debajo de Q1 − 1,5·IQR o por encima de Q3 + 1,5·IQR."),
    "pan_skew": ("Asimetría por variable", "Asimetría (skewness) de las 15 variables más asimétricas.",
                 "0 = simétrica; positivo = cola larga a la derecha; negativo = cola a la izquierda. Con |valor| > 1 la asimetría es marcada."),
    "bv_box": ("Boxplot por estado", "Distribución de la variable en empresas quebradas y no quebradas.",
               "Compara medianas (línea central) y alturas de las cajas. Si las cajas no se solapan, la variable separa bien los grupos."),
    "bv_stats": ("Estadísticas descriptivas por estado", "Resumen numérico de la variable en empresas quebradas y no quebradas: tamaño del grupo (N), media, desviación estándar (DS), mediana, mínimo, máximo, cuartiles (Q1, Q3) y rango intercuartílico (IQR = Q3 − Q1).",
                 "Compara las filas: medianas distintas indican que el valor típico cambia entre grupos; un IQR o una DS mayores indican más dispersión. Si la media se aleja mucho de la mediana, hay valores extremos o asimetría y conviene confiar más en la mediana.",
                 "Se calculan con todas las empresas (sin recortar extremos). Con pocas empresas quebradas, las cifras de ese grupo son menos estables."),
    "bv_infer": ("Prueba U de Mann-Whitney", "Contraste de hipótesis que compara la distribución de la variable entre ambos grupos usando rangos (posiciones), sin asumir normalidad y siendo robusto a outliers. Reporta el estadístico U, el p-valor y el tamaño de efecto r biserial.",
                 "H₀ plantea que no hay diferencia. Si p < 0,05 se rechaza H₀: la diferencia es estadísticamente significativa. El r biserial (−1 a +1) dice qué tan grande es: ±0,1 pequeño, ±0,3 moderado, ±0,5 grande; positivo = valores mayores en no quebradas.",
                 "Significativo no es sinónimo de importante: con miles de empresas incluso diferencias mínimas dan p pequeño. Mira siempre el tamaño de efecto."),
    "bv_cat": ("Proporción de bancarrota según la variable", "Para cada nivel del flag (0 y 1), el porcentaje de empresas quebradas y no quebradas dentro de ese nivel. Las dos barras de un mismo nivel suman 100 %; sobre cada barra se indica el porcentaje y el número de empresas (n).",
               "Compara la barra rosa (Bancarrota) entre los dos niveles: si es claramente mayor en uno, ese grupo tiene más riesgo de quiebra. Si los porcentajes son parecidos, la variable no distingue entre empresas sanas y quebradas.",
               "Revisa el n de cada nivel: con pocas empresas el porcentaje es inestable, aunque se vea muy diferente."),
    "bv_fisher": ("Prueba exacta de Fisher", "Contraste de hipótesis sobre la tabla 2×2 (nivel del flag × quiebra) que evalúa si la bancarrota es independiente de la variable. Reporta el odds ratio (OR), su intervalo de confianza y el p-valor exacto.",
                  "H₀ plantea que no hay asociación (OR = 1). Si p < 0,05 se rechaza H₀. El OR compara las odds de quiebra: OR > 1, más quiebra cuando el flag vale 1; OR < 1, menos; OR = 1, igual. Es válida aun con grupos muy pequeños, a diferencia del chi-cuadrado.",
                  "Si la variable es constante (un solo valor) no hay dos grupos que comparar y la prueba no se puede aplicar. Significativo no es sinónimo de importante: mira también el OR y el tamaño de los grupos."),
    "pan_bi": ("Variables que mejor separan los grupos", "Tamaño de efecto (r biserial) de las 15 variables con mayor diferencia entre grupos.",
               "Barras largas indican más separación. El color señala en qué grupo la variable tiene valores mayores."),
    "heat": ("Matriz de correlación", "Correlación de Pearson entre las variables más asociadas a la quiebra y la propia quiebra.",
             "Rojo = correlación positiva, azul = negativa, blanco ≈ 0. Bloques muy intensos entre variables indican redundancia (multicolinealidad)."),
    "scatter": ("Dispersión", "Cada punto es una empresa ubicada según las dos variables elegidas (muestra de hasta 2.000).",
                "Una tendencia diagonal indica relación entre las variables. Mira si los puntos rosas (quebradas) se agrupan en alguna zona."),
    "density": ("Mapa de densidad", "Cuántas empresas caen en cada zona del plano formado por las dos variables.",
                "Colores más intensos = más empresas. Muestra dónde se concentra la mayoría aunque los puntos se superpongan.",
                "Se recorta el 1 % extremo de cada variable."),
    "radar": ("Perfil de quebradas vs. sanas", "Promedio estandarizado (z) de las 8 variables más asociadas a la quiebra, en cada grupo.",
              "0 es el promedio general. Cuanto más separadas estén las dos líneas en un eje, más distinto es ese rasgo entre los grupos."),
    "pairs": ("Pares de variables más correlacionados", "Los 10 pares de variables con mayor correlación absoluta entre sí.",
              "Valores cercanos a 1 indican que miden casi lo mismo; en modelos lineales conviene conservar solo una de ellas."),
    "pca": ("Análisis de componentes principales (PCA)", "Resume todas las variables en dos ejes (PC1 y PC2) que capturan la mayor varianza.",
            "Si rosa y violeta se mezclan, las clases no se separan bien en 2D. Si el rosa se desplaza, hay señal.",
            "Es una proyección: puede ocultar separación que existe en más dimensiones."),
    "matrix": ("Matriz de dispersión", "Cruces de a pares entre las 4 variables más asociadas a la quiebra (muestra de hasta 1.500 empresas).",
               "Busca paneles donde los puntos rosas ocupen zonas distintas a las de los violetas: indican señal conjunta."),
    "top": ("Top 10 correlaciones con la quiebra", "Las 10 variables con mayor correlación absoluta con la quiebra.",
            "Rosa = sube cuando hay quiebra; violeta = baja. Barras largas indican una asociación más fuerte.", "Correlación no implica causalidad."),
}

STAT_TIPS = {"Media": "Promedio de todas las empresas.", "Desv. estándar": "Dispersión típica alrededor de la media.",
             "Mínimo": "Valor más bajo observado.", "Mediana": "Valor central: la mitad de las empresas queda por debajo.",
             "Máximo": "Valor más alto observado."}

def info(key):
    return html.Button("i", id={"type": "info", "key": key}, className="info", title="Cómo interpretar este gráfico",
                       **{"aria-label": f"Ayuda: {HELP[key][0]}"})

def hint(text):
    return html.Span("?", className="tip", tabIndex=0, role="note", **{"data-tip": text, "aria-label": text})

def field(label, control, tip=None):
    return html.Div([html.Div([html.Span(label)] + ([hint(tip)] if tip else []), className="field-l"), control], className="field")

def render_help(key):
    t, what, how, *warn = HELP[key]
    body = [html.H3(t), html.H5("Qué muestra"), html.P(what), html.H5("Cómo interpretarlo"), html.P(how)]
    if warn:
        body.append(html.Div([html.H5("Ten en cuenta"), html.P(warn[0])], className="warn"))
    return body


# ---------- Helpers ----------
GRID = "rgba(128,128,128,.18)"

def wrap(s, n=28):
    """Parte un texto largo en varias líneas (<br>) para que las etiquetas no se corten."""
    return "<br>".join(textwrap.wrap(str(s), n)) or str(s)

def style(fig, h=380, custom_margin=None):
    horiz = any(t.type == "bar" and t.orientation == "h" for t in fig.data)
    polar = any(t.type == "scatterpolar" for t in fig.data)
    legend = len(fig.data) > 1 or any(t.type == "pie" for t in fig.data)
    margin = dict(l=180 if horiz else 20, r=20, t=56 if legend else 44, b=50)    # automargin amplía si hace falta
    if polar:
        margin = dict(l=90, r=90, t=70, b=40)       # espacio para las etiquetas angulares del radar
    if custom_margin:
        margin = {**margin, **custom_margin}
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="",
                      font=dict(family="Inter, system-ui, sans-serif", size=12), margin=margin,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hoverlabel=dict(font_family="Inter, system-ui, sans-serif"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    for t in fig.data:     # scatter_matrix: Plotly crea xaxis2..n / yaxis2..n sin estilo; se les aplica el mismo
        if t.type == "splom":
            n = len(t.dimensions)
            ax = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True,
                      title=dict(font=dict(size=11), standoff=8))
            fig.update_layout({f"{a}axis{i if i > 1 else ''}": ax for a in "xy" for i in range(1, n + 1)})
    for ax in [*fig.select_xaxes(), *fig.select_yaxes()]:      # títulos de eje largos: en varias líneas
        t = ax.title.text
        if t and len(t) > 36 and "<br>" not in t:
            ax.update(title_text=wrap(t, 36))
    if polar:
        fig.update_polars(bgcolor="rgba(0,0,0,0)", angularaxis_gridcolor=GRID, radialaxis_gridcolor=GRID)
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)", selector=dict(type="pie"))
    try:
        fig.update_layout(barcornerradius=6)      # barras redondeadas (plotly >= 6)
    except ValueError:
        pass
    return fig

def card_h(title, key):
    return html.Div([html.H4(title), info(key)], className="card-h")

def graph(key, fig, h=380, text=None, margin=None):   # margin: sobrescribe el margen por defecto de style()
    kids = [card_h(HELP[key][0], key), dcc.Graph(figure=style(fig, h, margin), style={"height": f"{h}px"}, config={"displaylogo": False, "responsive": True})]
    if text:
        kids.append(html.P([html.B("Interpretación: "), text], className="interp"))
    return html.Div(kids, className="card")

def head(t, q):
    i, k = next((i, k) for i, (k, n, _, _) in enumerate(NAV, 1) if n == t)
    return html.Div([html.Div([html.P(f"Sección 0{i}", className="eyebrow"), html.H2(t), html.P(q, className="q")]),
                     html.Button([html.I("?"), "Cómo leer esta sección"], id={"type": "info", "key": "p_" + k}, className="guide")],
                    className="head")

def kpi(t, v, tip=None):
    lab = [html.Span(t)] + ([hint(tip)] if tip else [])
    return html.Div([html.Div(str(v), className="kv"), html.Div(lab, className="kt")], className="card kpi")

TBL = dict(
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "var(--head)", "color": "var(--text)", "fontWeight": "600", "border": "none"},
    style_cell={"backgroundColor": "var(--card)", "color": "var(--text)", "border": "none",
                "borderBottom": "1px solid var(--line)", "padding": "9px 10px", "fontSize": "13px", "textAlign": "left"},
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "var(--bg2)"}])

def table(d, **kw):
    return dash_table.DataTable(data=d.round(4).to_dict("records"), columns=[{"name": c, "id": c} for c in d.columns], **TBL, **kw)

def clip(v):
    lo, hi = df[v].quantile([.01, .99])
    return df[df[v].between(lo, hi)]

def picker(id_, value=None):
    return dcc.Dropdown(FEATS, value or FEATS[0], id=id_, clearable=False)

def selector(pref):
    opts = [{"label": "Cuantitativas (ratios)", "value": "num"}]
    if CAT:
        opts.append({"label": "Cualitativas (flags 0/1)", "value": "cat"})
    return html.Div([
        field("Tipo de variable", dcc.RadioItems(opts, "num", id=f"{pref}-tipo", inline=True, className="radio"),
              "Cuantitativas: ratios numéricos continuos. Cualitativas: indicadores 0/1 (flags)."),
        field("Variable", dcc.Dropdown(FEATS, FEATS[0], id=pref, clearable=False),
              "Escribe para buscar. Todos los gráficos de esta sección se actualizan al elegir otra variable.")],
        className="card sel")

def note(title, text):
    return html.Div([html.H4(title), html.P(text)], className="card note")

def mag(r):
    """Magnitud cualitativa del tamaño de efecto (|r biserial|)."""
    a = abs(r)
    return "despreciable" if a < .1 else "pequeño" if a < .3 else "moderado" if a < .5 else "grande"

@lru_cache(None)
def mann_whitney(v):
    """U de Mann-Whitney (bilateral, aprox. normal con corrección por empates y continuidad) + r biserial.
    U se calcula para el grupo «No bancarrota»; r > 0 = valores mayores en no quebradas."""
    d = df[[v, TARGET]].dropna()
    g0 = (d[TARGET] == 0).values
    n0, n1 = int(g0.sum()), int((~g0).sum())
    n = n0 + n1
    u = d[v].rank().values[g0].sum() - n0 * (n0 + 1) / 2
    ties = np.unique(d[v].values, return_counts=True)[1]
    sd = math.sqrt(n0 * n1 / 12 * ((n + 1) - (ties ** 3 - ties).sum() / (n * (n - 1))))
    z = (abs(u - n0 * n1 / 2) - .5) / sd if sd else 0.0
    return dict(U=u, z=z, p=math.erfc(z / math.sqrt(2)), r=2 * u / (n0 * n1) - 1, n0=n0, n1=n1)

def rank_biserial(v):
    """Tamaño de efecto (+ = valores mayores en no quebradas; - = mayores en quebradas)."""
    return mann_whitney(v)["r"]

def out_pct(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).mean()

def interp_balance():
    b = df[TARGET].mean()
    return (f"El {b:.1%} de las empresas quebró: hay unas {(1 - b) / b:.0f} sanas por cada quebrada. Un modelo que siempre "
            f"prediga «no quiebra» acertaría {1 - b:.0%} de las veces sin detectar ninguna quiebra, por eso conviene evaluar "
            "con recall, F1 o AUC-PR y tratar el desbalance antes de modelar.")

def interp_dist(v):
    s = df[v]; sk = s.skew(); o = out_pct(s)
    forma = "aproximadamente simétrica" if abs(sk) < .5 else ("sesgada a la derecha (cola de valores altos)" if sk > 0 else "sesgada a la izquierda (cola de valores bajos)")
    fuerza = "" if abs(sk) < .5 else (" y de forma marcada" if abs(sk) > 1 else " y de forma moderada")
    extra = ", bastantes: hay que decidir cómo tratarlos antes de modelar." if o > .1 else "."
    return (f"La distribución es {forma}{fuerza} (asimetría {sk:+.2f}); media {s.mean():.4g} y mediana {s.median():.4g}. "
            f"{o:.1%} de las empresas son outliers según la regla IQR{extra} La gráfica recorta el 1 % extremo de cada lado para poder verse.")

def interp_box(v):
    a, b = df[df[TARGET] == 0][v], df[df[TARGET] == 1][v]
    m0, m1 = a.median(), b.median()
    ra, rb = a.quantile([.25, .75]).values, b.quantile([.25, .75]).values
    solapan = ra[0] <= rb[1] and rb[0] <= ra[1]
    pos = "por debajo" if m1 < m0 else "por encima"
    fin = ("se solapan: hay mucha zona compartida entre ambos grupos." if solapan
           else "no se solapan: la variable separa bien a los dos grupos.")
    return (f"La caja de las quebradas está {pos}: mediana {m1:.4g} frente a {m0:.4g} de las no quebradas. "
            f"Sus rangos intercuartílicos ({rb[0]:.4g}–{rb[1]:.4g} y {ra[0]:.4g}–{ra[1]:.4g}) {fin}")

def interp_heat(c, n):
    t = c[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    top = ", ".join(f"{k} ({c.loc[k, TARGET]:+.2f})" for k in t.index[:3])
    f = c.drop(index=TARGET, columns=TARGET)
    k = int(((f.abs() >= .9).sum().sum() - len(f)) / 2)
    fuerza = ("ninguna supera 0,3 en valor absoluto: ninguna variable sola explica la quiebra con fuerza"
              if t.iloc[0] < .3 else f"la mayor llega a {t.iloc[0]:.2f}")
    mc = (f"Hay {k} pares de variables con |r| ≥ 0,9: son versiones casi idénticas del mismo ratio (multicolinealidad), a cuidar en modelos lineales."
          if k else "No hay pares de variables con |r| ≥ 0,9, así que no se ve multicolinealidad fuerte.")
    return f"Muestra las {n} variables más asociadas a la quiebra (Pearson). Con Bankrupt? destacan {top}; {fuerza}. {mc}"

def interp_scatter(x, y):
    if x == y:
        return "Elegiste la misma variable en ambos ejes; cambia una para ver la relación."
    r = df[x].corr(df[y]); a = abs(r)
    fuerza = "muy fuerte" if a >= .8 else "fuerte" if a >= .6 else "moderada" if a >= .3 else "débil"
    m = df.groupby(TARGET)[[x, y]].median()
    dx = "menores" if m.loc[1, x] < m.loc[0, x] else "mayores"
    dy = "menores" if m.loc[1, y] < m.loc[0, y] else "mayores"
    return (f"Relación {'positiva' if r > 0 else 'negativa'} {fuerza} entre ambas variables (r = {r:+.2f}). Las quebradas (rojo) tienen "
            f"medianas {dx} en «{x}» y {dy} en «{y}». Si rojo y azul se mezclan, ninguna de las dos variables separa los grupos por sí sola.")

def interp_top():
    top = CORR.reindex(BY_ABS.index[:10]); v = BY_ABS.index[0]; c = CORR[v]; pos = int((top > 0).sum())
    sentido = "a mayor valor, más probabilidad de quiebra" if c > 0 else "a menor valor, más probabilidad de quiebra"
    cierre = ("Las correlaciones son moderadas o débiles: ninguna variable sola basta para predecir, por eso hace falta un modelo multivariado."
              if BY_ABS.iloc[0] < .4 else "Hay al menos una asociación fuerte, pero conviene confirmarla con el modelo.")
    return (f"«{v}» es la variable más asociada a la quiebra (r = {c:+.2f}): {sentido}. De las {len(top)} variables del gráfico, "
            f"{pos} suben con la quiebra y el resto baja. {cierre}")

def interp_cat(v, g):
    """g: una fila por nivel de v con columnas v (str), n (empresas), k (quebradas), tasa."""
    if len(g) < 2:
        return f"«{v}» toma un único valor en todas las empresas, así que no distingue entre quebradas y no quebradas y no aporta información."
    hi, lo = g.loc[g["tasa"].idxmax()], g.loc[g["tasa"].idxmin()]
    t = ("Entre las empresas " + "; ".join(f"con {v} = {r[v]} quebraron {r['k']:,} de {r['n']:,} ({r['tasa']:.2%})" for _, r in g.iterrows()) + ". ")
    if hi["tasa"] == lo["tasa"]:
        t += "Las tasas de quiebra son iguales en ambos niveles: la variable no distingue entre grupos."
    elif lo["tasa"] > 0:
        t += f"El grupo {v} = {hi[v]} quiebra unas {hi['tasa'] / lo['tasa']:.1f} veces más que el grupo {v} = {lo[v]}."
    else:
        t += f"Ninguna empresa con {v} = {lo[v]} quebró, mientras que con {v} = {hi[v]} quebró el {hi['tasa']:.2%}."
    if min(hi["n"], lo["n"]) < 30:
        t += " Ojo: uno de los grupos tiene menos de 30 empresas, así que la diferencia puede ser inestable."
    return t + " Es una asociación descriptiva, no prueba causalidad."

# ---------- Gráficas adicionales (cada una con su interpretación) ----------
@lru_cache(None)
def effects():   # r biserial de todas las variables cuantitativas
    return pd.Series({v: rank_biserial(v) for v in FEATS}).dropna()

@lru_cache(None)
def feat_stats():
    return pd.DataFrame({"outliers": {v: out_pct(df[v]) for v in FEATS}, "asimetria": {v: df[v].skew() for v in FEATS}})

def hbar(d, x, y, **kw):
    d = d.copy(); d[y] = d[y].map(wrap)
    return px.bar(d, x=x, y=y, orientation="h", **kw)

def fmt_p(p):
    return "< 0.001" if p < .001 else f"{p:.3f}"

def stats_table(v):
    """Estadísticas descriptivas de v por estado (todas las empresas) + su lectura."""
    rows, st = [], {}
    for k in ("Bancarrota", "No bancarrota"):
        x = df.loc[df["Estado"] == k, v].dropna()
        q1, q3 = x.quantile([.25, .75])
        st[k] = dict(n=len(x), mean=x.mean(), sd=x.std(), med=x.median(), q1=q1, q3=q3, iqr=q3 - q1)
        rows.append({"Estado Financiero": k, "N": f"{len(x):,}", "Media": f"{x.mean():.4g}", "DS": f"{x.std():.4g}",
                     "Mediana": f"{x.median():.4g}", "Mínimo": f"{x.min():.4g}", "Máximo": f"{x.max():.4g}",
                     "Q1": f"{q1:.4g}", "Q3": f"{q3:.4g}", "IQR": f"{st[k]['iqr']:.4g}"})
    cols = list(rows[0])
    tbl = dash_table.DataTable(
        data=rows, columns=[{"name": c, "id": c} for c in cols], **{**TBL,
            "style_data_conditional": TBL["style_data_conditional"] + [
                {"if": {"filter_query": '{Estado Financiero} = "Bancarrota"', "column_id": "Estado Financiero"}, "color": ROSE},
                {"if": {"filter_query": '{Estado Financiero} = "No bancarrota"', "column_id": "Estado Financiero"}, "color": ACCENT}],
            "style_cell_conditional": [{"if": {"column_id": "Estado Financiero"}, "fontWeight": "600", "minWidth": "140px"}] +
                                      [{"if": {"column_id": c}, "textAlign": "right"} for c in cols[1:]]})
    b, nb = st["Bancarrota"], st["No bancarrota"]
    dif = b["med"] - nb["med"]
    t = [f"La mediana de «{v}» es {b['med']:.4g} en las quebradas y {nb['med']:.4g} en las no quebradas "
         f"({'menor' if dif < 0 else 'mayor' if dif > 0 else 'igual'} en las quebradas" + (f", con una diferencia de {abs(dif):.4g}" if dif else "") + "). "]
    t.append(f"Las medias son {b['mean']:.4g} y {nb['mean']:.4g}, respectivamente.")
    lejos = [k for k, x in st.items() if x["iqr"] and abs(x["mean"] - x["med"]) > .5 * x["iqr"]]
    if lejos:
        t.append(f" En {' y '.join(lejos).replace('Bancarrota', 'las quebradas').replace('No bancarrota', 'las no quebradas')} la media se aleja de la mediana "
                 "más de medio IQR: hay valores extremos o asimetría, así que la mediana representa mejor al grupo.")
    if b["iqr"] and nb["iqr"]:
        rr = b["iqr"] / nb["iqr"]
        t.append(f" La dispersión central (IQR) es {b['iqr']:.4g} en las quebradas frente a {nb['iqr']:.4g} en las no quebradas "
                 f"({'mayor' if rr > 1.1 else 'menor' if rr < .9 else 'similar'}, razón {rr:.2f}).")
    t.append(f" Ojo: solo hay {b['n']:,} empresas quebradas frente a {nb['n']:,} sanas, así que las cifras del primer grupo son menos estables.")
    return html.Div([card_h(HELP["bv_stats"][0], "bv_stats"), tbl,
                     html.P([html.B("Interpretación: "), "".join(t)], className="interp")], className="card")

def fisher_exact(a, b, c, d):
    """Prueba exacta de Fisher bilateral para [[a, b], [c, d]] (distribución hipergeométrica) + odds ratio muestral e IC 95 % (Woolf)."""
    n1, n2, m1 = a + b, c + d, a + c
    n = n1 + n2
    lc = lambda N, K: math.lgamma(N + 1) - math.lgamma(K + 1) - math.lgamma(N - K + 1)
    lp = lambda x: lc(n1, x) + lc(n2, m1 - x) - lc(n, m1)
    ref = lp(a) + 1e-7                      # mismo criterio de tolerancia que scipy
    p = min(1.0, sum(math.exp(lp(x)) for x in range(max(0, m1 - n2), min(m1, n1) + 1) if lp(x) <= ref))
    odds = (a * d) / (b * c) if b * c else (math.inf if a * d else math.nan)
    ci = None
    if min(a, b, c, d) > 0:
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci = (math.exp(math.log(odds) - 1.96 * se), math.exp(math.log(odds) + 1.96 * se))
    return dict(OR=odds, p=p, ci=ci)

def fmt_or(x):
    return "—" if math.isnan(x) else "∞" if math.isinf(x) else f"{x:.2f}"

def fisher_block(v):
    """Inferencia para una variable cualitativa: Fisher exacta (o aviso si es constante)."""
    levels = sorted(df[v].dropna().unique())
    head_ = lambda chip: html.Div([html.H4(HELP["bv_fisher"][0]), chip, info("bv_fisher")], className="card-h")
    if len(levels) != 2:
        msg = (f"«{v}» toma un único valor ({levels[0]:g}) en las {len(df):,} empresas: no tiene variabilidad, por lo que no existen dos grupos que comparar, "
               "no se puede construir la tabla 2×2 ni calcular un odds ratio. La variable no es estadísticamente contrastable y no aporta información para "
               "distinguir empresas quebradas de no quebradas; es candidata a eliminarse antes de modelar." if len(levels) == 1 else
               f"«{v}» tiene {len(levels)} niveles; la prueba de Fisher 2×2 solo aplica a variables binarias.")
        return html.Div([head_(html.Span("No contrastable", className="chip chip--muted")),
                         html.Div([html.B("Variable no contrastable. "), msg], className="alert", role="alert")], className="card")
    lo, hi = levels
    cnt = lambda lv, t: int(((df[v] == lv) & (df[TARGET] == t)).sum())
    a, b, c, d = cnt(hi, 1), cnt(hi, 0), cnt(lo, 1), cnt(lo, 0)      # filas: nivel alto / bajo · columnas: quebró / no quebró
    f, alpha = fisher_exact(a, b, c, d), .05
    p, odds, ci = f["p"], f["OR"], f["ci"]
    t_hi, t_lo = a / (a + b), c / (c + d)
    rech = p < alpha
    dirn = "mayores" if odds > 1 else "menores"
    txt = (f"Como p = {fmt_p(p)} < {alpha}, se rechaza H₀: hay evidencia estadística de asociación entre «{v}» y la bancarrota. "
           f"Con {v} = {hi:g} quebró el {t_hi:.2%} de las empresas ({a:,} de {a + b:,}) frente al {t_lo:.2%} con {v} = {lo:g} ({c:,} de {c + d:,}): "
           f"las odds de quiebra son {dirn} (OR = {fmt_or(odds)}" + (f", IC 95 %: {ci[0]:.2f}–{ci[1]:.2f}" if ci else "") + ")."
           if rech else
           f"Como p = {fmt_p(p)} ≥ {alpha}, no se rechaza H₀: no hay evidencia suficiente de asociación entre «{v}» y la bancarrota. "
           f"Con {v} = {hi:g} quebró el {t_hi:.2%} ({a:,} de {a + b:,}) y con {v} = {lo:g} el {t_lo:.2%} ({c:,} de {c + d:,}); "
           f"OR = {fmt_or(odds)}" + (f" (IC 95 %: {ci[0]:.2f}–{ci[1]:.2f})" if ci else "") +
           f". Esto no demuestra independencia: con solo {min(a + b, c + d):,} empresas en el grupo más pequeño la prueba puede tener poco poder.")
    if min(a, b, c, d) < 5:
        txt += " Alguna celda de la tabla tiene menos de 5 empresas: Fisher sigue siendo válida, pero el OR es una estimación inestable."
    if ci and (ci[0] <= 1 <= ci[1]) == rech:    # IC aproximado (Woolf) y p exacto pueden discrepar con celdas pequeñas
        txt += " El IC es una aproximación y aquí no coincide con el p-valor; con grupos tan pequeños prevalece el p-valor exacto de Fisher."
    txt += " Es una asociación descriptiva, no prueba causalidad."
    hyp = html.Div([
        html.Div([html.B("H₀: "), f"la bancarrota es independiente de «{v}»: la probabilidad de quiebra es la misma con {v} = {lo:g} y {v} = {hi:g} (OR = 1)."]),
        html.Div([html.B("H₁: "), "existe asociación: las odds de quiebra difieren entre ambos niveles (OR ≠ 1, prueba bilateral)."]),
        html.Div([html.B("Significancia: "), f"α = {alpha} · prueba exacta de Fisher bilateral · tabla 2×2: {a:,} / {b:,} / {c:,} / {d:,} "
                                              f"(quebró y no quebró con {v} = {hi:g}; quebró y no quebró con {v} = {lo:g})."]),
    ], className="hyp")
    res = html.Div([
        kpi("Odds Ratio (OR)", fmt_or(odds), f"Odds de quiebra con {v} = {hi:g} divididas entre las odds con {v} = {lo:g}. OR > 1: más quiebra con {hi:g}; OR < 1: menos; OR = 1: sin asociación."),
        kpi("p-valor (Fisher exacta)", fmt_p(p), "Probabilidad de observar una asociación tan fuerte o más si H₀ fuera cierta. Menor que α = 0.05 → se rechaza H₀."),
        kpi("IC 95 % del OR", f"{ci[0]:.2f} – {ci[1]:.2f}" if ci else "No estimable",
            "Rango de valores plausibles para el OR (aproximación de Woolf). Si incluye 1, la asociación no es significativa; muy amplio = estimación inestable. Con celdas pequeñas puede discrepar del p-valor exacto, que es el que prevalece. No estimable si alguna celda vale 0."),
    ], className="grid3 inner")
    chip = html.Span("Se rechaza H₀" if rech else "No se rechaza H₀", className="chip" + ("" if rech else " chip--warn"))
    return html.Div([head_(chip), hyp, res, html.P([html.B("Interpretación de la prueba: "), txt], className="interp")], className="card")

def inference_block(v):
    """Prueba U de Mann-Whitney para v: hipótesis, resultados y conclusión."""
    m = mann_whitney(v); p, r, a = m["p"], m["r"], .05
    rech = p < a
    quebradas = "menores" if r > 0 else "mayores"
    if rech:
        pract = (f"Sin embargo, el tamaño de efecto es {mag(r)} (r biserial = {r:+.2f}): con {m['n0'] + m['n1']:,} empresas incluso diferencias pequeñas resultan "
                 "significativas, por lo que no conviene confundir significancia estadística con relevancia práctica." if abs(r) < .3 else
                 f"Además, el tamaño de efecto es {mag(r)} (r biserial = {r:+.2f}), es decir, la diferencia también es relevante en la práctica.")
        txt = (f"Como p = {fmt_p(p)} < {a}, se rechaza H₀: hay evidencia estadística de que «{v}» se distribuye de forma distinta entre empresas quebradas y no quebradas; "
               f"las quebradas tienden a presentar valores {quebradas}. {pract}")
    else:
        txt = (f"Como p = {fmt_p(p)} ≥ {a}, no se rechaza H₀: no hay evidencia suficiente de que «{v}» se distribuya distinto entre ambos grupos "
               f"(r biserial = {r:+.2f}, efecto {mag(r)}). Esto no demuestra que sean iguales: con solo {m['n1']:,} empresas quebradas la prueba puede tener poco poder para detectar diferencias.")
    txt += " Es una asociación descriptiva, no prueba causalidad."
    hyp = html.Div([
        html.Div([html.B("H₀: "), f"la distribución de «{v}» es la misma en las empresas quebradas y no quebradas (ninguno de los grupos tiende a tener valores mayores)."]),
        html.Div([html.B("H₁: "), "las distribuciones difieren: un grupo tiende a tener valores mayores que el otro (prueba bilateral)."]),
        html.Div([html.B("Significancia: "), f"α = {a} · aproximación normal con corrección por empates y por continuidad · n₀ = {m['n0']:,} (no quebradas), n₁ = {m['n1']:,} (quebradas)."]),
    ], className="hyp")
    res = html.Div([
        kpi("Estadístico U", f"{m['U']:,.1f}", f"U del grupo No bancarrota: cuenta cuántas veces una empresa no quebrada supera a una quebrada (empates = ½). Con grupos idénticos vale n₀·n₁/2 = {m['n0'] * m['n1'] / 2:,.1f}."),
        kpi("p-valor", fmt_p(p), "Probabilidad de ver una diferencia tan grande o mayor si H₀ fuera cierta. Menor que α = 0.05 → se rechaza H₀."),
        kpi("r biserial (tamaño de efecto)", f"{r:+.2f}", "Magnitud de la diferencia, de −1 a +1. Positivo: valores mayores en no quebradas. ±0,1 pequeño, ±0,3 moderado, ±0,5 grande."),
    ], className="grid3 inner")
    chip = html.Span("Se rechaza H₀" if rech else "No se rechaza H₀", className="chip" + ("" if rech else " chip--warn"))
    return html.Div([html.Div([html.H4(HELP["bv_infer"][0]), chip, info("bv_infer")], className="card-h"), hyp, res,
                     html.P([html.B("Interpretación de la prueba: "), txt], className="interp")], className="card")


@lru_cache(None)
def panorama_uni():
    st = feat_stats()
    o = st["outliers"].sort_values().tail(15).rename("Outliers").rename_axis("Variable").reset_index()
    sk = st["asimetria"].reindex(st["asimetria"].abs().sort_values().tail(15).index).rename("Asimetría").rename_axis("Variable").reset_index()
    f1 = hbar(o, "Outliers", "Variable", color_discrete_sequence=[ROSE]); f1.update_xaxes(tickformat=".0%")
    f2 = hbar(sk, "Asimetría", "Variable", color_discrete_sequence=[ACCENT])
    vo, vs = st["outliers"].idxmax(), st["asimetria"].abs().idxmax()
    t1 = f"«{vo}» tiene {st['outliers'].max():.1%} de outliers (IQR); en {int((st['outliers'] > .1).sum())} de {len(st)} variables más del 10 % de las empresas es outlier."
    t2 = f"«{vs}» es la más asimétrica ({st['asimetria'][vs]:+.1f}); {int((st['asimetria'].abs() > 1).sum())} variables tienen asimetría marcada (|asimetría| > 1): candidatas a transformación o winsorizing."
    return html.Div([html.H4("Panorama de todas las variables cuantitativas"),
                     html.Div([graph("pan_out", f1, 580, t1), graph("pan_skew", f2, 580, t2)], className="grid2")])

@lru_cache(None)
def panorama_bi():
    e = effects()
    t = e.reindex(e.abs().sort_values().tail(15).index).rename("r").rename_axis("Variable").reset_index()
    t["Dirección"] = np.where(t["r"] > 0, "Mayor en no quebradas", "Mayor en quebradas")
    fig = hbar(t, "r", "Variable", color="Dirección",
               color_discrete_map={"Mayor en no quebradas": COLORS["No bancarrota"], "Mayor en quebradas": COLORS["Bancarrota"]})
    top = e.abs().idxmax()
    return html.Div([html.H4("Panorama: qué variables separan mejor a los grupos"),
                     graph("pan_bi", fig, 580, f"«{top}» es la que más separa a los grupos (r biserial = {e[top]:+.2f}); {int((e.abs() >= .3).sum())} de {len(e)} "
                                     "variables tienen efecto moderado o mayor (|r| ≥ 0,3). Selecciónala arriba para ver su detalle.")])

@lru_cache(None)
def radar():
    cols = list(BY_ABS.index[:8])
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    zg = z.groupby(df["Estado"]).mean().T
    g = zg.rename_axis("Variable").reset_index().melt(id_vars="Variable", var_name="Estado", value_name="z")
    g["Variable"] = g["Variable"].map(lambda t: wrap(t, 22))
    fig = px.line_polar(g, r="z", theta="Variable", color="Estado", line_close=True, color_discrete_map=COLORS)
    gap = (zg["Bancarrota"] - zg["No bancarrota"]).abs().idxmax()
    return html.Div([html.H4("Perfil de las empresas quebradas vs. no quebradas"),
                     graph("radar", fig, 460, f"Cada eje es una de las 8 variables más asociadas a la quiebra, estandarizada (0 = promedio general). La mayor diferencia "
                                     f"está en «{gap}» ({zg.loc[gap, 'Bancarrota']:+.2f} vs. {zg.loc[gap, 'No bancarrota']:+.2f} desviaciones). "
                                     "Cuanto más separadas estén las dos líneas, más distinto es el perfil financiero de las quebradas.")])

def density(x, y):
    d = df[df[x].between(*df[x].quantile([.01, .99])) & df[y].between(*df[y].quantile([.01, .99]))]
    fig = px.density_heatmap(d, x=x, y=y, nbinsx=40, nbinsy=40, color_continuous_scale=DENSITY)
    H, xe, ye = np.histogram2d(d[x], d[y], bins=40)
    i, j = np.unravel_index(H.argmax(), H.shape)
    return graph("density", fig, 460, f"La mayor concentración de empresas está cerca de {x} ≈ {(xe[i] + xe[i + 1]) / 2:.3g} y {y} ≈ {(ye[j] + ye[j + 1]) / 2:.3g} "
                           f"({H.max() / H.sum():.1%} de las empresas en esa celda). Se recortó el 1 % extremo de cada variable.")



@lru_cache(None)
def top_pairs():
    c = df[FEATS].corr().abs(); i, j = np.triu_indices_from(c, k=1)
    p = pd.DataFrame({"a": c.index[i], "b": c.columns[j], "r": c.values[i, j]}).dropna().sort_values("r", ascending=False).head(10)
    p["Par"] = [wrap(f"{a} ↔ {b}", 34) for a, b in zip(p["a"], p["b"])]
    return p[::-1]

@lru_cache(None)
def pairs_graph():
    p = top_pairs()
    fig = px.bar(p, x="r", y="Par", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_xaxes(title="|r| entre pares de variables (top 10)"); fig.update_yaxes(title="")
    return graph("pairs", fig, 560, f"Los pares más correlacionados llegan a |r| = {p['r'].max():.2f}: miden casi lo mismo (multicolinealidad). "
                           "Para un modelo lineal conviene dejar una sola variable por familia.")

@lru_cache(None)
def pca():
    X = df[FEATS].apply(lambda s: s.clip(*s.quantile([.01, .99])))
    Z = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    pcs = pd.DataFrame(U[:, :2] * S[:2], columns=["PC1", "PC2"]); pcs["Estado"] = df["Estado"].values
    return pcs, S ** 2 / np.sum(S ** 2)

@lru_cache(None)
def pca_graph():
    pcs, var = pca()
    m = pcs.groupby("Estado")["PC1"].median(); iqr = pcs["PC1"].quantile(.75) - pcs["PC1"].quantile(.25)
    sep = abs(m.get("Bancarrota", 0) - m.get("No bancarrota", 0)) / iqr
    t = (f"PC1 y PC2 resumen {var[:2].sum():.0%} de la varianza de las {len(FEATS)} variables. " +
         ("Las quebradas quedan mezcladas con las sanas: las clases no se separan bien en 2D, lo que sugiere probar modelos que capten relaciones complejas (por ejemplo, de árboles)."
          if sep < .3 else "Las quebradas se desplazan respecto a las sanas, pero con traslape: hay señal, no una separación limpia."))
    fig = px.scatter(pcs, x="PC1", y="PC2", color="Estado", color_discrete_map=COLORS, opacity=.5)
    return graph("pca", fig, 560, t)

@lru_cache(None)
def short(c, n=30, w=15):
    """Etiqueta abreviada: recorta a n caracteres y parte en líneas de w."""
    return wrap(c if len(c) <= n else c[:n - 1].rstrip() + "…", w)

def matrix_graph():
    cols = list(BY_ABS.index[:4])
    d = df.sample(min(len(df), 1500), random_state=0)
    fig = px.scatter_matrix(d, dimensions=cols, color="Estado", color_discrete_map=COLORS, opacity=.5,
                            labels={c: short(c) for c in cols})
    fig.update_traces(diagonal_visible=False, marker_size=3)
    c = df[cols].corr().abs().values[np.triu_indices(4, k=1)]
    return html.Div([html.H4("Las 4 variables más asociadas a la quiebra, cruzadas de a dos"),
                     graph("matrix", fig, 750, f"Entre estas variables la correlación media es |r| = {c.mean():.2f}. Si los puntos rojos aparecen en zonas "
                                     "distintas (colas o esquinas) en varios paneles, esas variables aportan señal conjunta; si se mezclan "
                                     "con los azules, ninguna combinación de a dos separa bien a los grupos.",
                           margin=dict(l=100, b=100, t=40, r=20))])

# ---------- Secciones ----------
@lru_cache(None)
def overview():
    counts = df["Estado"].value_counts().rename_axis("Estado").reset_index(name="n")
    fig = px.pie(counts, names="Estado", values="n", hole=.55, color="Estado", color_discrete_map=COLORS)
    links = [("UCI", "https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction"),
             ("Kaggle", "https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction"),
             ("GitHub", "https://github.com/lauraformore/DASH_VIZ")]

    return html.Div([
        head("Overview", "¿Qué empresas estoy viendo?"),

        html.Div([
            html.H4("Sobre el proyecto"),
            html.P("Predecir con anticipación el riesgo de bancarrota de una empresa es un problema de alto impacto: "
                   "permite a inversionistas, entidades financieras y a la propia gerencia tomar decisiones oportunas "
                   "para prevenir o mitigar pérdidas, en lugar de reaccionar cuando la quiebra ya es inminente. Contar con un "
                   "modelo que identifique señales tempranas de riesgo financiero, a partir de indicadores contables que "
                   "las empresas ya reportan, aporta valor tanto para el análisis de crédito como para la supervisión financiera."),
            html.P("Para este proyecto se utiliza el dataset del Taiwan Economic Journal, que reúne información financiera "
                   "de 6.819 empresas que cotizaron en la Bolsa de Taiwán entre 1999 y 2009. Cada empresa está descrita por "
                   "95 ratios financieros (rentabilidad, liquidez, endeudamiento, eficiencia operativa, entre otros) y "
                   "por una variable objetivo binaria que indica si la empresa terminó en bancarrota o no."),
            html.P("Flujo de trabajo: EDA → preprocesamiento → modelos de clasificación binaria → dashboard. "
                   "Este tablero integra el EDA y, más adelante, el modelo predictivo."),
            html.P([html.B("Laura Rivera · Natalý Cárdenas"), " · Visualización de Datos, Universidad del Norte · Datos y código: "] +
                   [x for n, u in links for x in (html.A(n, href=u, target="_blank", rel="noopener"), "  ")]),
        ], className="card note"),

        html.Div([
            dcc.Tabs(id="overview-tabs", value="marco", parent_className="otabs", className="otabs-bar",
                     content_className="otabs-content", children=[
                dcc.Tab(label="Marco teórico", value="marco", className="otab", selected_className="otab otab--sel", children=[
                    html.Div([
                        html.H3("Marco teórico"),
                        html.P("El análisis de quiebra empresarial mediante ratios financieros tiene una amplia tradición "
                               "en la literatura contable y financiera. Desde los trabajos pioneros de Altman (1968) con el Z-Score "
                               "y Ohlson (1980) con modelos logit, se ha demostrado que los estados financieros contienen "
                               "señales precursoras de insolvencia."),
                        html.P("En el contexto de empresas que cotizan en bolsa, el desbalance entre empresas sanas y "
                               "quebradas suele ser pronunciado (menos del 5 % de quiebras en la muestra), lo que exige el uso "
                               "de métricas de evaluación robustas ante clases desbalanceadas (como Recall, F1-Score, PR-AUC) "
                               "y técnicas de remuestreo o ajuste de pesos."),
                    ], className="otab-body")
                ]),
                dcc.Tab(label="Objetivos", value="objetivos", className="otab", selected_className="otab otab--sel", children=[
                    html.Div([
                        html.H3("Objetivos del proyecto"),
                        html.H4("Objetivo General:"),
                        html.P("Desarrollar un análisis exploratorio y un panel visual interactivo (Dashboard) para identificar "
                               "los principales patrones e indicadores contables asociados al riesgo de bancarrota en empresas "
                               "que cotizan en bolsa."),
                        html.H4("Objetivos Específicos:"),
                        html.Ul([
                            html.Li("Caracterizar las distribuciones y evaluar el grado de desbalance de la variable objetivo (Bankrupt?)."),
                            html.Li("Identificar las variables financieras cuantitativas y cualitativas con mayor poder de separación entre empresas sanas y quebradas."),
                            html.Li("Analizar la presencia de multicolinealidad entre los 95 ratios financieros para guiar la selección de variables."),
                            html.Li("Implementar un dashboard interactivo e intuitivo que facilite la exploración multivariada y la interpretación automática de hallazgos.")
                        ])
                    ], className="otab-body")
                ]),
                dcc.Tab(label="Metodología", value="metodologia", className="otab", selected_className="otab otab--sel", children=[
                    html.Div([
                        html.H3("Metodología"),
                        html.P("El proyecto sigue un enfoque estructurado de ciencia de datos aplicado a finanzas:"),
                        html.Ol([
                            html.Li([html.B("Estructuración y Limpieza: "), "Carga de 6.819 registros y 95 variables contables, validación de valores nulos y estandarización de nombres."]),
                            html.Li([html.B("Análisis Univariado y Bivariado: "), "Evaluación de asimetría, detección de outliers (regla IQR) y cálculo del tamaño de efecto mediante r biserial de rangos."]),
                            html.Li([html.B("Análisis Multivariado: "), "Cálculo de matrices de correlación (Pearson/Spearman), reducción de dimensionalidad con PCA y matrices de dispersión."]),
                            html.Li([html.B("Visualización e Interfaz: "), "Desarrollo del tablero interactivo en Plotly Dash con soporte para modo claro/oscuro e interpretaciones automatizadas."])
                        ])
                    ], className="otab-body")
                ]),
            ])
        ], className="card"),

        html.Div([kpi("Empresas", f"{len(df):,}", "Número de filas (empresas) del dataset."),
                  kpi("Variables cuantitativas", len(FEATS), "Ratios financieros numéricos disponibles para el análisis."),
                  kpi("Variables cualitativas", len(CAT), "Indicadores binarios (0/1) del dataset."),
                  kpi("Tasa de bancarrota", f"{df[TARGET].mean():.1%}", "Porcentaje de empresas que terminó en bancarrota."),
                  kpi("Valores nulos", f"{int(df[FEATS + CAT].isna().sum().sum()):,}", "Celdas vacías en las variables de análisis. Cero = no hay que imputar.")],
                 className="grid5"),

        html.Div([graph("balance", fig, 340, interp_balance()),
                  html.Div([card_h("Muestra de datos", "sample"), table(df[[TARGET] + FEATS[:6]].head(8))], className="card")],
                 className="grid2"),
    ])


def uni():
    return html.Div([head("Univariado", "¿Cómo se distribuye esta variable?"),
                     selector("uv"), html.Div(id="uv-out")])

def bi():
    return html.Div([head("Bivariado", "¿Qué cambia entre bancarrota y no bancarrota?"),
                     selector("bv"), html.Div(id="bv-out"), panorama_bi()])

def multi():
    return html.Div([
        head("Multivariado", "¿Cómo se relacionan las variables?"),
        html.Div(field("Variables en la matriz (las más asociadas a bancarrota)",
                       dcc.Slider(3, min(25, len(FEATS)), 1, value=min(10, len(FEATS)), id="mv-n", marks=None,
                                  tooltip={"placement": "bottom"}),
                       "Cuántas variables incluir en el mapa de calor, ordenadas por su asociación con la quiebra."), className="card"),
        html.Div(id="mv-heat"),
        html.Div([field("Eje X", picker("mv-x"), "Variable en el eje horizontal del gráfico de dispersión y del mapa de densidad."),
                  field("Eje Y", picker("mv-y", FEATS[min(1, len(FEATS) - 1)]), "Variable en el eje vertical.")],
                 className="grid2 card"),
        html.Div(id="mv-sc"),
        html.Div([pairs_graph(), pca_graph()], className="grid2"),
        matrix_graph(),
    ])

@lru_cache(None)
def insights():
    """Reporte EDA calculado sobre la muestra completa, sin inferir validación predictiva."""
    n = len(df)
    n0, n1 = int((df[TARGET] == 0).sum()), int((df[TARGET] == 1).sum())
    flag = "Liability-Assets Flag"
    if flag in CAT:
        counts = [int(((df[flag] == lv) & (df[TARGET] == y)).sum())
                  for lv, y in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        a, b, c, d = counts
        flag_n = a + b
        flag_text = (f"{flag} = 1: {a} quiebras de {flag_n} empresas ({a / flag_n:.2%})."
                     if flag_n else f"No hay observaciones con {flag} = 1.")
        fisher_text = (f"Fisher bilateral: p {fmt_p(fisher_exact(a, b, c, d)['p'])}; "
                       + ("hay evidencia de asociación (α = 0.05)." if fisher_exact(a, b, c, d)['p'] < .05
                          else "no hay evidencia suficiente de asociación (α = 0.05).")
                       if flag_n and c + d else "Fisher no es contrastable con un único nivel observado.")
    else:
        flag_text, fisher_text = "Liability-Assets Flag no está disponible.", "Fisher no disponible."
    income = "Net Income Flag"
    constant = income in CAT and df[income].nunique(dropna=True) == 1
    income_text = (f"Net Income Flag es constante = {df[income].dropna().iloc[0]:g} en la muestra completa; "
                   "no es contrastable y debe excluirse por varianza cero."
                   if constant else "Comprueba la disponibilidad y variabilidad de Net Income Flag en cada fold.")
    results = {v: mann_whitney(v) for v in FEATS if df.groupby(TARGET)[v].count().reindex([0, 1], fill_value=0).min() > 0}
    significant = sum(r['p'] < .05 for r in results.values())
    leaders = sorted(results, key=lambda v: abs(results[v]['r']), reverse=True)[:3]
    effect_text = "; ".join(f"{v}: r biserial = {results[v]['r']:+.2f}" for v in leaders)
    roa = next((v for v in FEATS if v.startswith("ROA(C)")), next((v for v in FEATS if "ROA" in v), None))
    if roa:
        means = df.groupby(TARGET)[roa].mean()
        medians = df.groupby(TARGET)[roa].median()
        central = (f"{roa}: media en quiebra {means.get(1, float('nan')):.4f} frente a {means.get(0, float('nan')):.4f} en solventes; "
                   f"medianas {medians.get(1, float('nan')):.4f} y {medians.get(0, float('nan')):.4f}, respectivamente. "
                   "Son valores en la escala del CSV; no equivalen automáticamente a porcentajes financieros originales. "
                   "Compara el solapamiento de los IQR en los boxplots y el desplazamiento de los ratios de deuda.")
    else:
        central = "Compara medias, medianas e IQR por estado en Bivariado; no hay una variable ROA disponible."
    outliers = feat_stats()['outliers']
    extreme = "; ".join(f"{v}: {outliers[v]:.2%}" for v in [roa, "Debt ratio %"] if v in outliers.index)
    out_text = (f"{int((outliers > .1).sum())} de {len(FEATS)} ratios tienen más del 10% de extremos IQR. "
                + (f"Detalle: {extreme}." if extreme else "Consulta el panorama univariado."))
    corr = df[FEATS].corr().abs().to_numpy()
    pair_count = int((corr[np.triu_indices_from(corr, k=1)] > .8).sum())
    redundancy = (f"{pair_count} pares de ratios presentan |r de Pearson| > 0.80. "
                  "Los bloques de la matriz evidencian redundancia lineal; PCA resume varianza, sin demostrar por sí solo separación predictiva.")
    missing = int(df[[TARGET] + FEATS + CAT].isna().sum().sum())
    missing_text = ("No se detectaron valores faltantes en las variables de análisis." if missing == 0
                    else f"Se detectaron {missing:,} celdas faltantes en las variables de análisis; evalúa su patrón e imputación.")
    target_text = f"Solvente (0): {n0:,} ({n0 / n:.2%}); quiebra (1): {n1:,} ({n1 / n:.2%})."
    inference = (f"Mann-Whitney bilateral: {significant} de {len(results)} ratios con p < 0.05 (sin ajuste por comparaciones múltiples). "
                 f"Mayores tamaños de efecto: {effect_text}. El contraste evalúa distribuciones; interpretar diferencias de medianas exige formas comparables.")
    rows = [
        ("Estructura", f"{n:,} observaciones, {len(FEATS)} variables cuantitativas y {len(CAT)} flags cualitativos.",
         "Conservar la división de entrenamiento y validación sin data leakage."),
        ("Target (Bankrupt?)", target_text,
         "Evaluar Recall, F1-Score, AUC-ROC y AUC-PR para bancarrota. Evitar Accuracy como criterio de selección."),
        ("Variables cualitativas", flag_text + " " + income_text,
         "Excluir flags de varianza cero y evaluar la utilidad de Liability-Assets Flag dentro de la validación."),
        ("Outliers IQR", out_text,
         "No eliminar por regla mecánica; evaluar winsorización, transformaciones y escalado robusto ajustados en entrenamiento."),
        ("Asociación cuantitativa", inference,
         "Contrastar boxplots y estadísticos por grupo con U de Mann-Whitney y r biserial; controlar multiplicidad."),
        ("Asociación categórica", fisher_text + " " + flag_text,
         "Considerar Liability-Assets Flag en el pipeline con cautela por su reducido tamaño muestral."),
        ("Multicolinealidad", redundancy,
         "Aplicar regularización L1/L2, selección de características o reducción de dimensionalidad dentro de cada fold."),
    ]
    conclusions = [
        ("Desbalance severo de clases", target_text + " Las frecuencias absolutas y la evaluación deben condicionarse a este desbalance. Reporta métricas de la clase positiva, precisión y matriz de confusión."),
        ("Diferencias en tendencias centrales", central),
        ("Inferencia estadística cuantitativa", inference + " Ningún p-valor por sí solo garantiza capacidad predictiva fuera de la muestra."),
        ("Inferencia en variables cualitativas", fisher_text + " " + income_text + " Comprueba estas propiedades también en entrenamiento; el Dashboard analiza la muestra completa."),
        ("Análisis de riesgo por categorías", flag_text + " La baja frecuencia absoluta exige precaución antes de considerarlo un predictor definitivo; evalúa incertidumbre y estabilidad fuera de la muestra."),
        ("Redundancia y multicolinealidad", redundancy + " Considera modelos penalizados (Lasso/Ridge) o selección de características."),
        ("Tratamiento de extremos e imputación", missing_text + " " + out_text + " Los extremos pueden reflejar dinámicas financieras reales o errores: revisa su origen antes de tratarlos y evita eliminarlos automáticamente."),
        ("Recomendaciones para el pipeline de ML", "Implementar StratifiedKFold, imputación si procede y estandarización robusta dentro de cada fold. Comparar XGBoost, Random Forest y Regresión Logística con pesos de clase; evaluar no-linealidad y desbalance. Ajustar transformaciones, selección, remuestreo y umbral exclusivamente con entrenamiento y reservar la validación final."),
    ]
    return html.Div([
        head("Insights", "Hallazgos y conclusiones del análisis exploratorio"),
        radar(),
        dbc.Card([
            card_h("Síntesis por aspectos del EDA", "eda_summary"),
            html.Div(dbc.Table([
                html.Thead(html.Tr([html.Th(label, scope="col") for label in
                                   ["Aspecto", "Hallazgo en el Dashboard", "Recomendación preliminar para Modelado"]])),
                html.Tbody([html.Tr([html.Th(aspect, scope="row"), html.Td(finding), html.Td(recommendation)])
                            for aspect, finding, recommendation in rows]),
            ], className="eda-table"), className="eda-table-scroll", tabIndex=0,
                role="region", **{"aria-label": "Tabla de síntesis del EDA"}),
        ], className="eda-report"),
        dbc.Card([
            card_h("Conclusiones del EDA", "eda_conclusions"),
            html.P("Resultados de la muestra completa · evidencia exploratoria y recomendaciones preliminares", className="eda-caption"),
            html.Ol([html.Li([html.H5(title), html.P(body)]) for title, body in conclusions], className="eda-conclusions"),
        ], className="eda-report"),
    ])

def explorer():
    cols = [TARGET] + FEATS + CAT
    return html.Div([
        head("Data Explorer", "¿Quiero investigar por mi cuenta?"),
        html.Div([field("Columnas a mostrar", dcc.Dropdown(cols, cols[:8], id="ex-cols", multi=True),
                        "Añade o quita columnas. Ordena con las flechas del encabezado y filtra escribiendo en la fila bajo el encabezado (ej.: >0.5)."),
                  html.Button("Descargar CSV", id="ex-btn", className="btn"), dcc.Download(id="ex-dl")],
                 className="card bar"),
        html.Div(dash_table.DataTable(id="ex-tbl", page_size=15, sort_action="native", filter_action="native",
                                      filter_options={"placeholder_text": "Filtrar… (Enter)"},
                                      style_filter={"backgroundColor": "var(--card)", "color": "var(--text)"}, **TBL), className="card"),
    ])

NAV = [("overview", "Overview", "¿Qué empresas estoy viendo?", overview),
       ("uni", "Univariado", "¿Cómo se distribuye esta variable?", uni),
       ("bi", "Bivariado", "¿Qué cambia entre bancarrota y no bancarrota?", bi),
       ("multi", "Multivariado", "¿Cómo se relacionan las variables?", multi),
       ("insights", "Insights", "¿Qué descubrimos?", insights),
       ("explorer", "Data Explorer", "¿Quiero investigar por mi cuenta?", explorer)]
PAGES = {k: f for k, _, _, f in NAV}

# ---------- App ----------
app = Dash(__name__, suppress_callback_exceptions=True, title=TITLE)
server = app.server  # para gunicorn en Render

app.layout = html.Div([
    html.Aside([
        html.Div([html.Div("B", className="logo"), html.Div([html.B(TITLE), html.Small("Dashboard interactivo")])], className="brand"),
        dcc.RadioItems(id="page", value="overview", className="nav",
                       options=[{"label": html.Div([html.B(t), html.Small(q)], className="ni"), "value": k}
                                for k, t, q, _ in NAV]),
        dcc.Checklist(id="theme", value=["dark"], className="theme", persistence=True,
                      options=[{"label": " Modo noche", "value": "dark"}]),
    ], className="side"),
    html.Main(id="content", className="main"),
    html.Div([html.Button(id="modal-bg", className="modal-bg", **{"aria-label": "Cerrar ayuda"}),
              html.Div([html.Button("✕", id="modal-close", className="modal-x", **{"aria-label": "Cerrar ayuda"}),
                        html.Div(id="modal-body")], className="modal-box", role="dialog", **{"aria-modal": "true"})],
             id="modal", className="modal"),
    html.Div(id="dummy", style={"display": "none"}),
], className="app")

app.clientside_callback(
    "function(v){document.documentElement.dataset.theme=(v&&v.includes('dark'))?'dark':'light';if(!window._esc){window._esc=1;document.addEventListener('keydown',function(e){if(e.key==='Escape'){var b=document.getElementById('modal-close');if(b)b.click();}});}return '';}",
    Output("dummy", "children"), Input("theme", "value"))

@app.callback(Output("modal", "className"), Output("modal-body", "children"),
              Input({"type": "info", "key": ALL}, "n_clicks"), Input("modal-close", "n_clicks"), Input("modal-bg", "n_clicks"),
              prevent_initial_call=True)
def toggle_help(*_):
    t = ctx.triggered_id
    if isinstance(t, dict):      # botón ⓘ / guía: abre solo si hubo un clic real (no al crearse el botón)
        return ("modal open", render_help(t["key"])) if ctx.triggered[0]["value"] else (no_update, no_update)
    return "modal", no_update

@app.callback(Output("content", "children"), Input("page", "value"))
def route(p):
    return PAGES[p]()

def _sync(pref):   # al cambiar cuantitativas/cualitativas se actualiza la lista de variables
    @app.callback(Output(pref, "options"), Output(pref, "value"), Input(f"{pref}-tipo", "value"))
    def _(t):
        o = CAT if t == "cat" else FEATS
        return o, o[0]

for _p in ("uv", "bv"):
    _sync(_p)

@app.callback(Output("uv-out", "children"), Input("uv", "value"))
def on_uni(v):
    if v in CAT:   # cualitativa: tarjetas de conteo + barras de frecuencia (ancho completo)
        c = df[v].value_counts().sort_index().rename_axis("cat").reset_index(name="n")
        c["cat"] = c["cat"].astype(str)
        c["pct"] = c["n"] / len(df)
        bar = px.bar(c, x="cat", y="n", text=c["pct"].map("{:.2%}".format), color_discrete_sequence=[ACCENT])
        bar.update_xaxes(title=v, type="category")
        big, small = c.loc[c["n"].idxmax()], c.loc[c["n"].idxmin()]
        if len(c) == 1:
            t1 = f"Todas las empresas tienen {v} = {big['cat']}: la variable es constante."
            t2 = "No aporta información para distinguir empresas; candidata a eliminarse antes de modelar."
        else:
            t1 = f"El {big['pct']:.1%} de las empresas tiene {v} = {big['cat']}."
            t2 = (f"Solo {small['n']:,} empresas ({small['pct']:.2%}) tienen {v} = {small['cat']}: categoría muy poco frecuente, "
                  "ojo con la estabilidad de cualquier conclusión sobre ese grupo." if small["pct"] < .05
                  else "Las categorías están repartidas de forma razonable.")
        return html.Div([html.Div([kpi(f"{v} = {r.cat}", f"{r.n:,} ({r.pct:.2%})", "Número y porcentaje de empresas en esta categoría.") for r in c.itertuples()], className="grid2"),
                         graph("uv_bar", bar, 380, f"{t1} {t2}")])
    s = df[v].describe()   # cuantitativa: histograma + boxplot
    stats = {"Media": s["mean"], "Desv. estándar": s["std"], "Mínimo": s["min"], "Mediana": s["50%"], "Máximo": s["max"]}
    fig = px.histogram(clip(v), x=v, nbins=50, marginal="box", color_discrete_sequence=[ACCENT])
    q = df[v].quantile([.1, .5, .9])
    t = f" La mitad de las empresas tiene {v} ≤ {q[.5]:.4g}; el 10 % más bajo llega a {q[.1]:.4g} y el 90 % a {q[.9]:.4g}."
    return html.Div([html.Div([kpi(k, f"{x:.4g}", STAT_TIPS[k]) for k, x in stats.items()], className="grid5"),
                     graph("uv_hist", fig, 460, interp_dist(v) + t),
                     panorama_uni()])      # el panorama cuantitativo solo aparece con variables cuantitativas

@app.callback(Output("bv-out", "children"), Input("bv", "value"))
def on_bi(v):
    if v in CAT:   # cualitativa: gráfico único (% por nivel y estado) + Fisher
        g = df.groupby(v)[TARGET].agg(tasa="mean", n="size", k="sum").reset_index()
        g[v] = g[v].astype(str)
        full = pd.MultiIndex.from_product([sorted(df[v].unique()), ["Bancarrota", "No bancarrota"]], names=[v, "Estado"])
        ct = df.groupby([v, "Estado"]).size().reindex(full, fill_value=0).rename("n").reset_index()
        ct["pct"] = ct["n"] / ct.groupby(v)["n"].transform("sum")
        ct["txt"] = [f"{p:.2%} (n={n:,})" for p, n in zip(ct["pct"], ct["n"])]
        ct[v] = ct[v].astype(str)
        fig = px.bar(ct, x=v, y="pct", color="Estado", barmode="group", text="txt", color_discrete_map=COLORS,
                     category_orders={"Estado": ["Bancarrota", "No bancarrota"]})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title="% de empresas dentro de cada nivel", tickformat=".0%", range=[0, 1.12], tickvals=[0, .25, .5, .75, 1])
        fig.update_xaxes(type="category")
        return html.Div([html.Div([kpi(f"{v} = {r[v]}", f"{r['n']:,} empresas · {r['tasa']:.2%} quebró", "Tamaño del grupo y porcentaje que quebró dentro de él.")
                                   for _, r in g.iterrows()], className="grid2"),
                         graph("bv_cat", fig, 440, interp_cat(v, g)), fisher_block(v)])
    box = px.box(clip(v), x="Estado", y=v, color="Estado", color_discrete_map=COLORS)   # cuantitativa: boxplot + tabla + inferencia
    box.update_yaxes(title="")      # el nombre de la variable ya está en la interpretación (evita montarse con el eje)
    return html.Div([graph("bv_box", box, 400, interp_box(v)), stats_table(v), inference_block(v)])

@app.callback(Output("mv-heat", "children"), Input("mv-n", "value"))
def on_heat(n):
    cols = list(BY_ABS.index[:n]) + [TARGET]
    c = df[cols].corr()
    fig = px.imshow(c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return graph("heat", fig, 520, interp_heat(c, n))

@app.callback(Output("mv-sc", "children"), Input("mv-x", "value"), Input("mv-y", "value"))
def on_scatter(x, y):
    d = df.sample(min(len(df), 2000), random_state=0)
    fig = px.scatter(d, x=x, y=y, color="Estado", color_discrete_map=COLORS, opacity=.6)
    return html.Div([graph("scatter", fig, 460, interp_scatter(x, y)), density(x, y)], className="grid2")

@app.callback(Output("ex-tbl", "data"), Output("ex-tbl", "columns"), Input("ex-cols", "value"))
def on_cols(cols):
    cols = cols or [TARGET]
    # type="numeric": sin él, el filtro nativo busca texto ("scontains") y un número como 1 no devuelve filas
    return df[cols].round(4).to_dict("records"), [{"name": c, "id": c, "type": "numeric"} for c in cols]

@app.callback(Output("ex-dl", "data"), Input("ex-btn", "n_clicks"),
              State("ex-tbl", "derived_virtual_data"), prevent_initial_call=True)
def download(_, rows):
    if rows is None:
        return no_update
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv, "seleccion.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)
