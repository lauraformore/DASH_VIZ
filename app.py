import os
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State

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
COLORS = {"No bancarrota": "#6d5efc", "Bancarrota": "#f43f5e"}
CORR = df[FEATS].corrwith(df[TARGET]).dropna()
BY_ABS = CORR.abs().sort_values(ascending=False)

# ---------- Helpers ----------
def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return fig

def graph(fig, h=380, text=None):
    kids = [dcc.Graph(figure=style(fig, h), config={"displaylogo": False})]
    if text:
        kids.append(html.P([html.B("Interpretación: "), text], className="interp"))
    return html.Div(kids, className="card")

def head(t, q):
    return html.Div([html.H2(t), html.P(q)], className="head")

def kpi(t, v):
    return html.Div([html.Div(str(v), className="kv"), html.Div(t, className="kt")], className="card kpi")

def table(d, **kw):
    return dash_table.DataTable(
        data=d.round(4).to_dict("records"),
        columns=[{"name": c, "id": c} for c in d.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "var(--head)", "color": "var(--text)", "fontWeight": "600", "border": "none"},
        style_cell={"backgroundColor": "var(--card)", "color": "var(--text)", "border": "none",
                    "borderBottom": "1px solid var(--line)", "padding": "8px", "fontSize": "13px", "textAlign": "left"},
        **kw)

def clip(v):
    lo, hi = df[v].quantile([.01, .99])
    return df[df[v].between(lo, hi)]

def picker(id_, value=None):
    return dcc.Dropdown(FEATS, value or FEATS[0], id=id_, clearable=False)

def selector(pref):
    opts = [{"label": "Cuantitativas (ratios)", "value": "num"}]
    if CAT:
        opts.append({"label": "Cualitativas (flags 0/1)", "value": "cat"})
    return html.Div([dcc.RadioItems(opts, "num", id=f"{pref}-tipo", inline=True, className="radio"),
                     dcc.Dropdown(FEATS, FEATS[0], id=pref, clearable=False)], className="card")

def note(title, text):
    return html.Div([html.H4(title), html.P(text)], className="card note")

def rank_biserial(v):
    """Tamaño de efecto (+ = valores mayores en no quebradas; - = mayores en quebradas)."""
    r = df[v].rank()
    n0, n1 = int((df[TARGET] == 0).sum()), int((df[TARGET] == 1).sum())
    u0 = r[df[TARGET] == 0].sum() - n0 * (n0 + 1) / 2
    return 2 * u0 / (n0 * n1) - 1

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

def interp_hist(v):
    r = rank_biserial(v); a = abs(r)
    mag = "despreciable" if a < .1 else "pequeño" if a < .3 else "moderado" if a < .5 else "grande"
    cierre = (" Por sí sola, esta variable casi no separa a los dos grupos." if a < .1 else
              " Hay diferencia, pero con mucho traslape entre ambos grupos." if a < .3 else
              " Los dos grupos se distinguen de forma notable.")
    return (f"Las quebradas se concentran en valores {'menores' if r > 0 else 'mayores'}. El tamaño de efecto "
            f"(r biserial = {r:+.2f}) es {mag}.{cierre} Es una asociación descriptiva, no prueba causalidad.")

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
    if len(g) < 2:
        return f"«{v}» toma un único valor en todas las empresas, así que no distingue entre quebradas y no quebradas y no aporta información."
    hi, lo = g.loc[g["tasa"].idxmax()], g.loc[g["tasa"].idxmin()]
    t = f"Las empresas con {v} = {hi[v]} quiebran en {hi['tasa']:.2%} de los casos, frente a {lo['tasa']:.2%} con {v} = {lo[v]}"
    t += f" (unas {hi['tasa'] / lo['tasa']:.1f} veces más)." if lo["tasa"] > 0 else "."
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
    d = d.copy(); d[y] = d[y].str[:45]
    return px.bar(d, x=x, y=y, orientation="h", **kw)

def ks_stat(v):
    a, b = np.sort(df[df[TARGET] == 0][v].values), np.sort(df[df[TARGET] == 1][v].values)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))

def overview_corr():
    a = CORR.abs()
    fig = px.histogram(CORR.rename("r").reset_index(), x="r", nbins=30, color_discrete_sequence=["#6d5efc"])
    fig.update_xaxes(title="Correlación con la quiebra (r)")
    cierre = "ninguna variable sola explica la quiebra con fuerza." if a.max() < .3 else "hay variables con una asociación fuerte."
    return graph(fig, 320, f"{int((a < .1).sum())} de las {len(a)} variables tienen |r| < 0,1 con la quiebra y {int((a >= .2).sum())} llegan a 0,2 o más: {cierre}")

def panorama_uni():
    st = feat_stats()
    o = st["outliers"].sort_values().tail(15).rename("Outliers").rename_axis("Variable").reset_index()
    sk = st["asimetria"].reindex(st["asimetria"].abs().sort_values().tail(15).index).rename("Asimetría").rename_axis("Variable").reset_index()
    f1 = hbar(o, "Outliers", "Variable", color_discrete_sequence=["#f43f5e"]); f1.update_xaxes(tickformat=".0%")
    f2 = hbar(sk, "Asimetría", "Variable", color_discrete_sequence=["#6d5efc"])
    vo, vs = st["outliers"].idxmax(), st["asimetria"].abs().idxmax()
    t1 = f"«{vo}» tiene {st['outliers'].max():.1%} de outliers (IQR); en {int((st['outliers'] > .1).sum())} de {len(st)} variables más del 10 % de las empresas es outlier."
    t2 = f"«{vs}» es la más asimétrica ({st['asimetria'][vs]:+.1f}); {int((st['asimetria'].abs() > 1).sum())} variables tienen asimetría marcada (|asimetría| > 1): candidatas a transformación o winsorizing."
    return html.Div([html.H4("Panorama de todas las variables cuantitativas"),
                     html.Div([graph(f1, 460, t1), graph(f2, 460, t2)], className="grid2")])

def panorama_bi():
    e = effects()
    t = e.reindex(e.abs().sort_values().tail(15).index).rename("r").rename_axis("Variable").reset_index()
    t["Dirección"] = np.where(t["r"] > 0, "Mayor en no quebradas", "Mayor en quebradas")
    fig = hbar(t, "r", "Variable", color="Dirección",
               color_discrete_map={"Mayor en no quebradas": COLORS["No bancarrota"], "Mayor en quebradas": COLORS["Bancarrota"]})
    top = e.abs().idxmax()
    return html.Div([html.H4("Panorama: qué variables separan mejor a los grupos"),
                     graph(fig, 460, f"«{top}» es la que más separa a los grupos (r biserial = {e[top]:+.2f}); {int((e.abs() >= .3).sum())} de {len(e)} "
                                     "variables tienen efecto moderado o mayor (|r| ≥ 0,3). Selecciónala arriba para ver su detalle.")])

def radar():
    cols = list(BY_ABS.index[:8])
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    zg = z.groupby(df["Estado"]).mean().T
    g = zg.rename_axis("Variable").reset_index().melt(id_vars="Variable", var_name="Estado", value_name="z")
    g["Variable"] = g["Variable"].str[:28]
    fig = px.line_polar(g, r="z", theta="Variable", color="Estado", line_close=True, color_discrete_map=COLORS)
    gap = (zg["Bancarrota"] - zg["No bancarrota"]).abs().idxmax()
    return html.Div([html.H4("Perfil de las empresas quebradas vs. no quebradas"),
                     graph(fig, 460, f"Cada eje es una de las 8 variables más asociadas a la quiebra, estandarizada (0 = promedio general). La mayor diferencia "
                                     f"está en «{gap}» ({zg.loc[gap, 'Bancarrota']:+.2f} vs. {zg.loc[gap, 'No bancarrota']:+.2f} desviaciones). "
                                     "Cuanto más separadas estén las dos líneas, más distinto es el perfil financiero de las quebradas.")])

def density(x, y):
    d = df[df[x].between(*df[x].quantile([.01, .99])) & df[y].between(*df[y].quantile([.01, .99]))]
    fig = px.density_heatmap(d, x=x, y=y, nbinsx=40, nbinsy=40, color_continuous_scale="Purples")
    H, xe, ye = np.histogram2d(d[x], d[y], bins=40)
    i, j = np.unravel_index(H.argmax(), H.shape)
    return graph(fig, 460, f"La mayor concentración de empresas está cerca de {x} ≈ {(xe[i] + xe[i + 1]) / 2:.3g} y {y} ≈ {(ye[j] + ye[j + 1]) / 2:.3g} "
                           f"({H.max() / H.sum():.1%} de las empresas en esa celda). Se recortó el 1 % extremo de cada variable.")

def quartile_chart():
    cols, rows = list(BY_ABS.index[:3]), []
    for v in cols:
        q = pd.qcut(df[v].rank(method="first"), 4, labels=["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"])
        rows += [(v[:30], k, x) for k, x in df.groupby(q, observed=True)[TARGET].mean().items()]
    d = pd.DataFrame(rows, columns=["Variable", "Cuartil", "Tasa"])
    fig = px.bar(d, x="Cuartil", y="Tasa", color="Variable", barmode="group", color_discrete_sequence=["#6d5efc", "#f43f5e", "#fdab3d"])
    fig.update_yaxes(tickformat=".0%", title="Tasa de bancarrota")
    r0 = d[d["Variable"] == cols[0][:30]]["Tasa"].values
    return graph(fig, 420, f"En «{cols[0]}» la tasa de quiebra va de {r0[0]:.1%} (cuartil más bajo) a {r0[-1]:.1%} (más alto). Si la tasa sube o baja "
                           "de forma escalonada entre cuartiles, la relación es monótona y la variable sirve para ordenar empresas por riesgo.")

def effect_vs_outliers():
    e, st = effects(), feat_stats()
    d = pd.DataFrame({"Variable": e.index, "Efecto": e.abs().values, "Outliers": st["outliers"].reindex(e.index).values})
    fig = px.scatter(d, x="Outliers", y="Efecto", hover_name="Variable", color_discrete_sequence=["#6d5efc"])
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(title="|r biserial|")
    n = int(((d["Efecto"] >= .3) & (d["Outliers"] > .1)).sum())
    t = (f"{n} variables combinan efecto moderado (≥ 0,3) con más de 10 % de outliers: son de las más útiles, pero piden cuidado al transformarlas."
         if n else "Ninguna variable combina efecto moderado (≥ 0,3) con más de 10 % de outliers.")
    return graph(fig, 420, "Cada punto es una variable: arriba separa mejor a los grupos y a la derecha tiene más outliers. " + t)

@lru_cache(None)
def top_pairs():
    c = df[FEATS].corr().abs(); i, j = np.triu_indices_from(c, k=1)
    p = pd.DataFrame({"a": c.index[i], "b": c.columns[j], "r": c.values[i, j]}).dropna().sort_values("r", ascending=False).head(10)
    p["Par"] = p["a"].str.slice(0, 24) + " ↔ " + p["b"].str.slice(0, 24)
    return p[::-1]

def pairs_graph():
    p = top_pairs()
    fig = px.bar(p, x="r", y="Par", orientation="h", color_discrete_sequence=["#6d5efc"])
    fig.update_xaxes(title="|r| entre pares de variables (top 10)"); fig.update_yaxes(title="")
    return graph(fig, 420, f"Los pares más correlacionados llegan a |r| = {p['r'].max():.2f}: miden casi lo mismo (multicolinealidad). "
                           "Para un modelo lineal conviene dejar una sola variable por familia.")

@lru_cache(None)
def pca():
    X = df[FEATS].apply(lambda s: s.clip(*s.quantile([.01, .99])))
    Z = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    pcs = pd.DataFrame(U[:, :2] * S[:2], columns=["PC1", "PC2"]); pcs["Estado"] = df["Estado"].values
    return pcs, S ** 2 / np.sum(S ** 2)

def pca_graph():
    pcs, var = pca()
    m = pcs.groupby("Estado")["PC1"].median(); iqr = pcs["PC1"].quantile(.75) - pcs["PC1"].quantile(.25)
    sep = abs(m.get("Bancarrota", 0) - m.get("No bancarrota", 0)) / iqr
    t = (f"PC1 y PC2 resumen {var[:2].sum():.0%} de la varianza de las {len(FEATS)} variables. " +
         ("Las quebradas quedan mezcladas con las sanas: las clases no se separan bien en 2D, lo que sugiere probar modelos que capten relaciones complejas (por ejemplo, de árboles)."
          if sep < .3 else "Las quebradas se desplazan respecto a las sanas, pero con traslape: hay señal, no una separación limpia."))
    fig = px.scatter(pcs, x="PC1", y="PC2", color="Estado", color_discrete_map=COLORS, opacity=.5)
    return graph(fig, 420, t)

def matrix_graph():
    cols = list(BY_ABS.index[:4])
    d = df.sample(min(len(df), 1500), random_state=0)
    fig = px.scatter_matrix(d, dimensions=cols, color="Estado", color_discrete_map=COLORS, opacity=.5,
                            labels={c: c[:16] for c in cols})
    fig.update_traces(diagonal_visible=False, marker_size=3)
    c = df[cols].corr().abs().values[np.triu_indices(4, k=1)]
    return html.Div([html.H4("Las 4 variables más asociadas a la quiebra, cruzadas de a dos"),
                     graph(fig, 620, f"Entre estas variables la correlación media es |r| = {c.mean():.2f}. Si los puntos rojos aparecen en zonas "
                                     "distintas (colas o esquinas) en varios paneles, esas variables aportan señal conjunta; si se mezclan "
                                     "con los azules, ninguna combinación de a dos separa bien a los grupos.")])

# ---------- Secciones ----------
def overview():
    counts = df["Estado"].value_counts().rename_axis("Estado").reset_index(name="n")
    fig = px.pie(counts, names="Estado", values="n", hole=.55, color="Estado", color_discrete_map=COLORS)
    links = [("UCI", "https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction"),
             ("Kaggle", "https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction"),
             ("GitHub", "https://github.com/lauraformore/BancarrotaEmpresarial_VIZ")]
    return html.Div([
        head("Overview", "¿Qué empresas estoy viendo?"),
        html.Div([
            html.H4("Sobre el proyecto"),
            html.P("Predecir con anticipación el riesgo de bancarrota permite a inversionistas, entidades financieras y gerencia "
                   "actuar antes de que la quiebra sea inminente. Este proyecto usa datos del Taiwan Economic Journal: 6.819 empresas "
                   "que cotizaron en la Bolsa de Taiwán entre 1999 y 2009, cada una descrita por 95 ratios financieros "
                   "(rentabilidad, liquidez, endeudamiento, eficiencia) y una variable binaria que indica si terminó en bancarrota."),
            html.P("Flujo de trabajo: EDA → preprocesamiento → modelos de clasificación binaria → dashboard. "
                   "Este tablero integra el EDA y, más adelante, el modelo predictivo."),
            html.P([html.B("Laura Rivera · Natalý Cárdenas"), " · Visualización de Datos, Universidad del Norte · Datos y código: "] +
                   [x for n, u in links for x in (html.A(n, href=u, target="_blank"), "  ")]),
        ], className="card note"),
        html.Div([kpi("Empresas", f"{len(df):,}"), kpi("Variables cuantitativas", len(FEATS)),
                  kpi("Variables cualitativas", len(CAT)),
                  kpi("Tasa de bancarrota", f"{df[TARGET].mean():.1%}"),
                  kpi("Valores nulos", f"{int(df[FEATS + CAT].isna().sum().sum()):,}")], className="grid5"),
        html.Div([graph(fig, 340, interp_balance()),
                  html.Div([html.H4("Muestra de datos"), table(df[[TARGET] + FEATS[:6]].head(8))], className="card")],
                 className="grid2"),
        overview_corr(),
    ])

def uni():
    return html.Div([head("Univariado", "¿Cómo se distribuye esta variable?"),
                     selector("uv"), html.Div(id="uv-out"), panorama_uni()])

def bi():
    return html.Div([head("Bivariado", "¿Qué cambia entre bancarrota y no bancarrota?"),
                     selector("bv"), html.Div(id="bv-out"), panorama_bi()])

def multi():
    return html.Div([
        head("Multivariado", "¿Cómo se relacionan las variables?"),
        html.Div([html.Label("Variables en la matriz (las más asociadas a bancarrota)"),
                  dcc.Slider(3, min(25, len(FEATS)), 1, value=min(10, len(FEATS)), id="mv-n",
                             marks=None, tooltip={"placement": "bottom"})], className="card"),
        html.Div(id="mv-heat"),
        html.Div([html.Div([html.Label("Eje X"), picker("mv-x")]),
                  html.Div([html.Label("Eje Y"), picker("mv-y", FEATS[min(1, len(FEATS) - 1)])])],
                 className="grid2 card"),
        html.Div(id="mv-sc"),
        radar(),
        html.Div([pairs_graph(), pca_graph()], className="grid2"),
        matrix_graph(),
    ])

def insights():
    top = CORR.reindex(BY_ABS.index[:10]).sort_values().reset_index()
    top.columns = ["Variable", "Correlación"]
    top["Signo"] = np.where(top["Correlación"] > 0, "Sube con bancarrota", "Baja con bancarrota")
    fig = px.bar(top, x="Correlación", y="Variable", orientation="h", color="Signo",
                 color_discrete_map={"Sube con bancarrota": "#f43f5e", "Baja con bancarrota": "#6d5efc"})
    b = df[TARGET].mean()
    v = BY_ABS.index[0]
    return html.Div([
        head("Insights", "¿Qué descubrimos?"),
        html.Div([
            kpi("Variable más asociada", v), kpi("Correlación con bancarrota", f"{CORR[v]:+.2f}"),
            kpi("Desbalance", f"1 de cada {round(1 / b) if b else '—'} quebró"),
        ], className="grid3"),
        graph(fig, 420, interp_top()),
        html.Div([quartile_chart(), effect_vs_outliers()], className="grid2"),
    ])

def explorer():
    cols = [TARGET] + FEATS + CAT
    return html.Div([
        head("Data Explorer", "¿Quiero investigar por mi cuenta?"),
        html.Div([dcc.Dropdown(cols, cols[:8], id="ex-cols", multi=True),
                  html.Button("Descargar CSV", id="ex-btn", className="btn"), dcc.Download(id="ex-dl")],
                 className="card bar"),
        html.Div(dash_table.DataTable(
            id="ex-tbl", page_size=15, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "var(--head)", "color": "var(--text)", "fontWeight": "600", "border": "none"},
            style_cell={"backgroundColor": "var(--card)", "color": "var(--text)", "border": "none",
                        "borderBottom": "1px solid var(--line)", "padding": "8px", "fontSize": "13px"},
            style_filter={"backgroundColor": "var(--card)", "color": "var(--text)"}), className="card"),
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
        html.Div([html.B(TITLE), html.Small("Dashboard interactivo")], className="brand"),
        dcc.RadioItems(id="page", value="overview", className="nav",
                       options=[{"label": html.Div([html.B(t), html.Small(q)], className="ni"), "value": k}
                                for k, t, q, _ in NAV]),
        dcc.Checklist(id="theme", value=["dark"], className="theme", persistence=True,
                      options=[{"label": " Modo noche", "value": "dark"}]),
    ], className="side"),
    html.Main(id="content", className="main"),
    html.Div(id="dummy", style={"display": "none"}),
], className="app")

app.clientside_callback(
    "function(v){document.documentElement.dataset.theme=(v&&v.includes('dark'))?'dark':'light';return '';}",
    Output("dummy", "children"), Input("theme", "value"))

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
    if v in CAT:   # cualitativa: dona + barras de frecuencia
        c = df[v].value_counts().sort_index().rename_axis("cat").reset_index(name="n")
        c["cat"] = c["cat"].astype(str)
        c["pct"] = c["n"] / len(df)
        donut = px.pie(c, names="cat", values="n", hole=.55, color_discrete_sequence=["#6d5efc", "#f43f5e"])
        bar = px.bar(c, x="cat", y="n", text=c["pct"].map("{:.2%}".format), color_discrete_sequence=["#6d5efc"])
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
        return html.Div([html.Div([kpi(f"{v} = {r.cat}", f"{r.n:,} ({r.pct:.2%})") for r in c.itertuples()], className="grid2"),
                         html.Div([graph(donut, 340, t1), graph(bar, 340, t2)], className="grid2")])
    s = df[v].describe()   # cuantitativa: histograma + boxplot
    stats = {"Media": s["mean"], "Desv. estándar": s["std"], "Mínimo": s["min"], "Mediana": s["50%"], "Máximo": s["max"]}
    fig = px.histogram(clip(v), x=v, nbins=50, marginal="box", color_discrete_sequence=["#6d5efc"])
    ec = px.ecdf(clip(v), x=v, color_discrete_sequence=["#6d5efc"])
    q = df[v].quantile([.1, .5, .9])
    t = (f"La mitad de las empresas tiene {v} ≤ {q[.5]:.4g}; el 10 % más bajo llega a {q[.1]:.4g} y el 90 % a {q[.9]:.4g}. "
         "Sirve para leer percentiles: una curva muy empinada indica valores concentrados en un rango estrecho.")
    return html.Div([html.Div([kpi(k, f"{x:.4g}") for k, x in stats.items()], className="grid5"),
                     html.Div([graph(fig, 420, interp_dist(v)), graph(ec, 420, t)], className="grid2")])

@app.callback(Output("bv-out", "children"), Input("bv", "value"))
def on_bi(v):
    if v in CAT:   # cualitativa: tasa de bancarrota por categoría
        g = df.groupby(v)[TARGET].agg(tasa="mean", n="size").reset_index()
        g[v] = g[v].astype(str)
        fig = px.bar(g, x=v, y="tasa", text=g["tasa"].map("{:.2%}".format), color_discrete_sequence=["#f43f5e"])
        fig.update_yaxes(title="Tasa de bancarrota", tickformat=".0%")
        fig.update_xaxes(type="category")
        ct = df.groupby([v, "Estado"]).size().reset_index(name="n")
        ct[v] = ct[v].astype(str)
        fig2 = px.bar(ct, x=v, y="n", color="Estado", barmode="group", log_y=True, color_discrete_map=COLORS)
        pv = ct.pivot(index=v, columns="Estado", values="n").fillna(0).astype(int)
        t2 = ("Conteos por categoría (escala logarítmica, porque las quebradas son pocas): " +
              "; ".join(f"{v} = {i}: {r.get('No bancarrota', 0):,} sanas y {r.get('Bancarrota', 0):,} quebradas" for i, r in pv.iterrows()) + ".")
        return html.Div([html.Div([kpi(f"{v} = {r[v]}", f"{r['n']:,} empresas · {r['tasa']:.2%} quebró")
                                   for _, r in g.iterrows()], className="grid2"),
                         html.Div([graph(fig, 340, interp_cat(v, g)), graph(fig2, 340, t2)], className="grid2")])
    d = clip(v)   # cuantitativa: boxplot + histograma por estado
    box = px.box(d, x="Estado", y=v, color="Estado", color_discrete_map=COLORS)
    hist = px.histogram(d, x=v, color="Estado", barmode="overlay", opacity=.6, nbins=50,
                        histnorm="probability density", color_discrete_map=COLORS)
    m = df.groupby("Estado")[v].median()
    kp = [kpi(f"Mediana · {k}", f"{x:.4g}") for k, x in m.items()] + [kpi("r biserial (tamaño de efecto)", f"{rank_biserial(v):+.2f}")]
    ec = px.ecdf(d, x=v, color="Estado", color_discrete_map=COLORS)
    pq = df.groupby("Estado")[v].quantile([.1, .25, .5, .75, .9]).reset_index()
    pq.columns = ["Estado", "p", "valor"]
    pq["p"] = (pq["p"] * 100).astype(int).astype(str) + "%"
    pb = px.bar(pq, x="p", y="valor", color="Estado", barmode="group", color_discrete_map=COLORS)
    pb.update_xaxes(title="Percentil", categoryorder="array", categoryarray=["10%", "25%", "50%", "75%", "90%"])
    pv = pq.pivot(index="p", columns="Estado", values="valor")
    if (pv["Bancarrota"] < pv["No bancarrota"]).all() or (pv["Bancarrota"] > pv["No bancarrota"]).all():
        tp = f"Las quebradas están por {'debajo' if (pv['Bancarrota'] < pv['No bancarrota']).all() else 'encima'} en todos los percentiles: la diferencia es consistente a lo largo de la distribución."
    else:
        tp = "El orden entre grupos cambia según el percentil: la diferencia no es uniforme, conviene mirar también las colas."
    te = (f"La distancia máxima entre las dos curvas (KS) es {ks_stat(v):.2f}: 0 = distribuciones idénticas, 1 = separación total. "
          "Cuanto mayor, mejor distingue la variable a los grupos.")
    return html.Div([html.Div(kp, className="grid3"),
                     html.Div([graph(box, 380, interp_box(v)), graph(hist, 380, interp_hist(v))], className="grid2"),
                     html.Div([graph(ec, 380, te), graph(pb, 380, tp)], className="grid2")])

@app.callback(Output("mv-heat", "children"), Input("mv-n", "value"))
def on_heat(n):
    cols = list(BY_ABS.index[:n]) + [TARGET]
    c = df[cols].corr()
    fig = px.imshow(c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return graph(fig, 520, interp_heat(c, n))

@app.callback(Output("mv-sc", "children"), Input("mv-x", "value"), Input("mv-y", "value"))
def on_scatter(x, y):
    d = df.sample(min(len(df), 2000), random_state=0)
    fig = px.scatter(d, x=x, y=y, color="Estado", color_discrete_map=COLORS, opacity=.6)
    return html.Div([graph(fig, 460, interp_scatter(x, y)), density(x, y)], className="grid2")

@app.callback(Output("ex-tbl", "data"), Output("ex-tbl", "columns"), Input("ex-cols", "value"))
def on_cols(cols):
    cols = cols or [TARGET]
    return df[cols].round(4).to_dict("records"), [{"name": c, "id": c} for c in cols]

@app.callback(Output("ex-dl", "data"), Input("ex-btn", "n_clicks"),
              State("ex-tbl", "derived_virtual_data"), prevent_initial_call=True)
def download(_, rows):
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv, "seleccion.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)