import os
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
    ])

def uni():
    return html.Div([head("Univariado", "¿Cómo se distribuye esta variable?"),
                     selector("uv"), html.Div(id="uv-out")])

def bi():
    return html.Div([head("Bivariado", "¿Qué cambia entre bancarrota y no bancarrota?"),
                     selector("bv"), html.Div(id="bv-out")])

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
    return html.Div([html.Div([kpi(k, f"{x:.4g}") for k, x in stats.items()], className="grid5"),
                     graph(fig, 420, interp_dist(v))])

@app.callback(Output("bv-out", "children"), Input("bv", "value"))
def on_bi(v):
    if v in CAT:   # cualitativa: tasa de bancarrota por categoría
        g = df.groupby(v)[TARGET].agg(tasa="mean", n="size").reset_index()
        g[v] = g[v].astype(str)
        fig = px.bar(g, x=v, y="tasa", text=g["tasa"].map("{:.2%}".format), color_discrete_sequence=["#f43f5e"])
        fig.update_yaxes(title="Tasa de bancarrota", tickformat=".0%")
        fig.update_xaxes(type="category")
        return html.Div([html.Div([kpi(f"{v} = {r[v]}", f"{r['n']:,} empresas · {r['tasa']:.2%} quebró")
                                   for _, r in g.iterrows()], className="grid2"),
                         graph(fig, 340, interp_cat(v, g))])
    d = clip(v)   # cuantitativa: boxplot + histograma por estado
    box = px.box(d, x="Estado", y=v, color="Estado", color_discrete_map=COLORS)
    hist = px.histogram(d, x=v, color="Estado", barmode="overlay", opacity=.6, nbins=50,
                        histnorm="probability density", color_discrete_map=COLORS)
    m = df.groupby("Estado")[v].median()
    kp = [kpi(f"Mediana · {k}", f"{x:.4g}") for k, x in m.items()] + [kpi("r biserial (tamaño de efecto)", f"{rank_biserial(v):+.2f}")]
    return html.Div([html.Div(kp, className="grid3"),
                     html.Div([graph(box, 380, interp_box(v)), graph(hist, 380, interp_hist(v))], className="grid2")])

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
    return graph(fig, 460, interp_scatter(x, y))

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