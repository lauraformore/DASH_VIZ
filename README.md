# Predicción de bancarrota empresarial · Dashboard interactivo

Dashboard en **Plotly Dash** para explorar el riesgo de bancarrota de empresas a partir de sus estados financieros. Integra el análisis exploratorio de datos (EDA) del proyecto y, en etapas posteriores, el modelo predictivo.

Proyecto de la asignatura **Visualización de Datos**, Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte (Barranquilla, Colombia).

**Autoras:** Laura Rivera · Natalý Cárdenas

## Contexto

Anticipar el riesgo de quiebra de una empresa permite a inversionistas, entidades financieras y gerencia actuar antes de que sea inminente. Este proyecto usa el dataset del **Taiwan Economic Journal**: 6.819 empresas que cotizaron en la Bolsa de Taiwán entre 1999 y 2009, descritas por 95 ratios financieros (rentabilidad, liquidez, endeudamiento, eficiencia operativa, entre otros) y una variable binaria (`Bankrupt?`) que indica si terminaron en bancarrota.

## Secciones del dashboard

El menú vertical responde una pregunta en cada sección:

| Sección | Pregunta | Contenido |
|---|---|---|
| **Overview** | ¿Qué empresas estoy viendo? | Contexto del proyecto, KPIs, dona de clases, muestra de datos, histograma de correlaciones con la quiebra |
| **Univariado** | ¿Cómo se distribuye esta variable? | Histograma + boxplot (cuantitativas), dona + barras (cualitativas), panorama de outliers y asimetría |
| **Bivariado** | ¿Qué cambia entre bancarrota y no bancarrota? | Boxplot e histograma por estado, tasa de bancarrota por categoría, ranking de variables que mejor separan a los grupos |
| **Multivariado** | ¿Cómo se relacionan las variables? | Heatmap de correlación, scatter con ejes a elegir, radar de perfiles, pares más correlacionados, PCA en 2D, matriz de dispersión |
| **Insights** | ¿Qué descubrimos? | Top 10 de correlaciones, tasa de quiebra por cuartiles, efecto frente a outliers |
| **Data Explorer** | ¿Quiero investigar por mi cuenta? | Tabla con filtros, orden, selección de columnas y descarga en CSV |

Otras características:

- **Modo noche / modo claro**, con interruptor en la barra lateral (se recuerda entre visitas).
- **Variables cuantitativas y cualitativas separadas**: los flags 0/1 (`Liability-Assets Flag`, `Net Income Flag`) se analizan con gráficas distintas a las de los ratios.
- **Interpretación automática** debajo de cada gráfica, calculada con los datos de la variable elegida.

## Estructura del proyecto

```
dash_bancarrota/
├── app.py              # aplicación Dash (datos, gráficas, callbacks)
├── assets/
│   └── style.css       # tema claro/oscuro y estilos del menú
├── data/
│   └── data.csv        # dataset (no incluido, ver más abajo)
├── requirements.txt
└── README.md
```

## Instalación y ejecución local
Siga los siguientes pasos en Anaconda PowerShell Promt:
1. **Clona el repositorio**

   ```bash
   git clone https://github.com/lauraformore/DASH_VIZ.git
   cd DASH_VIZ
   ```

2. **Crea un entorno** (se probó con Python 3.11)

   ```bash
   conda create -n eda_dashboard python=3.11 -y
   conda activate eda_dashboard
   ```

3. **Instala las dependencias**

   ```bash
   pip install -r requirements.txt
   python -m pip install "dash-bootstrap-components>=1.6,<2"
   ```

4. **Ejecuta**

   ```bash
   python app.py
   ```

   Abre <http://127.0.0.1:8050> en el navegador.

### Dependencias

`dash>=2.18,<3`, `plotly`, `pandas`, `numpy` y `gunicorn` (solo para despliegue).

> **Nota (Windows):** si Windows bloquea algún archivo de pandas (error *"Una directiva de Control de aplicaciones bloqueó este archivo"*), instala versiones anteriores: `pip install "pandas==2.2.3" "numpy<2"`.

## Despliegue en Render

El dashboard se encuentra desplegado y accesible públicamente en el siguiente enlace:
👉 **[Ver Dashboard en Vivo](https://company-bankruptcy-prediction-dash.onrender.com/)** 

### Pasos para replicar el despliegue en Render:

1. **Repositorio:** Vincula el repositorio de GitHub a tu cuenta de [Render](https://render.com/).
2. **Crear Web Service:** Selecciona la opción **Build and deploy from a Git repository** y elige `DASH_VIZ`.
3. **Configuración del servicio:**
   - **Environment / Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server`
   - **Plan:** Free

> **Nota sobre el entorno:** 
> - En el plan gratuito de Render, la aplicación entra en modo reposo tras 15 minutos de inactividad, por lo que la primera carga puede demorar alrededor de 30 a 50 segundos.
> - En producción local, asegúrate de mantener `debug=False` en `app.run_server(debug=False)` dentro de `app.py`.

## Metodología (resumen del EDA)

- Separación **train/test** estratificada antes del análisis, para evitar *data leakage*.
- Revisión de calidad: nulos, duplicados y variables constantes.
- Desbalance de clases: la bancarrota es una clase minoritaria, por lo que se recomienda evaluar con *recall*, F1 o AUC-PR.
- Tamaño de efecto con **r biserial de rangos** y correlaciones de Spearman/Pearson para distinguir variables relevantes.
- Identificación de **multicolinealidad** entre ratios casi equivalentes.

## Enlaces

- Dataset: [UCI — Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) · [Kaggle — Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
- Repositorio: [github.com/lauraformore/BancarrotaEmpresarial_VIZ](https://github.com/lauraformore/BancarrotaEmpresarial_VIZ)
