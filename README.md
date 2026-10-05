# Predicción de bancarrota empresarial · Dashboard interactivo

Dashboard en **Plotly Dash** para explorar el riesgo de bancarrota de empresas a partir de sus estados financieros. Integra el análisis exploratorio de datos (EDA) del proyecto y, en etapas posteriores, el modelo predictivo.

Proyecto de la asignatura **Visualización de Datos**, Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte (Barranquilla, Colombia).

**Autoras:** Laura Rivera · Natalý Cárdenas

## Contexto

Anticipar el riesgo de quiebra de una empresa permite a inversionistas, entidades financieras y gerencia actuar antes de que sea inminente. Este proyecto usa el dataset del **Taiwan Economic Journal**: 6.819 observaciones de empresas correspondientes al periodo 1999–2009, descritas por 95 variables financieras (ratios de rentabilidad, liquidez, endeudamiento, eficiencia operativa e indicadores binarios) y una variable objetivo (`Bankrupt?`) que indica si terminaron en bancarrota.

## Secciones del dashboard

El menú vertical responde una pregunta en cada sección:

| Sección | Pregunta | Contenido |
|---|---|---|
| **Overview** | ¿Qué empresas estoy viendo? | Contexto del proyecto, objetivos y metodología, KPIs, dona de balance de clases y muestra de datos |
| **Univariado** | ¿Cómo se distribuye esta variable? | Histograma + boxplot y estadísticas descriptivas (cuantitativas), conteos y barras con porcentajes (cualitativas), panorama de outliers IQR y asimetría |
| **Bivariado** | ¿Qué cambia entre bancarrota y no bancarrota? | Boxplot por estado, tabla de estadísticas descriptivas (N, media, DS, mediana, mínimo, máximo, Q1, Q3 e IQR), prueba U de Mann-Whitney y tamaño de efecto r biserial; barras agrupadas con porcentajes y conteos por categoría, prueba exacta de Fisher y ranking de variables que mejor separan a los grupos |
| **Multivariado** | ¿Cómo se relacionan las variables? | Heatmap de correlación, scatter y mapa de densidad con ejes a elegir, pares más correlacionados, PCA en 2D y matriz de dispersión |
| **Insights** | ¿Qué descubrimos? | Radar de perfiles de empresas quebradas y solventes, tabla de síntesis por siete aspectos del EDA y ocho conclusiones numeradas con hallazgos estadísticos y recomendaciones preliminares para modelado |
| **Data Explorer** | ¿Quiero investigar por mi cuenta? | Tabla con filtros, orden, selección de columnas y descarga en CSV |

Otras características:

- **Modo noche / modo claro**, con interruptor en la barra lateral (se recuerda entre visitas).
- **Variables cuantitativas y cualitativas separadas**: los flags 0/1 (`Liability-Assets Flag`, `Net Income Flag`) se analizan con gráficas y pruebas estadísticas distintas a las de los ratios.
- **Interpretación automática** debajo de cada gráfica, calculada con los datos de la variable elegida.
- **Ayuda contextual** mediante tooltips y botones `(i)` con explicaciones de los gráficos, pruebas estadísticas y limitaciones de interpretación.
- **Síntesis ejecutiva del EDA** con cifras calculadas a partir del dataset y recomendaciones sobre desbalance, valores extremos, multicolinealidad y validación sin fuga de información.

## Estructura del proyecto

```text
dash_bancarrota/
├── app.py              # aplicación Dash (datos, gráficas, pruebas estadísticas y callbacks)
├── assets/
│   └── style.css       # tema claro/oscuro, estilos del menú, tablas y conclusiones
├── data/
│   └── data.csv        # dataset utilizado por el dashboard
├── requirements.txt
└── README.md
```

## Instalación y ejecución local

Siga los siguientes pasos en Anaconda PowerShell Prompt:

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
   python -m pip install -r requirements.txt
   ```

4. **Ejecuta**

   ```bash
   python app.py
   ```

   Abre <http://127.0.0.1:8050> en el navegador.

### Dependencias

`dash>=2.18,<3`, `dash-bootstrap-components>=1.6.0,<2.0.0`, `plotly`, `pandas`, `numpy` y `gunicorn` (solo para despliegue).

> **Nota (Windows):** si Windows bloquea algún archivo de pandas (error *"Una directiva de Control de aplicaciones bloqueó este archivo"*), instala versiones anteriores: `pip install "pandas==2.2.3" "numpy<2"`.

## Despliegue en Render

El dashboard se encuentra desplegado y accesible públicamente en el siguiente enlace:

**[Dashboard Company Bankruptcy prediction](https://company-bankruptcy-prediction-dash.onrender.com/)**

### Pasos para replicar el despliegue en Render:

1. **Repositorio:** Vincula el repositorio de GitHub a tu cuenta de [Render](https://render.com/).
2. **Crear Web Service:** Selecciona la opción **Build and deploy from a Git repository** y elige `DASH_VIZ`.
3. **Configuración del servicio:**
   - **Environment / Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server`
   - **Plan:** Free

> **Nota sobre el entorno:**
> - En el plan gratuito de Render, la aplicación entra en modo reposo tras 15 minutos de inactividad; la reactivación puede tardar aproximadamente un minuto, según la [documentación de Render](https://render.com/docs/free).
> - Para una ejecución local sin modo de depuración, utiliza `app.run(debug=False)` dentro de `app.py`. En Render, el servicio se inicia mediante `gunicorn app:server`.

## Metodología (resumen del EDA)

- Exploración descriptiva sobre la **muestra completa**. Para la etapa de modelado se recomienda separación **train/test** y validación cruzada estratificada (`StratifiedKFold`), ajustando el preprocesamiento exclusivamente dentro de cada fold de entrenamiento para evitar *data leakage*.
- Revisión de calidad: valores nulos, variables constantes y extremos mediante la regla **IQR**. Los outliers no se eliminan automáticamente; se recomienda evaluar su origen y comparar winsorización, transformaciones o escalado robusto.
- Desbalance de clases: la bancarrota es una clase minoritaria (**220 de 6.819 empresas; 3,23 %**). Se recomienda evaluar con **Recall, F1-Score, AUC-ROC y AUC-PR**, evitando Accuracy como criterio principal.
- Asociación cuantitativa mediante **U de Mann-Whitney**, **r biserial de rangos** y correlaciones de **Pearson**. Los p-valores se interpretan junto con el tamaño de efecto y las distribuciones; en Insights se advierte que no están ajustados por comparaciones múltiples.
- Asociación categórica mediante la **prueba exacta de Fisher**, odds ratio e intervalo de confianza cuando corresponde. Se considera la escasa frecuencia de `Liability-Assets Flag = 1` y la varianza cero de `Net Income Flag`.
- Identificación de **multicolinealidad** entre ratios casi equivalentes mediante matrices de correlación, pares correlacionados y PCA; se recomienda evaluar regularización L1/L2, selección de características o reducción de dimensionalidad.
- Síntesis de resultados en **Insights**, distinguiendo asociaciones exploratorias de capacidad predictiva: ningún p-valor ni tamaño de efecto garantiza rendimiento fuera de la muestra.

## Enlaces

- Dataset: [UCI — Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) · [Kaggle — Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
- Repositorio: [github.com/lauraformore/BancarrotaEmpresarial_VIZ](https://github.com/lauraformore/BancarrotaEmpresarial_VIZ)
