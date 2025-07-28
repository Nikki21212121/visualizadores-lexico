"""
Autoría: Sara Badia Climent
Proyecto de investigación: "Hacia la caracterización diacrónica del siglo XX" (CIPROM/2021/038)
Universitat de València – Grupo Val.Es.Co.

Descripción:
Este script forma parte del trabajo de investigación para el análisis microdiacrónico del léxico en el siglo XX.
Su objetivo es ofrecer una visualización interactiva de la evolución léxica mediante burbujas animadas utilizando Dash y Plotly.

Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0)
→ Puedes copiar, distribuir, modificar y adaptar este código, incluso con fines comerciales,
   siempre que cites claramente a la autora y el proyecto original.

Se permite su uso y adaptación siempre que se reconozca la autoría original.

Cita sugerida:
Badia Climent, S. (2025). Visualizador de la evolución temporal del léxico mediante líneas con Dash.
Proyecto CIPROM/2021/038, Universitat de València.

Contacto: sara.badia@uv.es
Versión: 1.0
Fecha: 20/07/2025

© Sara Badia Climent – 2025
"""







import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State

# === Cargar y preparar datos ===
file_path = '../Prueba visualización.xlsx'
df = pd.read_excel(file_path)
df = df.rename(columns={df.columns[0]: 'Palabra'})

# Detectar columnas de años
anios = [col for col in df.columns if isinstance(col, int)]

# === Normalizaciones ===
df_raw = df.copy()

df_log = df.copy()
df_log[anios] = np.log10(df[anios] + 1)

df_minmax = df.copy()
df_minmax[anios] = df_minmax[anios].apply(
    lambda fila: (fila - fila.min()) / (fila.max() - fila.min()) if fila.max() > 0 else fila,
    axis=1
)

df_1000 = df.copy()
df_1000[anios] = df_1000[anios].div(df_1000[anios].sum(axis=1), axis=0) * 1000

# === Medias móviles ===
df_raw_rolling = df_raw.copy()
df_raw_rolling[anios] = df_raw_rolling[anios].T.rolling(window=5, min_periods=1).mean().T

df_log_rolling = df_log.copy()
df_log_rolling[anios] = df_log_rolling[anios].T.rolling(window=5, min_periods=1).mean().T

df_minmax_rolling = df_minmax.copy()
df_minmax_rolling[anios] = df_minmax_rolling[anios].T.rolling(window=5, min_periods=1).mean().T

df_1000_rolling = df_1000.copy()
df_1000_rolling[anios] = df_1000_rolling[anios].T.rolling(window=5, min_periods=1).mean().T

# === Categorías disponibles ===
categorias = sorted(df['Categoría'].dropna().unique())

# === App Dash ===
app = Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(style={'padding': '20px'}, children=[

    html.H1('Frecuencia de Palabras por Año', style={'textAlign': 'center'}),

    html.Label('Modo de visualización:', style={'fontWeight': 'bold'}),
    dcc.RadioItems(
        id='modo-visualizacion',
        options=[
            {'label': 'Palabra', 'value': 'palabra'},
            {'label': 'Categoría', 'value': 'categoria'}
        ],
        value='palabra',
        labelStyle={'display': 'inline-block', 'marginRight': '20px'},
        style={'marginBottom': '20px'}
    ),

    html.Label('Selecciona categorías:', style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='categoria-seleccionada',
        options=[{'label': cat, 'value': cat} for cat in categorias],
        multi=True,
        placeholder='Selecciona una o varias categorías',
        style={'marginBottom': '20px'}
    ),

    html.Div(id='contenedor-palabras', children=[
        html.Label('Selecciona palabras (opcional):', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='palabras-seleccionadas',
            multi=True,
            placeholder='Selecciona palabras o deja vacío para mostrar todas'
        )
    ], style={'marginBottom': '20px'}),

    html.Label('Normalización:', style={'fontWeight': 'bold'}),
    dcc.RadioItems(
        id='tipo-normalizacion',
        options=[
            {'label': 'Frecuencia real', 'value': 'real'},
            {'label': 'Log base 10', 'value': 'log'},
            {'label': 'Min-Max', 'value': 'minmax'},
            {'label': 'Por mil', 'value': 'mil'}
        ],
        value='real',
        labelStyle={'display': 'inline-block', 'marginRight': '10px'},
        style={'marginBottom': '20px'}
    ),

    html.Label('Activar media móvil (5 años):', style={'fontWeight': 'bold'}),
    dcc.Checklist(
        id='media-movil',
        options=[{'label': 'Media móvil', 'value': 'roll'}],
        value=[],
        style={'marginBottom': '20px'}
    ),

    html.Label('Seleccionar automáticamente las palabras más frecuentes:', style={'fontWeight': 'bold'}),
    dcc.Checklist(
        id='auto-seleccion',
        options=[{'label': 'Seleccionar automáticamente', 'value': 'auto'}],
        value=[]
    ),

    html.Label('Número de palabras más frecuentes:', style={'fontWeight': 'bold', 'marginTop': '10px'}),
    html.Div([
        dcc.Slider(
            id='n-palabras',
            min=1,
            max=20,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1, 21)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'marginBottom': '30px'}),

    html.Label('Selecciona el rango de años:', style={'fontWeight': 'bold'}),
    dcc.RangeSlider(
        id='slider-anos',
        min=min(anios),
        max=max(anios),
        step=1,
        value=[min(anios), max(anios)],
        marks={str(a): str(a) for a in range(min(anios), max(anios) + 1, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Div([
        dcc.Graph(id='grafica-lineal', style={'height': '700px'})
    ])
])

# === Mostrar/Ocultar selector de palabras según el modo ===
@app.callback(
    Output('contenedor-palabras', 'style'),
    Input('modo-visualizacion', 'value')
)
def mostrar_ocultar_selector(modo):
    return {'display': 'block', 'marginBottom': '20px'} if modo == 'palabra' else {'display': 'none'}

# === Actualizar palabras disponibles según categorías ===
@app.callback(
    Output('palabras-seleccionadas', 'options'),
    Output('palabras-seleccionadas', 'value'),
    Input('categoria-seleccionada', 'value'),
    Input('auto-seleccion', 'value'),
    Input('n-palabras', 'value'),
    Input('modo-visualizacion', 'value')
)
def actualizar_palabras(categorias_seleccionadas, auto, n, modo):
    if modo == 'categoria':
        return [], []

    subdf = df.copy() if not categorias_seleccionadas else df[df['Categoría'].isin(categorias_seleccionadas)]

    if subdf.empty:
        return [], []

    palabras = sorted(subdf['Palabra'].unique())
    options = [{'label': p, 'value': p} for p in palabras]

    if 'auto' in auto:
        subdf['total'] = subdf[anios].sum(axis=1)
        top = subdf.sort_values('total', ascending=False)['Palabra'].head(n).tolist()
        return options, top

    return options, palabras

# === Generar gráfico ===
@app.callback(
    Output('grafica-lineal', 'figure'),
    Input('modo-visualizacion', 'value'),
    Input('categoria-seleccionada', 'value'),
    Input('palabras-seleccionadas', 'value'),
    Input('slider-anos', 'value'),
    Input('tipo-normalizacion', 'value'),
    Input('media-movil', 'value')
)
def actualizar_grafica(modo, categorias_sel, palabras_sel, rango_anios, tipo_norm, media_movil):
    fig = go.Figure()
    anio_ini, anio_fin = rango_anios
    anios_filtrados = [a for a in anios if anio_ini <= a <= anio_fin]
    usar_rolling = 'roll' in media_movil

    # Elegir dataframe según normalización
    datos = {
        'real': df_raw_rolling if usar_rolling else df_raw,
        'log': df_log_rolling if usar_rolling else df_log,
        'minmax': df_minmax_rolling if usar_rolling else df_minmax,
        'mil': df_1000_rolling if usar_rolling else df_1000
    }.get(tipo_norm, df_raw)

    y_label = {
        'real': 'Frecuencia absoluta',
        'log': 'log10(Frecuencia + 1)',
        'minmax': 'Frecuencia normalizada (Min-Max)',
        'mil': 'Frecuencia por mil'
    }.get(tipo_norm, 'Frecuencia')

    subdf = datos.copy()

    if categorias_sel:
        subdf = subdf[subdf['Categoría'].isin(categorias_sel)]

    if modo == 'palabra':
        if palabras_sel:
            subdf = subdf[subdf['Palabra'].isin(palabras_sel)]
        else:
            palabras_sel = subdf['Palabra'].unique()

        for palabra in palabras_sel:
            fila = subdf[subdf['Palabra'] == palabra]
            if not fila.empty:
                fig.add_trace(go.Scatter(
                    x=anios_filtrados,
                    y=fila[anios_filtrados].values[0],
                    mode='lines+markers',
                    name=palabra
                ))
    else:  # modo == 'categoria'
        df_grouped = subdf.groupby('Categoría')[anios].sum().reset_index()
        for _, row in df_grouped.iterrows():
            fig.add_trace(go.Scatter(
                x=anios_filtrados,
                y=row[anios_filtrados],
                mode='lines+markers',
                name=row['Categoría']
            ))

    fig.update_layout(
        title='Evolución léxica',
        xaxis_title='Año',
        yaxis_title=y_label,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f9f9f9',
        font=dict(color='#333'),
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    return fig

# === Ejecutar ===
if __name__ == '__main__':
    app.run(debug=True)


"""
Librerías utilizadas:

- pandas: para la manipulación y análisis de datos estructurados.
- numpy: para operaciones numéricas y vectorizadas.
- plotly: para la creación de gráficos interactivos (plotly.graph_objs).
- Dash: para la creación de dashboards web interactivos (Dash, dcc, html, Input, Output, State).

Estas herramientas son de código abierto y han sido fundamentales para el desarrollo de esta visualización.
Agradezco a las comunidades de desarrollo de estas librerías por hacer posible este tipo de herramientas.

"""
