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
Badia Climent, S. (2025). Visualizador de la evolución temporal del léxico animado con Dash.
Proyecto CIPROM/2021/038, Universitat de València.

Contacto: sara.badia@uv.es
Versión: 1.0
Fecha: 20/07/2025

© Sara Badia Climent – 2025
"""



### Este script crea una aplicación web interactiva en Dash para visualizar la
### evolución de las frecuencias de palabras o categorías a lo largo del tiempo
### mediante un gráfico de burbujas animado.
##Requisitos del archivo Excel: el archivo Excel debe tener una primera columna con palabras, una segunda con categorías, y las siguientes con los años (ej. 2000, 2001...).




import pandas as pd  #Lector de Excel
import numpy as np   #Matemáticas
import plotly.express as px  #Gráficos estadísticos
import dash #Crear entorno web interactivo
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc #Elementos gráficos

# === Cargar y preparar datos: lectura de Excel y asegurarse de nombrar primera columna como 'Palabra' ===
file_path = "../prueba_burbuja.xlsx"
df = pd.read_excel(file_path)
df = df.rename(columns={df.columns[0]: "Palabra"})

# === Conversión de datos en formato horizontal (ancho) a vertical (largo) ===
# === Conversión de datos de formato ancho (una columna por año)
#     a formato largo (una fila por palabra y año) ===

columnas_ano = [col for col in df.columns if str(col).isdigit()]
df_largo_base = df.melt(
    id_vars=["Palabra", "Categoría"],
    value_vars=columnas_ano,
    var_name="Año",
    value_name="Frecuencia"
)
df_largo_base["Año"] = df_largo_base["Año"].astype(int)
df_largo_base["Frecuencia"] = df_largo_base["Frecuencia"].fillna(0)

anios_ordenados = sorted(df_largo_base["Año"].unique())
categorias_disponibles = sorted(df_largo_base["Categoría"].dropna().unique())


# === Función que transforma los datos según el método de normalización elegido ===
# (logarítmico, min-max, por mil, o sin cambios) y los devuelve en formato largo.
def normalizar(df, metodo, usar_media_movil=False):
    df_pivot = df.pivot_table(index=["Palabra", "Categoría"], columns="Año", values="Frecuencia", fill_value=0)

    if metodo == "log":
        df_norm = np.log10(df_pivot + 1)
    elif metodo == "minmax":
        df_norm = df_pivot.apply(lambda row: (row - row.min()) / (row.max() - row.min()) if row.max() > 0 else row,
                                 axis=1)
    elif metodo == "mil":
        df_norm = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 1000
    else:
        df_norm = df_pivot.copy()

    #if usar_media_movil:
        #df_norm = df_norm.T.rolling(window=5, min_periods=1).mean().T   ##La media móvil se aplica a otro gráfico de líneas.

    df_reset = df_norm.reset_index().melt(
        id_vars=["Palabra", "Categoría"],
        var_name="Año",
        value_name="Frecuencia"
    ) # reset_index() convierte "Palabra" y "Categoría", que estaban como índices, en columnas normales.
        # melt() convierte las columnas de año en una única columna "Año", con sus frecuencias asociadas.
    df_reset["Año"] = df_reset["Año"].astype(int) # Convierte los valores de la columna "Año" a enteros.
    return df_reset


# === Inicializar app ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# === Layout: estructura visual de la app. Aquí se define lo que el usuario ve y cómo está organizado. ===
app.layout = dbc.Container([  # Container es el contenedor general que agrupa todo el contenido de la app.
    html.H1("Evolución léxica animada", className="text-center mt-4 mb-4"), #título página

    # == Elección de categoría o palabra ==
    dbc.Row([ #Línea horizontal
        dbc.Col([ #Primeras 3 columnas (máximo 12 por línea)
            html.Label("Modo de visualización:", className="fw-bold"),
            dbc.RadioItems(
                id="modo-selector",
                options=[
                    {"label": "Palabra", "value": "palabra"},
                    {"label": "Categoría", "value": "categoría"},
                ],
                value="palabra",
                inline=True
            )
        ], md=3),

        #== Botones para elegir la normalización ==
        dbc.Col([
            html.Label("Normalización:", className="fw-bold"),
            dbc.RadioItems(
                id="tipo-normalizacion",
                options=[
                    {"label": "Real", "value": "real"},
                    {"label": "Log base 10", "value": "log"},
                    {"label": "Min-Max", "value": "minmax"},
                    {"label": "Por mil", "value": "mil"}
                ],
                value="real",
                inline=True
            )
        ], md=5),

        #== Botón media móvil (desactivado ahora) ==

        dbc.Col([
            html.Label("Media móvil (5 años):", className="fw-bold"),
            dbc.Checklist(
                id="media-movil",
                options=[{"label": "Activar", "value": "roll"}],
                value=[], #No activada
                inline=True
            )
        ], md=2)

    ], className="mb-3"), #Margen inferior


    #== Filtros opcionales ==

    #== 1. Desplegable para seleccionar categorías
    dbc.Row([
        dbc.Col([
            html.Label("Filtrar por categoría (opcional):", className="fw-bold"),
            dcc.Dropdown(
                id="dropdown-categoria",
                options=[{"label": cat, "value": cat} for cat in categorias_disponibles],
                placeholder="Selecciona una categoría", # placeholder es el texto que aparece antes de seleccionar nada.
                multi=True #Permite selección múltiple de categorías
            )
        ], md=6),

        #== 2. Desplegable por palabra
        dbc.Col([
            html.Label("Seleccionar palabras (opcional):", className="fw-bold"),
            dcc.Dropdown(
                id="dropdown-palabras",
                multi=True, #Selección múltiple
                placeholder="Palabras disponibles según categoría" # placeholder es el texto que aparece antes de seleccionar nada.
            )
        ], md=6)
    ], className="mb-4"), #Margen inferior

    #== Espacio para el gráfico de burbujas
    dcc.Graph(id="grafico-burbujas", style={"height": "800px"})
], fluid=True) #Diseño responsivo


# === Callbacks - conectores de funciones con layout ===
# == 1. actualizar palabras al seleccionar categoría ==
@app.callback(
    Output("dropdown-palabras", "options"), #Modifica las opciones del menú desplegable de palabras
    Input("dropdown-categoria", "value"), #Se activa cuando se selecciona una categoría
    prevent_initial_call=True #Evita que la acción se dispare sola, solo cuando el usuario actúa
)

## Función de actualizar palabras - se ejecutará automáticamente cuando cambie el valor del desplegable de categorías ##
def actualizar_palabras(categorias_seleccionadas): #Categorías seleccionadas --> lo que el usuario ha marcado. Se creará una lista de cadenas de caracteres
    if not categorias_seleccionadas:
        return [] #Devuelve lista vacía si no se selecciona nada
    subdf = df_largo_base[df_largo_base["Categoría"].isin(categorias_seleccionadas)] #Se toma el dataframe base y se filtran solo las filas cuya categoría esté en la lista seleccionada por el usuario.
    palabras = sorted(subdf["Palabra"].unique()) #Se extraen las palabras sin repetir en orden alfabético

    # Se devuelve la lista de palabras en el formato que necesita Dash:
    # [{"label": "palabra1", "value": "palabra1"}, ...]
    return [{"label": p, "value": p} for p in palabras] #Equipara el label, lo que el usuario ve en pantalla, con el value, que es el valor interno


# === Callback 2: generar gráfico ===

#Cada vez que el usuario cambie alguno de los parámetros siguientes, ejecuta la función generar_grafico
@app.callback(
    Output("grafico-burbujas", "figure"),
    Input("modo-selector", "value"),
    Input("tipo-normalizacion", "value"),
    Input("media-movil", "value"),
    Input("dropdown-categoria", "value"),
    Input("dropdown-palabras", "value")
)
def generar_grafico(modo, tipo_norm, media_movil, categorias, palabras): #Recibe todos los valores anteriores
    usar_rolling = 'roll' in media_movil # Detecta si la opción de media móvil está activada por el usuario
    df_norm = normalizar(df_largo_base, tipo_norm, usar_rolling) #Se llama a la función normalizar, que devuelve la copia del corpus con la normalización elegida

    # === Filtrar por categorías/palabras si se ha seleccionado ===
    if categorias:
        df_norm = df_norm[df_norm["Categoría"].isin(categorias)]
    if palabras:
        df_norm = df_norm[df_norm["Palabra"].isin(palabras)]

    # == Agrupar datos según el modo elegido ==
    # Si se elige modo “categoría”, se suman todas las palabras de esa categoría.
    # Si no, se mantienen las palabras individuales (no se agrupan).
    if modo == "categoría":
        df_viz = df_norm.groupby(["Categoría", "Año"], as_index=False).agg({"Frecuencia": "sum"})
        df_viz["Entidad"] = df_viz["Categoría"]
    else:
        df_viz = df_norm.copy()
        df_viz["Entidad"] = df_viz["Palabra"]

    # == Crear el gráfico de burbujas animado con Plotly ==
    # Cada burbuja representa una palabra o categoría según el modo seleccionado.
    fig = px.scatter(
        df_viz,
        x="Entidad",
        y="Frecuencia",
        size="Frecuencia",
        color="Entidad",
        animation_frame="Año",
        size_max=80,
        title=f"Evolución por {'categoría' if modo == 'categoría' else 'palabra'}",
        labels={"Frecuencia": "Frecuencia", "Entidad": "Categoría" if modo == "categoría" else "Palabra"}
    )

    #Ajustar el diseño del gráfico
    fig.update_layout(
        height=800,
        xaxis=dict(title="Entidad", tickangle=45),
        yaxis=dict(title="Frecuencia", range=[0, df_viz["Frecuencia"].max() * 1.2] if not df_viz.empty else [0, 1]),
        font_family="Segoe UI",
        title_font_size=24,
        showlegend=False,
        margin=dict(l=60, r=60, t=100, b=140)
    )

    return fig # Devuelve el gráfico para que aparezca en la interfaz


# === Run ===
if __name__ == "__main__":
    app.run(debug=True)


"""

Librerías utilizadas:

-pandas: para la manipulación de datos.
-numpy: para operaciones matemáticas.
-plotly: para la visualización interactiva.
-dash y dash-bootstrap-components: para crear interfaces web visuales.

Estas herramientas son de código abierto y han sido fundamentales para el desarrollo de esta visualización.
Agradezco a las comunidades de desarrollo de estas librerías por hacer posible este tipo de herramientas.

"""