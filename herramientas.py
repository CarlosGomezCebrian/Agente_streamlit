import os
import warnings # <-- IMPORTANTE: Importamos la librería de advertencias
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

# --- FIX PARA EVITAR LA RECURSIÓN CON STREAMLIT Y SEABORN ---
pd.options.mode.chained_assignment = None  # Apaga las advertencias de Pandas
warnings.filterwarnings('ignore')          # Apaga las advertencias generales de Python
import matplotlib
matplotlib.use('Agg')                      # Mantiene Matplotlib en modo seguro
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model_name = GROQ_MODEL_NAME,
    temperature = 0
)

# Herramienta de informaciones
@tool
def informacion_df(pregunta:str, df:pd.DataFrame) -> str:
    """
    Utiliza esta herramienta siempre que el usuario solicite informaciones generales sobre el
    DataFrame, incluyendo el número de columnas y filas, nombres de las columnas y sus tipos de 
    datos, conteo de datos nulos y duplicados para dar un panorama general sobre el archivo.
    """
    shape = df.shape
    columns = df.dtypes
    nulos = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
    duplicados = df.duplicated().sum()

    plantilla_respuesta = PromptTemplate(
    template = """
               Eres un analista de datos encargado de presentar un resumen informativo sobre un **DataFrame** a partir de una {pregunta} hecha por el usuario.

                A continuación, encontrarás la información general de la base de datos:

                ================= INFORMACIÓN DEL DATAFRAME =================

                Dimensiones: {shape}

                Columnas y tipos de datos:
                {columns}

                Valores nulos por columna:
                {nulos}

                Cadenas 'nan' (en cualquier capitalización) por columna:
                {nans_str}

                Filas duplicadas: {duplicados}

                ============================================================

                Con base en esta información, redacta un resumen claro y organizado que contenga:

                1. Un título: ## Informe de información general sobre el dataset,
                2. La dimensión total del DataFrame;
                3. La descripción de cada columna (incluyendo nombre, tipo de dato y qué representa esa columna);
                4. Las columnas que contienen datos nulos, con la respectiva cantidad;
                5. Las columnas que contienen cadenas 'nan', con la respectiva cantidad;
                6. La existencia (o no) de datos duplicados;
                7. Un párrafo sobre los análisis que se pueden realizar con estos datos;
                8. Un párrafo sobre los tratamientos que se pueden aplicar a los datos.
               """,
               input_variables = ['pregunta','shape','columns','nulos','nans_str','duplicados']
    )

    cadena = plantilla_respuesta | llm | StrOutputParser()
    respuesta = cadena.invoke({"pregunta":pregunta,
                                "shape":shape,
                                "columns":columns,
                                "nulos":nulos,
                                "nans_str":nans_str,
                                "duplicados":duplicados
                                })    
    return respuesta

# Herramienta de resumen estadístico
@tool
def resumen_estadistico(pregunta: str, df:pd.DataFrame) -> str:
    """
    Utiliza esta herramienta siempre que el usuario solicite un resumen estadístico completo
    y descriptivo de la base de datos, incluyendo varias estadísticas (promedio, desvío típico, 
    mínimo, máximo, etc.).
    """

    resumen = df.describe(include='number').transpose().to_string()
    plantilla_respuesta = PromptTemplate(
        template = """
        Eres un analista de datos encargado de interpretar resultados estadísticos de una base de datos a partir de una {pregunta} realizada por el usuario.
        
            A continuación, encontrarás las estadísticas descriptivas de la base de datos:

            ================= ESTADÍSTICAS DESCRIPTIVAS =================

            {resumen}

            ============================================================
        

        Con base en estos datos, elabora un resumen explicativo con un lenguaje claro, accesible y fluido, destacando los principales puntos de los resultados. Incluye:
        
            1. Un título: ## Informe de estadísticas descriptivas;  
            2. Una visión general de las estadísticas de las columnas numéricas;  
            3. Un párrafo sobre cada una de las columnas, comentando información sobre sus valores;  
            4. Identificación de posibles valores atípicos con base en los valores mínimo y máximo;  
            5. Recomendaciones de próximos pasos en el análisis en función de los patrones identificados.          
        """,
        input_variables = ['pregunta','resumen']
    )

    cadena = plantilla_respuesta | llm | StrOutputParser()
    respuesta = cadena.invoke({"pregunta":pregunta,
                               "resumen": resumen})
    return respuesta

# Herramienta de generar gráficos
@tool
def generar_grafico(pregunta: str, df:pd.DataFrame) -> str:
    """
    Utiliza esta herramienta siempre que el usuario solicite un gráfico a partir de un DataFrame
    pandas (`df`) con base en una instrucción del usuario. La instrucción podrá contener solicitudes
    como por ejemplo: 'Crea un gráfico de promedio de tiempo de entrega por clima', 'Grafica la 
    distribución del tiempo de entrega', 'Haz un plot de la relación entre la clasificación de los
    agentes y el tiempo de entrega', entre otros. Las Palabras-clave comunes que indican el uso de 
    esta herramienta incluyen: 'crea un gráfico', 'haz un plot', 'visualiza', 'muestra la distribución', 'representación visual', etc.
    """

    columnas_info = '\n'.join([f"- {col} ({dtype})" for col,dtype in df.dtypes.items()])
    muestra_datos = df.head(3).to_dict(orient='records')

    plantilla_respuesta = PromptTemplate( 
        template = """
        Eres un especialista en visualización de datos. Tu tarea es generar **únicamente el código Python** para graficar con base en la solicitud del usuario.

        ## Solicitud del usuario:
        "{pregunta}"

        ## Metadatos del DataFrame:
        {columnas}

        ## Muestra de los datos (3 primeras filas):
        {muestra}

        ## Instrucciones obligatorias:
        1. Usa las bibliotecas `matplotlib.pyplot` (como `plt`) y `seaborn` (como `sns`);
        2. Define el tema con `sns.set_theme()`;
        3. Asegúrate de que todas las columnas mencionadas en la solicitud existan en el DataFrame llamado `df`;
        4. Elige el tipo de gráfico adecuado según el análisis solicitado:
        - **Distribución de variables numéricas**: `histplot`, `kdeplot`, `boxplot` o `violinplot`
        - **Distribución de variables categóricas**: `countplot`
        - **Comparación entre categorías**: `barplot`
        - **Relación entre variables**: `scatterplot`
        - **Series temporales**: `lineplot`, con el eje X formateado como fechas
        5. Configura el tamaño del gráfico con `figsize=(8, 4)`;
        6. Añade título y etiquetas (`labels`) apropiadas a los ejes;
        7. Posiciona el título a la izquierda con `loc='left'`, deja el `pad=20` y usa `fontsize=14`;
        8. Mantén los ticks del eje X sin rotación con `plt.xticks(rotation=0)`;
        9. Elimina los bordes superior y derecho del gráfico con `sns.despine()`;
        10. Finaliza el código con `plt.show()`.

        Devuelve ÚNICAMENTE el código Python, sin ningún texto adicional ni explicación.

        Código Python:
        """,
        input_variables = ['pregunta','columnas','muestra']
    )

    cadena = plantilla_respuesta | llm | StrOutputParser()
    script_bruto = cadena.invoke({"pregunta":pregunta,
                               "columnas": columnas_info,
                               "muestra":muestra_datos})
    script_limpio = script_bruto.replace("```python","").replace("```","")
    exec_globals = {"df":df,
                    "plt":plt,
                    "sns":sns}
    exec_locals = {}

    exec(script_limpio, exec_globals, exec_locals)
    fig = plt.gcf()

    st.pyplot(fig, clear_figure=True)

    return ""

def crear_herramientas(df):
    @tool("informaciones_df", description="""
                 Utilice esta herramienta siempre que el usuario solicite informaciones generales sobre el dataframe, 
                 incluyendo el número de columnas y filas, nombres de las columnas, y sus tipos de datos, 
                 conteo de datos nulos, y duplicados para dar un panorama general sobre el archivo.
                  """)
    def herramienta_informacion_df(pregunta: str) -> str:
        return informacion_df.invoke({"pregunta": pregunta, "df": df})

    @tool("resumen_estadistico", description="""
                    Utilice esta herramienta siempre que el usuario solicite  un resumen estadístico completo 
                    y descriptivo de la base de datos ,incluyendo varias estadísticas (promedio, desvío típico, 
                    mínimo, máximo, etc.). No utilice esta herramienta para calcular una única métrica como 
                    por ejemplo: 'Cuál es el promedio de x?' o 'Cuál es la correlación de las variables?' ; 
                    en estos casos utiliza la herramienta_codigos_python.
                    """)
    def herramienta_resumen_estadistico(pregunta: str) -> str:
        return resumen_estadistico.invoke({"pregunta": pregunta, "df": df})

    @tool("generar_grafico", description="""
                    Utilice esta herramienta siempre que el usuario solicite una gráfica a partir de un DataFrame pandas (`df`) 
                    con base en una instrucción del usuario. La instrucción puede contener solicitudes tales como: 
                    'Crea un gráfico de promedio de tiempo de entrega por clima', 
                    'Haz un plot de la distribución del tiempo de entrega', 
                    'Haz un plot entre la clasificación de los colaboradores y el tiempo de entrega'. 
                    Las palabras-clave que indican el uso de esta herramienta incluyen: 'crea un gráfico', 
                    'reliza un plot', 'plotea', 'visualiza', 'muestra la distribución', 
                    'representa graficamente', entre otras.
                    """)
    def herramienta_generar_grafico(pregunta: str) -> str:
        return generar_grafico.invoke({"pregunta": pregunta, "df": df})

    herramienta_codigos_python = PythonAstREPLTool(
        name="herramienta_codigos_python",
        locals={"df": df},
        description="""
                    Utilice esta herramienta siempre que el usuario solicite cálculos, 
                    consultas o transformaciones específicas usando Python directamente sobre el DataFrame (`df`). 
                    Ejemplos de uso incluyen: 'Cuál es el promedio de la columna X?', 
                    'Cuáles son los valores únicos de la columna Y?', 'Cuál es la correlación entre A y B?', 
                    entre otros cálculos puntuales. Evita utilizar esta herramienta para solicitudes más 
                    amplias o descriptivas tales como informaciones generales sobre el DataFrame, 
                    resumenes estadísticos completos o la generación de gráficas; en estos casos, 
                    utiliza las herramientas adecuadas.
                    """
    )

    return [
        herramienta_informacion_df,
        herramienta_resumen_estadistico,
        herramienta_generar_grafico,
        herramienta_codigos_python,
    ]
