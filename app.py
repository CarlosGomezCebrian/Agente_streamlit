import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from herramientas import (
    crear_herramientas,
    generar_grafico,
    informacion_df,
    resumen_estadistico,
)

# Inicia la aplicación
st.set_page_config(page_title="Asistente de Análisis de Datos con IA", layout="centered")
st.title("🦜 Asistente de Análisis de Datos con IA")

# Descripción de la herramienta
st.info("""
Este asistente utiliza un agente, creado con Langchain, para ayudarte a explorar, analizar y visualizar datos de forma interactiva.
Basta con subir un archivo CSV y podrás:

* 📄 **Generar reportes automáticos**:

  * **Reporte de información general**: presenta la dimensión del DataFrame, nombres y tipos de las columnas, conteo de datos nulos y duplicados, además de sugerencias de tratamientos y análisis adicionales.
  * **Reporte de estadísticas descriptivas**: muestra valores como media, mediana, desviación estándar, mínimo y máximo; identifica posibles outliers y sugiere próximos pasos con base en los patrones detectados.

* 🔎 **Hacer preguntas simples sobre los datos**: como "¿Cuál es el promedio de la columna X?", "¿Cuántos registros existen para cada categoría de la columna Y?".

* 📊 **Crear gráficos automáticamente** a partir de preguntas en lenguaje natural.

Ideal para analistas, científicos de datos y equipos que buscan agilidad e insights rápidos con apoyo de IA.
""")

# Upload de CSV
st.markdown("### 📁 Realiza la carga de tu archivo CSV")
archivo_cargado = st.file_uploader("Selecciona un archivo CSV", type="csv", label_visibility="collapsed")

if archivo_cargado:
    df = pd.read_csv(archivo_cargado)
    st.success("Archivo cargado exitosamente!")
    st.markdown("### 🔍 Primeras filas de tu conjunto de datos")
    st.dataframe(df.head())

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0
    )

    # Herramientas
    tools = crear_herramientas(df)

    # Contexto del agente
    df_head = df.head().to_markdown()
    instrucciones = f"""
        Eres un asistente de análisis de datos y respondes en castellano.
        Tienes acceso a un DataFrame pandas llamado `df` mediante las herramientas disponibles.
        Estas son las primeras filas del DataFrame:

        {df_head}

        Elige y usa las herramientas cuando sean necesarias. Responde con claridad y de forma concisa.
    """
    agente = create_agent(model=llm, tools=tools, system_prompt=instrucciones)

    def consultar_agente(pregunta: str) -> str:
        respuesta = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
        return respuesta["messages"][-1].content

    # ACCIONES RÁPIDAS
    st.markdown("---")
    st.markdown("## ⚡ Acciones rápidas")

    # Reporte de Informaciones Generales
    if st.button("📄 Reporte de Informaciones Generales", key="boton_reporte_general"):
        with st.spinner("Generando Reporte 🦜"):
            st.session_state['reporte_general'] = informacion_df.invoke({
                "pregunta": "Quiero un informe con información sobre los datos.",
                "df": df,
            })

    # Exhibe el reporte con botón de descarga
    if 'reporte_general' in st.session_state:
        with st.expander("Resultado: Reporte de Informaciones Generales"):
            st.markdown(st.session_state['reporte_general'])

            st.download_button(
                label="📥 Descargar Reporte",
                data=st.session_state['reporte_general'],
                file_name="reporte_informaciones_generales.md",
                mime="text/markdown"
            )

    # Reporte de estadísticas descriptivas
    if st.button("📄 Reporte de estadísticas descriptivas", key="boton_reporte_estadisticas"):
        with st.spinner("Generando Reporte 🦜"):
            st.session_state['reporte_estadisticas'] = resumen_estadistico.invoke({
                "pregunta": "Quiero un Reporte de estadísticas descriptivas.",
                "df": df,
            })

    # Exhibe el reporte almacenado con opción de descarga
    if 'reporte_estadisticas' in st.session_state:
        with st.expander("Resultado: Reporte de estadísticas descriptivas"):
            st.markdown(st.session_state['reporte_estadisticas'])

            st.download_button(
                label="📥 Descargar Reporte",
                data=st.session_state['reporte_estadisticas'],
                file_name="reporte_estadisticas_descritivas.md",
                mime="text/markdown"  
            )
   
   # PERGUNTA SOBRE LOS DATOS
    st.markdown("---")
    st.markdown("## 🔎 Preguntas sobre los datos")
    pregunta_sobre_datos = st.text_input("Realiza una pregunta sobre los datos (ej: 'Cuál es el promedio de tiempo de entrega?')")
    if st.button("Responder pregunta", key="responder_pregunta_datos"):
        with st.spinner("Analizando los datos 🦜"):
            st.markdown(consultar_agente(pregunta_sobre_datos))


    # GENERACIÓN DE GRÁFICOS
    st.markdown("---")
    st.markdown("## 📊 Crear gráfico con base en una pregunta")

    pregunta_grafico = st.text_input("Qué deseas visualizar? (ej: 'Genera un gráfico del promedio de tiempo de entrega por clima.')")
    if st.button("Generar gráfico", key="generar_grafico"):
        if not pregunta_grafico.strip():
            st.warning("Describe el gráfico que deseas generar.")
        else:
            with st.spinner("Generando el gráfico 🦜"):
                try:
                    # Esta acción siempre debe ejecutar la herramienta de gráfico;
                    # el agente puede decidir no seleccionarla al recibir la misma pregunta.
                    generar_grafico.invoke({"pregunta": pregunta_grafico, "df": df})
                except Exception as error:
                    st.error(f"No fue posible generar el gráfico: {error}")
