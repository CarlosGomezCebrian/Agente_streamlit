# 🦜 Asistente de Análisis de Datos con IA

Bienvenido al **Asistente de Análisis de Datos con IA**, una aplicación interactiva construida con Streamlit, LangChain y modelos de lenguaje de Groq (como `llama-3.3-70b-versatile`) que permite explorar, analizar y visualizar conjuntos de datos (CSV) utilizando lenguaje natural.

## 🌟 Características Principales

Esta herramienta está diseñada para analistas, científicos de datos y equipos que buscan obtener *insights* rápidos a través de la inteligencia artificial. Sus funciones principales incluyen:

- **Carga y Vista Previa de Datos:** Sube cualquier archivo `.csv` y visualiza instantáneamente las primeras filas de tu conjunto de datos.
- **Reportes Automáticos Descargables:**
  - 📄 **Reporte de Información General:** Genera un resumen con las dimensiones del dataset, tipos de datos, conteo de valores nulos y duplicados, junto con sugerencias de análisis.
  - 📄 **Reporte de Estadísticas Descriptivas:** Obtén un resumen estadístico (media, desviación estándar, mínimos, máximos) con identificación de valores atípicos y recomendaciones.
- **Consultas en Lenguaje Natural:** Realiza preguntas directas sobre tus datos (ej. *"¿Cuál es el promedio de tiempo de entrega?"*) y obtén respuestas precisas gracias a un agente inteligente que ejecuta código Python subyacente.
- **Generación Automática de Gráficos:** Pide una visualización (ej. *"Genera un gráfico del promedio de tiempo de entrega por clima"*) y la aplicación crea y renderiza la figura de Matplotlib/Seaborn directamente en pantalla. Esta acción invoca siempre la herramienta de gráficos, por lo que no depende de que el agente decida seleccionarla.

## 🛠️ Tecnologías Utilizadas

- **Interfaz de Usuario:** [Streamlit](https://streamlit.io/)
- **Procesamiento de Datos:** [Pandas](https://pandas.pydata.org/)
- **Visualización:** [Matplotlib](https://matplotlib.org/) y [Seaborn](https://seaborn.pydata.org/)
- **Orquestación de IA y Agentes:** [LangChain](https://www.langchain.com/) 1.x (`create_agent`, PromptTemplates, Tools y PythonAstREPLTool)
- **Modelos de Lenguaje (LLM):** [Groq](https://groq.com/) API 

## 📂 Estructura del Proyecto

El proyecto se compone principalmente de la lógica de interfaz y el motor de herramientas del agente:

1. **`app.py` (Principal):** Contiene la interfaz de Streamlit, la configuración del agente moderno de LangChain (`create_agent`), la lectura de variables de entorno y las acciones rápidas. El botón de gráficos llama de forma directa a la herramienta de visualización y muestra los errores en pantalla si no se puede generar la figura.
2. **`herramientas.py` (Módulo):** Define las herramientas personalizadas (`@tool`) que el agente utiliza para interactuar con el DataFrame:
   - `informacion_df`: Extrae metadatos y crea un reporte general del dataset.
   - `resumen_estadistico`: Interpreta estadísticas descriptivas generadas por Pandas.
   - `generar_grafico`: Escribe y ejecuta código Python para generar gráficos de Matplotlib y Seaborn basados en la petición del usuario.
   - `herramienta_codigos_python`: Un entorno REPL (Read-Eval-Print Loop) seguro y encapsulado para consultas analíticas directas sobre el DataFrame.

## 🚀 Instalación y Configuración

Sigue estos pasos para ejecutar la aplicación en tu entorno local:

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd asistente-analisis-datos
```

### 2. Instalar dependencias
Asegúrate de tener Python instalado. Se recomienda usar un entorno virtual e instalar las versiones validadas del proyecto desde `requirements.txt`.
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Las dependencias principales actuales son LangChain 1.3.14, LangChain Groq 1.1.3, Streamlit 1.59.2, pandas 3.0.3 y Matplotlib 3.11.1.

### 3. Configurar Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto y configura tus credenciales de Groq. El código espera las siguientes variables:
```env
GROQ_API_KEY_2="tu_api_key_de_groq_aqui"
GROQ_MODEL_NAME="llama-3.3-70b-versatile"
```

### 4. Ejecutar la aplicación
Inicia el servidor de Streamlit con el siguiente comando:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador por defecto (usualmente en `http://localhost:8501`).

## 💡 Ejemplos de Uso

1. **Sube un archivo:** Haz clic en "Selecciona un archivo CSV" y carga tu dataset.
2. **Genera un reporte:** En la sección "⚡ Acciones rápidas", haz clic en "Reporte de Informaciones Generales". Podrás leer el resumen y descargarlo en formato `.md`.
3. **Pregunta a los datos:** En "🔎 Preguntas sobre los datos", escribe: *"¿Cuántos registros existen para cada categoría de la columna X?"*.
4. **Crea un gráfico:** En "📊 Crear gráfico con base en una pregunta", escribe: *"Muestra la distribución de la columna Y con un histograma"*.

Si la petición está vacía o el modelo genera código no válido para el conjunto de datos, la aplicación mostrará un mensaje de error en la misma pantalla.

---
*Desarrollado para agilizar el análisis exploratorio de datos utilizando el poder de los LLMs.*
