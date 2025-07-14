# TSP Interactivo para Rutas Turísticas en Colombia

Una aplicación interactiva desarrollada en Streamlit que resuelve el Problema del Agente Viajero (TSP) para un subconjunto seleccionable de las capitales de Colombia, ordenadas por su importancia turística.

---

## 👥 Integrantes
* Brayan Armando Cumbalaza Vallejo
* Mateo Mora Montero

---

## 📝 Descripción del Proyecto

Este proyecto va más allá de una simple solución al TSP. Permite a los usuarios explorar la logística de rutas turísticas en Colombia de una manera interactiva. La aplicación utiliza un dataset de las 32 capitales departamentales, pre-ordenadas según estadísticas de turismo.

El usuario puede seleccionar un "Top N" de las ciudades más visitadas y la herramienta calculará la ruta más corta que conecta estos destinos, utilizando un modelo matemático robusto y visualizando cada paso del proceso.

---

## ✨ Características Principales

* **Selección Dinámica del "Top N"**: Mediante un slider, el usuario puede elegir cuántas de las ciudades más turísticas (desde 4 hasta 32) desea incluir en el cálculo de la ruta.
* **Visualización en Tiempo Real**: Observa cómo el optimizador Gurobi trabaja, añadiendo restricciones y eliminando sub-rutas en un mapa de Colombia actualizado en vivo.
* **Exploración de la Ruta Final**: Una vez calculada la ruta óptima, puedes:
    * Seleccionar cualquier ciudad de la ruta como punto de partida.
    * Ver la secuencia del recorrido en texto, tanto en sentido óptimo como en sentido inverso.
    * Visualizar el trazado del recorrido seleccionado en un mapa animado que dibuja la ruta paso a paso.

---

## 🛠 Stack Tecnológico

* **Lenguaje**: Python
* **Optimización**: Gurobi (`gurobipy`) para resolver el modelo de Programación Entera Mixta.
* **Interfaz y Visualización**:
    * Streamlit para la construcción de la aplicación web interactiva.
    * Plotly para la creación de los mapas geoespaciales interactivos y animados.
* **Manejo de Datos**: Pandas

---

## ⚙️ Detalles Técnicos del Modelo

El problema se resuelve implementando la célebre formulación **Dantzig-Fulkerson-Johnson (DFJ)**.
- **Modelo**: Se usan variables binarias para cada posible trayecto.
- **Motor de Solución**: Gurobi utiliza el **algoritmo Simplex** dentro de una estrategia general de **Branch and Cut**.
- **Cortes (Cuts)**: Las restricciones de eliminación de sub-rutas, clave del modelo DFJ, se añaden dinámicamente como "cortes perezosos" (lazy constraints), lo cual se visualiza en la primera animación.

---

## 🚀 Instalación y Uso

### Requisitos Previos
* Python 3.10+
* **Gurobi Optimizer**: Debe estar instalado en el sistema y contar con una licencia activa (las licencias académicas son gratuitas).
* **Archivos de Datos**: Los archivos `ubicacion.csv` y `distancias.csv` deben estar en la misma carpeta que el script de la aplicación.

### Pasos
1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd [NOMBRE-DE-LA-CARPETA]
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    # En Windows
    venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar las dependencias:**
    Crea un archivo `requirements.txt` con el siguiente contenido y luego ejecuta `pip install -r requirements.txt`.
    ```txt
    streamlit
    pandas
    gurobipy
    plotly
    numpy
    ```

4.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```
---

## 📄 Licencia
Distribuido bajo la licencia MIT.