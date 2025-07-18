# 🔬 Laboratorio de Optimización de Rutas para Colombia

Una aplicación interactiva desarrollada en Streamlit para resolver, comparar y visualizar soluciones al Problema del Vendedor Viajero (TSP) aplicado a las principales capitales de Colombia.

---

## 👥 Integrantes
* Brayan Armando Cumbalaza Vallejo
* Mateo Mora Montero

---

## 📝 Descripción del Proyecto

Este proyecto es una herramienta didáctica e interactiva diseñada para explorar las complejidades del Problema del Vendedor Viajero (TSP). La aplicación permite a los usuarios resolver el problema para un número seleccionable de ciudades colombianas utilizando dos enfoques fundamentalmente diferentes:

1.  **Un método exacto:** La formulación clásica de **Dantzig-Fulkerson-Johnson (DFJ)**, resuelta con el optimizador matemático Gurobi para garantizar la ruta más corta posible.
2.  **Un método heurístico:** Un **Algoritmo Genético (AG)** personalizable que imita los principios de la evolución natural para encontrar soluciones de alta calidad de manera eficiente.

El objetivo es permitir a los usuarios no solo encontrar la ruta óptima, sino también **visualizar, comparar y entender empíricamente** las diferencias, ventajas y desventajas entre una estrategia de optimización exacta y una heurística.

---

## ✨ Características Principales

* **Enfoque Dual de Resolución:**
    * **Método Exacto (DFJ):** Garantiza encontrar el óptimo global. Ideal para entender la base matemática del TSP y para problemas de tamaño moderado.
    * **Método Heurístico (AG):** Encuentra soluciones casi óptimas en una fracción del tiempo. Perfecto para problemas más grandes y para explorar conceptos de metaheurísticas.

* **Parámetros Altamente Configurables:**
    * **Selección Dinámica de Ciudades:** Un slider permite elegir fácilmente cuántas ciudades (de 4 a 32) incluir en el problema.
    * **Ajuste Fino del Algoritmo Genético:** Control total sobre los parámetros clave del AG: tamaño de la población, número de generaciones, tasas de elitismo, cruce y mutación, y tamaño del torneo.

* **Visualización Dinámica e Interactiva:**
    * **Animación del Solver DFJ:** Observa en tiempo real cómo el modelo matemático añade "cortes perezosos" para eliminar sub-rutas inválidas hasta converger en una solución única y óptima.
    * **Animación de la Evolución Genética:** Visualiza cómo el Algoritmo Genético mejora la ruta candidata a lo largo de las generaciones.

* **Análisis Comparativo de Resultados:**
    * Un panel de resultados resume y compara el rendimiento de ambos métodos lado a lado, mostrando métricas clave:
        * Distancia total de la ruta.
        * Tiempo de cómputo.
        * Número de iteraciones (callbacks del solver vs. generaciones del AG).
        * **Gap de optimalidad:** Un porcentaje que muestra qué tan cerca estuvo la solución heurística de la solución óptima garantizada.

* **Exploración de la Ruta Final:**
    * Una vez que se obtiene una solución (de cualquiera de los dos métodos), puedes explorarla interactivamente.
    * Selecciona cualquier ciudad como punto de partida.
    * Observa la secuencia del recorrido en texto (en sentido horario y antihorario).
    * Disfruta de una **animación final** que dibuja la ruta óptima paso a paso en el mapa.

---

## 🛠 Stack Tecnológico

* **Lenguaje**: Python
* **Optimización Exacta**: Gurobi (`gurobipy`) para resolver el modelo de Programación Entera Mixta (PEM).
* **Interfaz y Visualización**:
    * Streamlit para la construcción de la aplicación web interactiva.
    * Plotly para la creación de los mapas geoespaciales interactivos y animados.
* **Manejo de Datos**: Pandas y NumPy.

---

## ⚙️ Detalles Técnicos de los Modelos

### 1. Modelo Exacto (Formulación Dantzig-Fulkerson-Johnson)
-   **Estrategia**: Se utiliza un modelo de Programación Lineal Entera Mixta. Gurobi lo resuelve mediante un algoritmo de **Branch and Cut**.
-   **Variables**: $x_{ij} = 1$ si se viaja de la ciudad $i$ a la $j$, y $0$ en caso contrario.
-   **Restricciones Clave**:
    1.  **Restricciones de Grado**: Cada ciudad debe tener exactamente dos arcos conectados a ella (uno de entrada y uno de salida).
    2.  **Restricciones de Eliminación de Sub-rutas (SEC)**: La clave del modelo DFJ. Se añaden dinámicamente como **cortes perezosos (lazy constraints)** cada vez que el solver encuentra una solución con ciclos desconectados.
-   **Propiedades**:
    * **Complejidad:** NP-duro (peor caso exponencial, $O(n^2 2^n)$).
    * **Garantía de Optimalidad:** Siempre encuentra la mejor solución posible.

### 2. Algoritmo Genético (Heurístico)
-   **Estrategia**: Es una metaheurística inspirada en la selección natural que evoluciona una población de soluciones candidatas.
-   **Componentes**:
    1.  **Cromosoma**: Una permutación de las ciudades que representa una ruta.
    2.  **Fitness**: La distancia total de la ruta (menor es mejor).
    3.  **Selección**: **Selección por Torneo** para elegir a los "padres" de la siguiente generación.
    4.  **Cruce**: **Cruce de Orden (OX1)** para combinar dos rutas padre y crear una hija válida.
    5.  **Mutación**: Intercambio simple o inversión de una subsecuencia para mantener la diversidad genética.
    6.  **Parada**: El algoritmo se detiene al alcanzar el número máximo de generaciones o si la mejor solución no ha mejorado durante un número de generaciones definido (**parada temprana**).
-   **Propiedades**:
    * **Complejidad:** Polinomial (aproximadamente $O(G \cdot P \cdot N)$).
    * **Garantía de Optimalidad:** No garantizada. Busca encontrar óptimos locales de alta calidad de forma muy rápida.

---

## 💡 Como Prototipo para una Solución Comercial o de Consultoría 📈
Tu aplicación es la base perfecta para una herramienta de optimización logística real.

* **Público Objetivo**: Pequeñas y medianas empresas de logística, distribuidores, empresas de e-commerce o cualquier negocio con una flota de vehículos en Colombia.
* **Propuesta de Valor**: Ofreces una prueba de concepto (Proof of Concept) funcional que demuestra el potencial de ahorro en costos (combustible, tiempo) al optimizar rutas. La comparación DFJ vs. GA es clave aquí: puedes mostrar que para problemas pequeños pueden tener la ruta perfecta (DFJ), y para problemas grandes, una ruta excelente y rápida (GA).
* **Cómo "Venderlo"**:
    * **Pitch de Consultoría**: "He construido este prototipo que resuelve el problema de ruteo para un solo vehículo. Puedo adaptarlo para resolver los problemas específicos de su negocio, como el Problema de Ruteo de Vehículos (VRP), que incluye:
        * Múltiples vehículos.
        * Ventanas horarias de entrega.
        * Capacidades de los vehículos.
        * Puntos de inicio y fin diferentes."
    * **Evolución del Producto**: Para hacerlo comercial, podrías añadir funcionalidades como:
        * Carga de direcciones desde un archivo Excel/CSV por parte del usuario.
        * Integración con APIs de mapas (Google Maps, Mapbox) para usar distancias y tiempos de viaje reales.
        * Creación de cuentas de usuario para guardar rutas y resultados.

---

## 🚀 Instalación y Uso

### Requisitos Previos
* Python 3.10+
* **Gurobi Optimizer**: Debe estar instalado en el sistema y contar con una licencia activa (las licencias académicas son gratuitas).
* **Archivos de Datos**: Los archivos `ubicacion.csv` y `distancias.csv` deben estar en la misma carpeta que el script `apps.py`.

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
    streamlit run apps.py
    ```
---

## 📄 Licencia
Distribuido bajo la licencia MIT.