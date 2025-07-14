# Solución al Problema del Viajante (TSP) para las Capitales de Colombia

Solución del Problema del Viajante (TSP) para encontrar la ruta óptima que conecta las 32 ciudades capitales de Colombia. El problema se modela usando Programación Lineal Entera Mixta y se resuelve con el optimizador Gurobi, que implementa métodos como el **Simplex** y **Branch and Cut**. La interfaz interactiva ha sido desarrollada en **Streamlit** para visualizar la solución.

***

# 👥 Integrantes
* Brayan Armando Cumbalaza Vallejo
* Mateo Mora Montero

***

# 📝 Descripción del Proyecto y el Problema Técnico

El **Problema del Agente Viajero** (TSP, por sus siglas en inglés) es uno de los desafíos más conocidos en el campo de la optimización combinatoria. El objetivo es simple de enunciar pero computacionalmente complejo de resolver:

> Dada una lista de ciudades y las distancias entre cada par de ellas, ¿cuál es la ruta más corta posible que visita cada ciudad exactamente una vez y regresa a la ciudad de origen?

La dificultad radica en que el número de posibles rutas crece de forma factorial ($(n-1)!/2$). Para un recorrido por las 32 capitales de Colombia, una búsqueda por fuerza bruta es computacionalmente inviable.

## Nuestra Solución: Programación Lineal y Restricciones Inteligentes

Para resolver este problema, hemos implementado un modelo de **Programación Lineal Entera Mixta (MILP)** utilizando Python y la librería Gurobi. El enfoque técnico es el siguiente:

1.  **Modelo Matemático:**
    * **Variables de Decisión:** Se define una variable binaria $x_{ij}$ para cada par de ciudades $(i, j)$. La variable toma el valor $1$ si la ruta óptima incluye el trayecto de la ciudad $i$ a la $j$, y $0$ en caso contrario.
    * **Función Objetivo:** Se busca minimizar la distancia total del recorrido. Matemáticamente, esto se expresa como:
        $$\min \sum_{i,j} d_{ij} \cdot x_{ij}$$
        donde $d_{ij}$ es la distancia entre la ciudad $i$ y la $j$.

2.  **Restricciones del Modelo:**
    * **Restricciones de Grado:** Se asegura que a cada ciudad se llega una sola vez y se sale una sola vez. Esto se logra garantizando que exactamente dos arcos (uno de entrada y uno de salida) estén conectados a cada ciudad.
    * **Eliminación de Sub-rutas (Subtours):** Una solución que satisface solo las restricciones de grado podría resultar en múltiples ciclos desconectados en lugar de una única ruta. Para evitar esto, implementamos **restricciones perezosas (lazy constraints)**. El optimizador busca una solución y, si esta contiene sub-rutas, nuestra función `subtourelim` detecta el ciclo más corto y añade dinámicamente una nueva restricción que lo prohíbe. Este proceso se repite hasta que se encuentra una ruta única y conectada.

***

# 🛠 Stack Tecnológico

### 🖥️ Backend y Optimización
* **Python** como lenguaje principal.
* **Gurobi (`gurobipy`)**: Potente motor de optimización para modelar y resolver el problema de programación lineal. Utiliza algoritmos avanzados como el **método Simplex** y **Branch and Cut**.
* **NumPy**: Para operaciones numéricas eficientes, especialmente en el manejo de coordenadas.
* **Pandas**: Utilizado para la manipulación y gestión de datos.

### 🖼️ Interfaz de Usuario (UI)
* **Streamlit**: Framework utilizado para construir la aplicación web interactiva.
* **Visualización Interactiva**: La interfaz permite al usuario generar un número variable de destinos (`st.slider`) y visualiza tanto los puntos como la ruta óptima encontrada.
* **Matplotlib**: Librería encargada de generar los gráficos 2D que muestran las ciudades y el recorrido final.

***

# 🚀 Instalación y Configuración

### Requisitos
* Python 3.10+
* **Gurobi Optimizer instalado**: Gurobi no es solo una librería de Python, requiere la instalación del optimizador en tu sistema y una licencia (ofrecen licencias académicas gratuitas).
* Un archivo `requirements.txt` con las dependencias de Python.

### Pasos para la Instalación
```bash
# 1. Clonar el repositorio
git clone [https://github.com/BrayanCumbalazaVallejo/Project_PL_Traveling-Colombia-TSP](https://github.com/BrayanCumbalazaVallejo/Project_PL_Traveling-Colombia-TSP)

# 2. Navegar al directorio del proyecto
cd Project_PL_Traveling-Colombia-TSP

# 3. Crear y activar un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 4. Instalar las dependencias de Python
pip install -r requirements.txt

# 5. Ejecutar la aplicación de Streamlit
streamlit run app.py

# Para desactivar el entorno virtual cuando termines
deactivate