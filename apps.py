# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para resolver y comparar soluciones al Problema del Vendedor Viajero (TSP)
en Colombia, utilizando un método exacto (Dantzig-Fulkerson-Johnson con Gurobi) y un
método heurístico (Algoritmo Genético), con una sección interactiva para explorar la ruta final.
"""

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import copy
import os
import time
import math
import plotly.graph_objects as go
from gurobipy import Model, GRB, quicksum, tuplelist

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide")
st.title('🔬 Laboratorio de Optimización de Rutas para Colombia')
st.markdown("""
Esta aplicación te permite explorar el **Problema del Vendedor Viajero (TSP)** utilizando dos enfoques diferentes:
1.  **Método Exacto (DFJ):** Utiliza optimización matemática para **garantizar** la ruta más corta posible.
2.  **Método Heurístico (Algoritmo Genético):** Imita la evolución natural para encontrar una ruta de **muy alta calidad** de forma rápida.

Usa el control deslizante para elegir el número de ciudades, ejecuta un algoritmo y luego explora la ruta resultante en la sección inferior.
""")

# --- CARGA DE DATOS (CACHEADA) ---
@st.cache_data
def load_data():
    """Carga los archivos de ubicación y distancias desde CSV."""
    try:
        df_ubicaciones = pd.read_csv('ubicacion.csv')
        df_distancias = pd.read_csv('distancias.csv', index_col='Ciudad')
        return df_ubicaciones, df_distancias
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que 'ubicacion.csv' y 'distancias.csv' estén en la misma carpeta.")
        return None, None

# --- LÓGICA DEL MÉTODO EXACTO (GUROBI) ---
def find_subtours(edges, num_cities):
    """Encuentra ciclos (sub-rutas) en un conjunto de arcos."""
    unvisited = list(range(num_cities))
    cycles = []
    while unvisited:
        this_cycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            this_cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        cycles.append(this_cycle)
    return cycles

def subtour_elimination_callback(model, where):
    """Callback de Gurobi para añadir restricciones de eliminación de sub-rutas (SEC) de forma perezosa."""
    if where == GRB.Callback.MIPSOL:
        model._iterations += 1
        vals = model.cbGetSolution(model._vars)
        selected_edges = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        
        tours = find_subtours(selected_edges, model._n)
        if len(tours) > 1:
            model._subtours_found += len(tours) -1
            for tour in tours:
                if len(tour) < model._n:
                    model.cbLazy(quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2)) <= len(tour) - 1)
        
        current_length = round(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        model._summary_placeholder.markdown(f"""
        - **Iteración (Callback):** `{model._iterations}`
        - **Distancia Actual:** `{current_length:,} km`
        - **Sub-rutas Encontradas:** `{len(tours)}`
        - **Restricciones Añadidas:** `{model._subtours_found}`
        """)
        
        fig = draw_map(model._df_ubicaciones, tours=tours, title=f"DFJ - Iteración {model._iterations}: Eliminando sub-rutas")
        model._plot_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(4.0) # Pausa para visualización

# --- LÓGICA DEL ALGORITMO GENÉTICO ---
class TSPCandidate:
    """Representa una solución candidata (una ruta) en el Algoritmo Genético."""
    def __init__(self, chromosomes, distance_matrix):
        self.distance_matrix = distance_matrix
        if chromosomes is None:
            self.chromosomes = np.arange(len(distance_matrix))
            np.random.shuffle(self.chromosomes)
        else:
            self.chromosomes = chromosomes
        self._fitness = None

    @property
    def fitness_score(self) -> float:
        """Calcula la distancia total de la ruta (el fitness). Un valor más bajo es mejor."""
        if self._fitness is None:
            rolled_chromosomes = np.roll(self.chromosomes, -1)
            distances = self.distance_matrix[self.chromosomes, rolled_chromosomes]
            self._fitness = np.sum(distances)
        return self._fitness

    def mutate(self, mutation_rate):
        """Aplica una mutación a la ruta con una cierta probabilidad."""
        if np.random.rand() < mutation_rate:
            if np.random.rand() < 0.5:
                idx1, idx2 = np.random.choice(len(self.chromosomes), 2, replace=False)
                self.chromosomes[idx1], self.chromosomes[idx2] = self.chromosomes[idx2], self.chromosomes[idx1]
            else:
                start, end = np.sort(np.random.choice(len(self.chromosomes), 2, replace=False))
                self.chromosomes[start:end+1] = self.chromosomes[start:end+1][::-1]
            self._fitness = None

    def crossover(self, other_parent: "TSPCandidate") -> "TSPCandidate":
        """Realiza un cruce ordenado (OX1) con otro padre para crear un hijo."""
        p1, p2 = self.chromosomes, other_parent.chromosomes
        n_genes = len(p1)
        start, end = np.sort(np.random.choice(n_genes, 2, replace=False))
        child_chromo = np.full(n_genes, -1, dtype=int)
        child_chromo[start:end+1] = p1[start:end+1]
        p2_idx = 0
        child_idx = 0
        while child_idx < n_genes:
            if start <= child_idx < end + 1:
                child_idx += 1
                continue
            if p2[p2_idx] not in child_chromo:
                child_chromo[child_idx] = p2[p2_idx]
                child_idx += 1
            p2_idx += 1
        return TSPCandidate(chromosomes=child_chromo, distance_matrix=self.distance_matrix)

class Population:
    """Gestiona una población de candidatos para el Algoritmo Genético."""
    def __init__(self, size, distance_matrix):
        self.distance_matrix = distance_matrix
        self.candidates = [TSPCandidate(None, self.distance_matrix) for _ in range(size)]

    def evolve(self, elitism_rate, crossover_rate, mutation_rate, tournament_size):
        """Evoluciona la población a la siguiente generación."""
        self.candidates.sort(key=lambda c: c.fitness_score)
        elitism_size = int(len(self.candidates) * elitism_rate)
        next_gen = self.candidates[:elitism_size]
        while len(next_gen) < len(self.candidates):
            p1 = self._tournament_selection(tournament_size)
            p2 = self._tournament_selection(tournament_size)
            child = p1.crossover(p2) if np.random.rand() < crossover_rate else copy.deepcopy(p1)
            child.mutate(mutation_rate)
            next_gen.append(child)
        self.candidates = next_gen

    def _tournament_selection(self, tournament_size) -> "TSPCandidate":
        """Selecciona un candidato usando el método de torneo."""
        competitors = np.random.choice(self.candidates, tournament_size, replace=False)
        return min(competitors, key=lambda c: c.fitness_score)

# --- FUNCIONES DE VISUALIZACIÓN ---
def draw_map(df_ubicaciones, route_indices=None, tours=None, title=""):
    """Dibuja el mapa de Colombia con las ciudades y opcionalmente las rutas."""
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=df_ubicaciones['Latitud'], lon=df_ubicaciones['Longitud'],
        mode='markers+text', marker={'size': 10, 'color': 'darkblue'},
        text=df_ubicaciones['Capital'], textposition='top right'
    ))
    if tours:
        for i, tour in enumerate(tours):
            if len(tour) < len(df_ubicaciones):
                tour_copy = tour[:] + [tour[0]]
                tour_points = df_ubicaciones.iloc[tour_copy]
                fig.add_trace(go.Scattermapbox(lat=tour_points['Latitud'], lon=tour_points['Longitud'], mode='lines', line={'width': 2, 'color': 'gray'}, name=f"Sub-ruta {i+1}"))
    if route_indices is not None:
        route_points = df_ubicaciones.iloc[np.append(route_indices, route_indices[0])]
        fig.add_trace(go.Scattermapbox(
            lat=route_points['Latitud'], lon=route_points['Longitud'],
            mode='lines', line={'width': 3, 'color': 'crimson'}, name="Ruta Principal"
        ))
    fig.update_layout(
        title=title, showlegend=False, mapbox_style="carto-positron", mapbox_zoom=4.2,
        mapbox_center={"lat": 4.5709, "lon": -74.2973}, margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

# --- INICIALIZACIÓN DE LA APP ---
df_ubicaciones_full, df_distancias_full = load_data()

# Inicializar session_state
if 'dfj_results' not in st.session_state: st.session_state.dfj_results = None
if 'ga_results' not in st.session_state: st.session_state.ga_results = None
if 'active_route_solution' not in st.session_state: st.session_state.active_route_solution = None
if 'selected_start_city_index' not in st.session_state: st.session_state.selected_start_city_index = None
if 'last_run_n' not in st.session_state: st.session_state.last_run_n = 0

# --- INTERFAZ DE USUARIO ---
if df_ubicaciones_full is not None:
    st.sidebar.header("⚙️ Configuración General")
    n_ciudades = st.sidebar.slider(
        "Selecciona el número de ciudades a visitar:",
        min_value=4, max_value=len(df_ubicaciones_full), value=10, key="n_slider"
    )

    if n_ciudades != st.session_state.last_run_n:
        st.session_state.dfj_results = None
        st.session_state.ga_results = None
        st.session_state.active_route_solution = None
        st.session_state.selected_start_city_index = None
        st.session_state.last_run_n = n_ciudades
    
    df_ubicaciones = df_ubicaciones_full.head(n_ciudades)
    cities_names = df_ubicaciones['Capital'].tolist()
    df_distancias_matrix = df_distancias_full.iloc[:n_ciudades, :n_ciudades].to_numpy()
    dist_dict = {(i, j): df_distancias_matrix[i, j] for i in range(n_ciudades) for j in range(i)}

    tab1, tab2 = st.tabs(["**Método Exacto (Dantzig-Fulkerson-Johnson)**", "**Método Heurístico (Algoritmo Genético)**"])

    with tab1:
        st.header("Optimización con Gurobi (Branch and Cut + DFJ / Simplex)")
        
        with st.expander("Haz clic para ver el Planteamiento del TSP como Programación Lineal"):
            st.markdown(r"""
            El Problema del Vendedor Viajero (TSP) es un famoso problema de optimización clasificado como **NP-duro**. Esto significa que no existe un algoritmo que pueda resolverlo de forma rápida y exacta para un gran número de ciudades.

            Resolverlo por **fuerza bruta** tiene una **complejidad factorial** ($O(n!)$), lo que lo vuelve computacionalmente imposible incluso para un número modesto de ciudades. Por esta razón, se deben usar métodos inteligentes como la **programación lineal entera**.

            #### Elementos Fundamentales
            - **Conjunto de Nodos (Ciudades):** $V = \{0, 1, \dots, n-1\}$
            - **Parámetro de Costo/Distancia:** $c_{ij}$, la distancia entre la ciudad $i$ y $j$.
            - **Variable de Decisión (Binaria):**
                $$
                x_{ij} = \begin{cases}
                1 & \text{si se toma la ruta entre } i \text{ y } j \\
                0 & \text{en caso contrario}
                \end{cases}
                $$

            #### Función Objetivo
            Minimizar la distancia total del recorrido:
            $$ \min \sum_{i \in V} \sum_{j > i} c_{ij} x_{ij} $$

            #### Restricciones Fundamentales (De Grado)
            Aseguran que a cada ciudad se llega una vez y se sale una vez:
            $$ \sum_{j \neq i} x_{ij} = 2 \quad \forall i \in V $$

            El principal desafío es que estas restricciones por sí solas no impiden la formación de **sub-rutas**. Para evitarlo, se necesitan las restricciones adicionales de DFJ.
            """)
        
        with st.expander("Haz clic para ver la Formulación Dantzig-Fulkerson-Johnson (DFJ)"):
            st.markdown(r"""
            La formulación propuesta por Dantzig, Fulkerson y Johnson es el pilar de los solvers modernos para el TSP. Su aporte clave es la **Restricción de Eliminación de Sub-rutas (SEC)**, que se añade al modelo general anterior.

            #### Restricción de Eliminación de Sub-Rutas (SEC)
            $$ \sum_{i,j \in Q, i < j} x_{ij} \le |Q|-1 \quad \forall Q \subsetneq V, |Q| \ge 2 $$

            - **Implementación Práctica (Cortes Perezosos):** Hay una cantidad exponencial de posibles sub-rutas. Añadirlas todas desde el inicio es inviable. En su lugar, esta aplicación usa **cortes perezosos**:
                1. Se resuelve el modelo solo con las restricciones de grado.
                2. Si la solución contiene sub-rutas (como se ve en la animación), se identifica cada sub-ruta `Q`.
                3. Se añade una restricción SEC específica para cada sub-ruta encontrada.
                4. Se vuelve a resolver el modelo.
            Este proceso se repite hasta que la solución es un único tour completo.
                        
            #### Propiedades del Método
            - **Complejidad Algorítmica:** El TSP es **NP-duro**. El método de *Branch and Cut* usado en esta app (basado en DFJ) tiene una complejidad en el peor de los casos de $O(n^2 2^n)$, lo que hace que el tiempo de cómputo crezca exponencialmente con el número de ciudades.
            - **Tipo de Óptimo:** Este es un método **exacto**. Si se le da suficiente tiempo, **garantiza encontrar el óptimo global**, es decir, la mejor solución posible sin lugar a dudas.
            - **Garantía de Parada:** El algoritmo de *Branch and Cut* es finito y **siempre converge**. Explora sistemáticamente el árbol de soluciones y "poda" las ramas que no pueden contener la solución óptima. El proceso termina cuando todo el árbol ha sido explorado o la brecha entre la mejor solución encontrada y la mejor cota teórica es cero.
            """)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("Presiona para encontrar la **solución óptima garantizada**.")
            if st.button('**Resolver por DFJ Simplex**', key="solve_dfj"):
                st.session_state.dfj_results = None
                st.session_state.active_route_solution = None
                st.session_state.selected_start_city_index = None
                summary_placeholder_dfj = st.empty()
                plot_placeholder_dfj = st.empty()
                
                summary_placeholder_dfj.info("Iniciando optimización...")
                plot_placeholder_dfj.plotly_chart(draw_map(df_ubicaciones, title="Mapa inicial"), use_container_width=True, key="dfj_running_map")
                
                m = Model()
                m.Params.lazyConstraints = 1
                m.Params.OutputFlag = 0
                m._df_ubicaciones, m._n, m._subtours_found, m._iterations = df_ubicaciones, n_ciudades, 0, 0
                m._summary_placeholder, m._plot_placeholder = summary_placeholder_dfj, plot_placeholder_dfj
                
                vars = m.addVars(dist_dict.keys(), obj=dist_dict, vtype=GRB.BINARY, name='e')
                for i, j in list(vars.keys()): vars[j, i] = vars[i, j]
                m._vars = vars
                m.addConstrs(vars.sum(i, '*') == 2 for i in range(n_ciudades))
                m.optimize(subtour_elimination_callback)

                final_tour_edges = [k for k, v in m.getAttr('x', vars).items() if v > 0.5]
                final_tours = find_subtours(tuplelist(final_tour_edges), n_ciudades)
                
                st.session_state.dfj_results = {"distance": m.objVal, "iterations": m._iterations, "route": final_tours[0], "runtime": m.Runtime}
                st.session_state.active_route_solution = final_tours[0]

        with col2:
            if st.session_state.dfj_results:
                res = st.session_state.dfj_results
                st.success(f"¡Optimización Completada! Distancia Óptima: **{res['distance']:,.2f} km**")
                final_fig = draw_map(df_ubicaciones, route_indices=res['route'], title="Ruta Óptima Final Encontrada por DFJ")
                st.plotly_chart(final_fig, use_container_width=True, key="dfj_final_map")
            else:
                st.plotly_chart(draw_map(df_ubicaciones, title="Mapa de ciudades a optimizar"), use_container_width=True, key="dfj_initial_map")

    with tab2:
        st.header("Aproximación Heurística con Algoritmo Genético")
        st.sidebar.header("🧬 Parámetros del Algoritmo Genético")
        POPULATION_SIZE = st.sidebar.slider("Tamaño Población", 50, 500, 250, 50)
        N_ITERATIONS = st.sidebar.slider("Nº Generaciones", 100, 2000, 500, 100)
        ELITISM_RATE = st.sidebar.slider("Tasa Elitismo", 0.01, 0.5, 0.1, 0.01)
        CROSSOVER_RATE = st.sidebar.slider("Tasa Cruce", 0.5, 1.0, 0.9, 0.05)
        MUTATION_RATE = st.sidebar.slider("Tasa Mutación", 0.01, 0.5, 0.18, 0.01)
        TOURNAMENT_SIZE = st.sidebar.slider("Tamaño Torneo", 2, 20, 7, 1)
        EARLY_STOPPING_PATIENCE = st.sidebar.slider("Paciencia para Parada Temprana", 10, 500, 200, 10)
        
        with st.expander("Haz clic para ver la explicación del Algoritmo Genético"):
            st.markdown(r"""
            Un enfoque alternativo son los **algoritmos genéticos (AG)**, una heurística inspirada en la evolución natural. No garantizan la solución óptima, pero pueden encontrar soluciones de muy alta calidad en tiempos razonables, especialmente para problemas grandes.

            #### Componentes Clave del AG para el TSP:

            1.  **Representación (Cromosoma):**
                Una ruta se representa como una permutación de los índices de las ciudades. Por ejemplo, `[3, 1, 4, 0, 2]` es un "cromosoma" que codifica la ruta `(Ciudad 3 -> Ciudad 1 -> Ciudad 4 -> Ciudad 0 -> Ciudad 2 -> Ciudad 3)`.
                $$ \text{Ruta candidata} = [c_1, c_2, \dots, c_n] $$

            2.  **Función de Aptitud (Fitness):**
                La aptitud de una ruta es simplemente su longitud total. El objetivo es encontrar el cromosoma con el menor valor de fitness.
                $$ \text{Fitness} = \sum_{i=0}^{n-2} \text{dist}(c_i, c_{i+1}) + \text{dist}(c_{n-1}, c_0) $$

            3.  **Selección:**
                Se seleccionan las rutas "más aptas" (más cortas) para que sean "padres" de la siguiente generación. Esta app usa la **selección por torneo**, donde un pequeño grupo de candidatos compite y el mejor (el de menor distancia) es elegido. El `Tamaño del Torneo` controla la presión de selección.

            4.  **Cruce (Crossover):**
                Dos rutas "padre" se combinan para crear una "hija", con la esperanza de que herede las mejores características de ambos. Se usa un operador como el **cruce de orden (OX)**, que preserva la estructura de las rutas para evitar soluciones inválidas. La `Tasa de Cruce` define la probabilidad de que dos padres se crucen.

            5.  **Mutación:**
                Se introducen pequeños cambios aleatorios en las rutas hijas (ej. intercambiar dos ciudades) para mantener la diversidad genética y evitar que el algoritmo se estanque en una solución subóptima. La `Tasa de Mutación` controla la frecuencia de estos cambios.
            
            6.  **Elitismo:**
                Se garantiza que un porcentaje de las mejores rutas (`Tasa de Elitismo`) de una generación pasen directamente a la siguiente sin cambios, preservando las mejores soluciones encontradas hasta el momento.

            #### Propiedades del Método
            - **Complejidad Algorítmica:** La complejidad es polinomial, aproximadamente $O(G \cdot P \cdot N)$, donde $G$ es el número de generaciones, $P$ el tamaño de la población y $N$ el número de ciudades. Esto es significativamente más rápido que la complejidad exponencial del método exacto.
            - **Tipo de Óptimo:** Es un método **heurístico**, por lo que **no garantiza encontrar el óptimo global**. Busca de manera inteligente en el espacio de soluciones y usualmente encuentra **óptimos locales** de muy alta calidad, que a menudo son idénticos o muy cercanos al óptimo global.
            - **Garantía de Parada:** La parada del algoritmo está garantizada y se controla principalmente por dos criterios:
                1.  **Criterio Principal (Límite de Generaciones):** El algoritmo siempre se detendrá después de completar el número de generaciones (`Nº Generaciones`) que el usuario ha especificado en los parámetros.
                2.  **Criterio Secundario (Convergencia por Estancamiento):** Esta implementación incluye una **parada temprana**. El algoritmo monitorea si la mejor solución ha dejado de mejorar. Si no se encuentra una ruta mejor después de un número de generaciones consecutivas (definido por el parámetro `Paciencia para Parada Temprana`), se asume que ha convergido a una solución estable y se detiene. Esto hace que el proceso sea más eficiente al no gastar tiempo en búsquedas que ya no rinden fruto.
                        
            """)



        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("Presiona para encontrar una **solución de alta calidad**.")
            if st.button('**Resolver con Método Genético**', key="solve_ga"):
                st.session_state.ga_results = None
                st.session_state.active_route_solution = None
                st.session_state.selected_start_city_index = None
                summary_placeholder_ga = st.empty()
                plot_placeholder_ga = st.empty()
                
                summary_placeholder_ga.info("Iniciando evolución...")
                plot_placeholder_ga.plotly_chart(draw_map(df_ubicaciones, title="Mapa inicial"), use_container_width=True, key="ga_running_map")

                start_time = time.time()
                population = Population(POPULATION_SIZE, df_distancias_matrix)
                best_overall_score, best_iteration, best_route = float('inf'), 0, None
                generations_without_improvement = 0

                for i in range(N_ITERATIONS):
                    population.evolve(ELITISM_RATE, CROSSOVER_RATE, MUTATION_RATE, TOURNAMENT_SIZE)
                    current_best_candidate = population.candidates[0]
                    
                    new_best_found = current_best_candidate.fitness_score < best_overall_score
                    if new_best_found:
                        best_overall_score = current_best_candidate.fitness_score
                        best_iteration, best_route = i + 1, current_best_candidate.chromosomes
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                    
                    if new_best_found or (i + 1) % 25 == 0 or i == 0 or (i + 1) == N_ITERATIONS:
                        summary_placeholder_ga.info(f"Generación {i+1}/{N_ITERATIONS}\n\nMejor Distancia: {best_overall_score:,.2f} km")
                        fig = draw_map(df_ubicaciones, route_indices=current_best_candidate.chromosomes, title=f"AG - Mejor Ruta en Generación {i+1}")
                        plot_placeholder_ga.plotly_chart(fig, use_container_width=True)
                        time.sleep(1.8)
                    
                    if generations_without_improvement >= EARLY_STOPPING_PATIENCE:
                        st.toast(f"Parada temprana en generación {i+1} por estancamiento de la solución.")
                        break
                
                end_time = time.time()
                st.session_state.ga_results = {"distance": best_overall_score, "iterations": best_iteration, "route": best_route, "runtime": end_time - start_time}
                st.session_state.active_route_solution = best_route.tolist()

        with col4:
            if st.session_state.ga_results:
                res = st.session_state.ga_results
                st.success(f"¡Optimización Completada! Mejor Distancia: **{res['distance']:,.2f} km**")
                final_fig = draw_map(df_ubicaciones, route_indices=res['route'], title="Mejor Ruta Final Encontrada por AG")
                st.plotly_chart(final_fig, use_container_width=True, key="ga_final_map")
            else:
                st.plotly_chart(draw_map(df_ubicaciones, title="Mapa de ciudades a optimizar"), use_container_width=True, key="ga_initial_map")

    # --- SECCIÓN DE RESULTADOS Y COMPARACIÓN ---
    st.markdown("---")
    st.header("🏆 Resultados y Comparación")
    if not st.session_state.dfj_results and not st.session_state.ga_results:
        st.info("Ejecuta uno o ambos algoritmos para ver los resultados aquí.")
    else:
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Método Exacto (DFJ)")
            if st.session_state.dfj_results:
                dfj_res = st.session_state.dfj_results
                st.metric("Distancia Óptima", f"{dfj_res['distance']:,.2f} km")
                st.metric("Iteraciones (Callbacks)", f"{dfj_res['iterations']}")
                st.metric("Tiempo de Cómputo (Gurobi)", f"{dfj_res['runtime']:.2f} s")
            else:
                st.warning("Aún no se ha ejecutado.")
        
        with res_col2:
            st.subheader("Algoritmo Genético")
            if st.session_state.ga_results:
                ga_res = st.session_state.ga_results
                st.metric("Mejor Distancia Encontrada", f"{ga_res['distance']:,.2f} km")
                st.metric("Generación Óptima", f"{ga_res['iterations']}")
                st.metric("Tiempo de Cómputo (Python)", f"{ga_res['runtime']:.2f} s")
            else:
                st.warning("Aún no se ha ejecutado.")
        
        if st.session_state.dfj_results and st.session_state.ga_results:
            st.markdown("---")
            st.subheader("Análisis Comparativo")
            dfj_res = st.session_state.dfj_results
            ga_res = st.session_state.ga_results
            gap = ((ga_res['distance'] - dfj_res['distance']) / dfj_res['distance']) * 100
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            comp_col1.metric(label="Diferencia de Distancia (Gap)", value=f"{gap:.2f}%", help="Qué tan lejos está la solución del AG de la óptima.")
            comp_col2.metric(label="Diferencia de Iteraciones", value=f"{ga_res['iterations'] - dfj_res['iterations']}", help="Generaciones (AG) vs Callbacks (DFJ).")
            comp_col3.metric(label="Diferencia de Tiempo de Cómputo", value=f"{ga_res['runtime'] - dfj_res['runtime']:.2f} s", help="Diferencia en segundos del procesamiento.")

    # --- NUEVA SECCIÓN: EXPLORACIÓN INTERACTIVA DE LA RUTA ---
    if st.session_state.active_route_solution:
        st.markdown("---")
        st.header("🗺️ Explora la Ruta Final")
        st.write("Selecciona una ciudad de inicio para ver la ruta detallada y su animación en el mapa.")

        cols_per_row = 8
        num_rows = math.ceil(n_ciudades / cols_per_row)
        city_index = 0
        for _ in range(num_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                if city_index < n_ciudades:
                    if cols[i].button(cities_names[city_index], key=f"city_{city_index}"):
                        st.session_state.selected_start_city_index = city_index
                    city_index += 1
        
        if st.session_state.selected_start_city_index is not None:
            start_node_index = st.session_state.selected_start_city_index
            optimal_sequence = st.session_state.active_route_solution
            
            start_pos = optimal_sequence.index(start_node_index)
            reordered_forward = optimal_sequence[start_pos:] + optimal_sequence[:start_pos]
            
            reversed_sequence = optimal_sequence[::-1]
            start_pos_rev = reversed_sequence.index(start_node_index)
            reordered_backward = reversed_sequence[start_pos_rev:] + reversed_sequence[:start_pos_rev]

            st.markdown("---")
            vis_col1, vis_col2 = st.columns([1, 2])
            with vis_col1:
                st.markdown("#### Visualizar Recorrido")
                route_to_show = st.radio("Ruta a visualizar:", ("Sentido Óptimo", "Sentido Inverso"), key="route_selector", horizontal=True)
                
                highlight_sequence = reordered_forward if route_to_show == "Sentido Óptimo" else reordered_backward
                route_names = [cities_names[i] for i in highlight_sequence] + [cities_names[highlight_sequence[0]]]
                st.info(" ➔ ".join(route_names))

            with vis_col2:
                route_map_placeholder = st.empty()
                fig_anim = draw_map(df_ubicaciones, title=f"Animación desde {cities_names[start_node_index]}")
                route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.5)

                route_with_end = highlight_sequence + [highlight_sequence[0]]
                for i in range(len(route_with_end) - 1):
                    segment = [route_with_end[i], route_with_end[i+1]]
                    segment_points = df_ubicaciones.iloc[segment]
                    fig_anim.add_trace(go.Scattermapbox(lat=segment_points['Latitud'], lon=segment_points['Longitud'], mode='lines', line={'width': 4, 'color': 'crimson'}))
                    route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
                    time.sleep(1.5)

elif df_ubicaciones_full is None:
    st.error("La aplicación no puede iniciar porque faltan los archivos de datos.")

