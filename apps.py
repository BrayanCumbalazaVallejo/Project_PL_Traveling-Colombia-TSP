# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para resolver y comparar soluciones al Problema del Vendedor Viajero (TSP)
en Colombia, utilizando un método exacto (Dantzig-Fulkerson-Johnson con Gurobi) y un
método heurístico (Algoritmo Genético).
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

Usa el control deslizante para elegir el número de ciudades y luego explora cada pestaña para ejecutar los algoritmos y comparar sus resultados.
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
        
        fig = draw_map(model._df_ubicaciones, tours=tours, title=f"DFJ - Iteración {model._iterations}: Encontrando y eliminando sub-rutas")
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
            # Elige entre dos tipos de mutación: intercambio o inversión
            if np.random.rand() < 0.5: # Swap mutation
                idx1, idx2 = np.random.choice(len(self.chromosomes), 2, replace=False)
                self.chromosomes[idx1], self.chromosomes[idx2] = self.chromosomes[idx2], self.chromosomes[idx1]
            else: # Inversion mutation
                start, end = np.sort(np.random.choice(len(self.chromosomes), 2, replace=False))
                self.chromosomes[start:end+1] = self.chromosomes[start:end+1][::-1]
            self._fitness = None # Recalcular fitness después de la mutación

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
    
    # Dibuja las ciudades
    fig.add_trace(go.Scattermapbox(
        lat=df_ubicaciones['Latitud'],
        lon=df_ubicaciones['Longitud'],
        mode='markers+text',
        marker={'size': 10, 'color': 'darkblue'},
        text=df_ubicaciones['Capital'],
        textposition='top right'
    ))

    # Dibuja sub-rutas (para DFJ)
    if tours:
        for i, tour in enumerate(tours):
            if len(tour) < len(df_ubicaciones): # Solo dibuja sub-rutas reales
                tour_copy = tour[:]
                tour_copy.append(tour_copy[0])
                tour_points = df_ubicaciones.iloc[tour_copy]
                fig.add_trace(go.Scattermapbox(lat=tour_points['Latitud'], lon=tour_points['Longitud'], mode='lines', line={'width': 2, 'color': 'gray'}, name=f"Sub-ruta {i+1}"))

    # Dibuja la ruta principal (para ambos algoritmos)
    if route_indices is not None:
        route_points = df_ubicaciones.iloc[np.append(route_indices, route_indices[0])]
        fig.add_trace(go.Scattermapbox(
            lat=route_points['Latitud'],
            lon=route_points['Longitud'],
            mode='lines',
            line={'width': 3, 'color': 'crimson'},
            name="Ruta Principal"
        ))

    fig.update_layout(
        title=title,
        showlegend=False,
        mapbox_style="carto-positron",
        mapbox_zoom=4.2,
        mapbox_center={"lat": 4.5709, "lon": -74.2973},
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

# --- INICIALIZACIÓN DE LA APP ---
df_ubicaciones_full, df_distancias_full = load_data()

# Inicializar session_state
if 'dfj_results' not in st.session_state:
    st.session_state.dfj_results = None
if 'ga_results' not in st.session_state:
    st.session_state.ga_results = None
if 'last_run_n' not in st.session_state:
    st.session_state.last_run_n = 0

# --- INTERFAZ DE USUARIO ---
if df_ubicaciones_full is not None:
    
    # --- PANEL DE CONFIGURACIÓN ---
    st.sidebar.header("⚙️ Configuración General")
    n_ciudades = st.sidebar.slider(
        "Selecciona el número de ciudades a visitar:",
        min_value=4, max_value=len(df_ubicaciones_full), value=10, key="n_slider"
    )

    # Resetear resultados si el número de ciudades cambia
    if n_ciudades != st.session_state.last_run_n:
        st.session_state.dfj_results = None
        st.session_state.ga_results = None
        st.session_state.last_run_n = n_ciudades
    
    # Filtrar datos según la selección
    df_ubicaciones = df_ubicaciones_full.head(n_ciudades)
    df_distancias_matrix = df_distancias_full.iloc[:n_ciudades, :n_ciudades].to_numpy()
    dist_dict = {
        (i, j): df_distancias_matrix[i, j] 
        for i in range(n_ciudades) for j in range(i)
    }

    # --- PESTAÑAS PARA CADA MÉTODO ---
    tab1, tab2 = st.tabs(["**Método Exacto (Dantzig-Fulkerson-Johnson)**", "**Método Heurístico (Algoritmo Genético)**"])

    with tab1:
        st.header("Optimización Matemática con Gurobi")
        
        with st.expander("Haz clic para ver la explicación del Planteamiento DFJ"):
            st.markdown(r"""
            La formulación propuesta por Dantzig, Fulkerson y Johnson (DFJ) es el pilar de los solvers modernos para el TSP. Su aporte clave es la **Restricción de Eliminación de Sub-rutas (SEC)**.

            #### Elementos Fundamentales
            - **Variable de Decisión (Binaria):** $x_{ij} = 1$ si la ruta va de la ciudad $i$ a la $j$, y $0$ si no.
            - **Función Objetivo:** Minimizar la distancia total: $\min \sum c_{ij} x_{ij}$
            - **Restricciones de Grado:** Aseguran que cada ciudad se visita exactamente una vez.
            - **Restricción Clave (SEC):** $\sum_{i \in Q, j \in Q} x_{ij} \le |Q|-1$ para cualquier subconjunto de ciudades $Q$. Esta restricción impide la formación de ciclos cortos (sub-rutas).

            En esta aplicación, las restricciones SEC se añaden dinámicamente como **"cortes perezosos"**. El solver encuentra una solución con sub-rutas, y solo entonces se añade la restricción que prohíbe ese ciclo, repitiendo el proceso hasta encontrar el tour único y óptimo.
            """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("Presiona el botón para encontrar la **solución óptima garantizada**.")
            if st.button('**Resolver por DFJ Simplex**', key="solve_dfj"):
                st.session_state.dfj_results = None
                summary_placeholder_dfj = st.empty()
                plot_placeholder_dfj = st.empty()
                
                summary_placeholder_dfj.info("Iniciando el proceso de optimización...")
                plot_placeholder_dfj.plotly_chart(draw_map(df_ubicaciones, title="Mapa inicial de ciudades"), use_container_width=True, key="dfj_running_map")
                
                m = Model()
                m.Params.lazyConstraints = 1
                m.Params.OutputFlag = 0 # Silenciar logs de Gurobi en la consola
                
                # Asignar variables y placeholders al modelo para el callback
                m._df_ubicaciones = df_ubicaciones
                m._n = n_ciudades
                m._subtours_found = 0
                m._iterations = 0
                m._summary_placeholder = summary_placeholder_dfj
                m._plot_placeholder = plot_placeholder_dfj
                
                vars = m.addVars(dist_dict.keys(), obj=dist_dict, vtype=GRB.BINARY, name='e')
                for i, j in list(vars.keys()):
                    vars[j, i] = vars[i, j]
                m._vars = vars
                
                m.addConstrs(vars.sum(i, '*') == 2 for i in range(n_ciudades))
                
                m.optimize(subtour_elimination_callback)

                final_tour_edges = [k for k, v in m.getAttr('x', vars).items() if v > 0.5]
                final_tours = find_subtours(tuplelist(final_tour_edges), n_ciudades)
                
                st.session_state.dfj_results = {
                    "distance": m.objVal,
                    "iterations": m._iterations,
                    "route": final_tours[0],
                    "runtime": m.Runtime
                }

        with col2:
            if st.session_state.dfj_results:
                res = st.session_state.dfj_results
                st.success(f"¡Optimización Completada! Distancia Óptima: **{res['distance']:,.2f} km**")
                final_fig = draw_map(df_ubicaciones, route_indices=res['route'], title="Ruta Óptima Final Encontrada por DFJ")
                st.plotly_chart(final_fig, use_container_width=True, key="dfj_final_map")
            else:
                # Placeholder para el mapa antes de la ejecución
                st.plotly_chart(draw_map(df_ubicaciones, title="Mapa de ciudades a optimizar"), use_container_width=True, key="dfj_initial_map")

    with tab2:
        st.header("Aproximación Heurística con Algoritmo Genético")
        
        # --- PARÁMETROS DEL AG EN LA SIDEBAR ---
        st.sidebar.header("🧬 Parámetros del Algoritmo Genético")
        POPULATION_SIZE = st.sidebar.slider("Tamaño de la Población", 50, 500, 250, 50)
        N_ITERATIONS = st.sidebar.slider("Número de Generaciones", 100, 2000, 500, 100)
        ELITISM_RATE = st.sidebar.slider("Tasa de Elitismo", 0.01, 0.5, 0.1, 0.01)
        CROSSOVER_RATE = st.sidebar.slider("Tasa de Cruce", 0.5, 1.0, 0.9, 0.05)
        MUTATION_RATE = st.sidebar.slider("Tasa de Mutación", 0.01, 0.5, 0.18, 0.01)
        TOURNAMENT_SIZE = st.sidebar.slider("Tamaño del Torneo", 2, 20, 7, 1)
        
        with st.expander("Haz clic para ver la explicación del Algoritmo Genético"):
            st.markdown("""
            Un **algoritmo genético (AG)** es una heurística que imita la evolución natural para encontrar soluciones de alta calidad. No garantiza la optimalidad, pero es muy eficiente.

            - **Candidato (Cromosoma):** Una ruta específica, representada como una secuencia ordenada de ciudades.
            - **Población:** Un conjunto de diversas rutas que "evolucionan" en cada generación.
            - **Función de Fitness:** La distancia total de la ruta. Un valor más bajo es mejor.
            - **Selección por Torneo:** Se elige un subconjunto aleatorio de la población y el mejor de ese subconjunto es seleccionado como "padre".
            - **Cruce (Crossover):** Combina dos rutas "padre" para crear una "hija", heredando buenas características.
            - **Mutación:** Introduce pequeños cambios aleatorios en una ruta para mantener la diversidad y evitar estancarse.
            - **Elitismo:** Asegura que las mejores rutas de una generación pasen directamente a la siguiente.
            """)
        
        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown("Presiona el botón para encontrar una **solución de alta calidad**.")
            if st.button('**Resolver con Método Genético**', key="solve_ga"):
                st.session_state.ga_results = None
                summary_placeholder_ga = st.empty()
                plot_placeholder_ga = st.empty()
                
                summary_placeholder_ga.info("Iniciando la evolución de rutas...")
                plot_placeholder_ga.plotly_chart(draw_map(df_ubicaciones, title="Mapa inicial de ciudades"), use_container_width=True, key="ga_running_map")

                start_time = time.time()
                population = Population(POPULATION_SIZE, df_distancias_matrix)
                best_overall_score = float('inf')
                best_iteration = 0
                best_route = None

                for i in range(N_ITERATIONS):
                    population.evolve(ELITISM_RATE, CROSSOVER_RATE, MUTATION_RATE, TOURNAMENT_SIZE)
                    current_best_candidate = population.candidates[0]
                    
                    new_best_found = current_best_candidate.fitness_score < best_overall_score
                    if new_best_found:
                        best_overall_score = current_best_candidate.fitness_score
                        best_iteration = i + 1
                        best_route = current_best_candidate.chromosomes
                    
                    # Actualizar UI en la primera, última, y cada 25 generaciones, o si se encuentra un nuevo mejor
                    if new_best_found or (i + 1) % 25 == 0 or i == 0 or (i + 1) == N_ITERATIONS:
                        summary_placeholder_ga.info(f"Generación {i+1}/{N_ITERATIONS}\n\nMejor Distancia: {best_overall_score:,.2f} km")
                        fig = draw_map(df_ubicaciones, route_indices=current_best_candidate.chromosomes, title=f"AG - Mejor Ruta en Generación {i+1}")
                        plot_placeholder_ga.plotly_chart(fig, use_container_width=True)
                        if new_best_found:
                            time.sleep(1.8) # Pausa para visualización
                
                end_time = time.time()
                
                st.session_state.ga_results = {
                    "distance": best_overall_score,
                    "iterations": best_iteration,
                    "route": best_route,
                    "runtime": end_time - start_time
                }
        
        with col4:
            if st.session_state.ga_results:
                res = st.session_state.ga_results
                st.success(f"¡Optimización Completada! Mejor Distancia Encontrada: **{res['distance']:,.2f} km**")
                final_fig = draw_map(df_ubicaciones, route_indices=res['route'], title="Mejor Ruta Final Encontrada por AG")
                st.plotly_chart(final_fig, use_container_width=True, key="ga_final_map")
            else:
                st.plotly_chart(draw_map(df_ubicaciones, title="Mapa de ciudades a optimizar"), use_container_width=True, key="ga_initial_map")

    # --- SECCIÓN DE RESULTADOS Y COMPARACIÓN ---
    st.markdown("---")
    st.header("🏆 Resultados y Comparación")
    
    dfj_res = st.session_state.dfj_results
    ga_res = st.session_state.ga_results
    
    if not dfj_res and not ga_res:
        st.info("Ejecuta uno o ambos algoritmos para ver los resultados aquí.")
    else:
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Método Exacto (DFJ)")
            if dfj_res:
                st.metric("Distancia Óptima", f"{dfj_res['distance']:,.2f} km")
                st.metric("Iteraciones (Callbacks)", f"{dfj_res['iterations']}")
                st.metric("Tiempo de Cómputo (Gurobi)", f"{dfj_res['runtime']:.2f} s")
            else:
                st.warning("Aún no se ha ejecutado.")
        
        with res_col2:
            st.subheader("Algoritmo Genético")
            if ga_res:
                st.metric("Mejor Distancia Encontrada", f"{ga_res['distance']:,.2f} km")
                st.metric("Generación Óptima", f"{ga_res['iterations']}")
                st.metric("Tiempo de Cómputo (Python)", f"{ga_res['runtime']:.2f} s")
            else:
                st.warning("Aún no se ha ejecutado.")
        
        if dfj_res and ga_res:
            st.markdown("---")
            st.subheader("Análisis Comparativo")
            
            gap = ((ga_res['distance'] - dfj_res['distance']) / dfj_res['distance']) * 100
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            comp_col1.metric(
                label="Diferencia de Distancia (Gap)",
                value=f"{gap:.2f}%",
                help="Qué tan lejos (en porcentaje) está la solución del AG de la solución óptima real. Un valor más bajo es mejor."
            )
            comp_col2.metric(
                label="Diferencia de Iteraciones",
                value=f"{ga_res['iterations'] - dfj_res['iterations']}",
                help="Generaciones (AG) vs Callbacks (DFJ). No es una comparación directa, pero da una idea de la 'complejidad' del proceso."
            )
            comp_col3.metric(
                label="Diferencia de Tiempo de Cómputo",
                value=f"{ga_res['runtime'] - dfj_res['runtime']:.2f} s",
                help="Diferencia en segundos del tiempo real de procesamiento del algoritmo (sin contar las pausas de visualización)."
            )

            st.info(f"""
            La solución del Algoritmo Genético estuvo a un **{gap:.2f}%** de la distancia óptima.
            Esto demuestra la capacidad de la heurística para encontrar soluciones casi perfectas de manera muy eficiente.
            """)

elif df_ubicaciones_full is None:
    st.error("La aplicación no puede iniciar porque faltan los archivos de datos.")
