import streamlit as st
import pandas as pd
import itertools
from gurobipy import Model, GRB, quicksum, tuplelist
import plotly.graph_objects as go
import time
import math

st.set_page_config(layout="wide")
st.title('TSP: Ruta Turística Óptima por Colombia')

@st.cache_data
def load_data():
    try:
        df_ubicaciones = pd.read_csv('ubicacion.csv')
        df_distancias = pd.read_csv('distancias.csv', index_col='Ciudad')
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}.")
        return None, None
    return df_ubicaciones, df_distancias

def subtour(edges, num_cities):
    unvisited = list(range(num_cities))
    cycles = []
    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        cycles.append(thiscycle)
    return cycles

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        model._iterations += 1
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        
        tours = subtour(selected, model._n)
        if len(tours) > 1:
            model._subtours += 1
            for tour in tours:
                model.cbLazy(quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2)) <= len(tour) - 1)
        
        current_length = round(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        model._summary.markdown(f"""
        - **Iteración:** `{model._iterations}`
        - **Distancia Actual:** `{current_length:,} km`
        - **Sub-rutas Encontradas:** `{len(tours)}`
        - **Restricciones Añadidas:** `{model._subtours}`
        """)
        
        fig = draw_map(model._df_ubicaciones, tours=tours)
        model._plot.plotly_chart(fig, use_container_width=True)
        time.sleep(4)

def draw_map(df_ubicaciones, tours=None, highlighted_tour=None):
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=df_ubicaciones['Latitud'],
        lon=df_ubicaciones['Longitud'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=10, color='darkblue'),
        text=df_ubicaciones['Capital'],
    ))

    if tours:
        for i, tour in enumerate(tours):
            tour_copy = tour[:]
            tour_copy.append(tour_copy[0])
            tour_points = df_ubicaciones.iloc[tour_copy]
            fig.add_trace(go.Scattermapbox(lat=tour_points['Latitud'], lon=tour_points['Longitud'], mode='lines', line=go.scattermapbox.Line(width=2, color='gray'), name=f"Sub-ruta {i+1}"))

    if highlighted_tour:
        tour_copy = highlighted_tour[:]
        tour_copy.append(tour_copy[0])
        tour_points = df_ubicaciones.iloc[tour_copy]
        fig.add_trace(go.Scattermapbox(lat=tour_points['Latitud'], lon=tour_points['Longitud'], mode='lines', line=go.scattermapbox.Line(width=4, color='crimson'), name="Ruta Seleccionada"))

    fig.update_layout(showlegend=False, mapbox_style="carto-positron", mapbox_zoom=4.5, mapbox_center={"lat": 4.5709, "lon": -74.2973}, margin={"r":0,"t":0,"l":0,"b":0})
    return fig

df_ubicaciones_full, df_distancias_full = load_data()

if 'optimal_sequence' not in st.session_state:
    st.session_state.optimal_sequence = None
if 'selected_start_city_index' not in st.session_state:
    st.session_state.selected_start_city_index = None
if 'last_run_n' not in st.session_state:
    st.session_state.last_run_n = 0

if df_ubicaciones_full is not None:
    st.markdown("### 1. Define tu Top de Ciudades Turísticas")
    n_ciudades = st.slider("Selecciona el Top N de ciudades más visitadas para calcular la ruta óptima:", min_value=4, max_value=32, value=32, key="n_slider")

    df_ubicaciones = df_ubicaciones_full.head(n_ciudades)
    df_distancias = df_distancias_full.iloc[:n_ciudades, :n_ciudades]
    cities = df_ubicaciones['Capital'].tolist()
    dist = {(i, j): df_distancias.iloc[i, j] for i in range(n_ciudades) for j in range(i)}

    st.markdown("---")
    st.markdown(f"### 2. Optimiza la Ruta para el Top {n_ciudades}")
    st.subheader("Planteamientos del Problema del Vendedor Viajero (TSP)")
    st.info("""
    A continuación se presentan diferentes formas de abordar el TSP. Esta aplicación implementa la formulación **Dantzig-Fulkerson-Johnson (DFJ)**, que es un método de optimización matemática exacta.
    """)
    
    # --- ORDEN DE DESPLEGABLES SOLICITADO ---

    # 1. Planteamiento General de Programación Lineal
    with st.expander("Haz clic para ver el Planteamiento del TSP como Programación Lineal"):
        st.markdown(r"""
        El Problema del Vendedor Viajero (TSP) puede formularse como un problema de **programación lineal entera**. El objetivo es seleccionar un conjunto de arcos (caminos entre ciudades) que formen un ciclo único (un "tour") que visite cada ciudad exactamente una vez, minimizando la distancia total.

        #### Elementos Fundamentales
        - **Conjunto de Nodos (Ciudades):** $V = \{1, 2, \dots, n\}$
        - **Parámetro de Costo/Distancia:** $c_{ij}$, la distancia entre la ciudad $i$ y la ciudad $j$.
        - **Variable de Decisión (Binaria):**
            $$
            x_{ij} = \begin{cases}
            1 & \text{si la ruta va directamente de la ciudad } i \text{ a la } j \\
            0 & \text{en caso contrario}
            \end{cases}
            $$

        #### Función Objetivo
        Minimizar la suma de las distancias de todos los arcos seleccionados:
        $$ \min \sum_{i \in V} \sum_{j \in V, i \neq j} c_{ij} x_{ij} $$

        #### Restricciones Fundamentales (De Grado)
        Estas restricciones aseguran que cada ciudad sea parte de un camino:
        1.  **Salir de cada ciudad una vez:** Para cada ciudad $i$, se debe tomar exactamente un arco que salga de ella.
            $$ \sum_{j \in V, j \neq i} x_{ij} = 1 \quad \forall i \in V $$
        2.  **Entrar a cada ciudad una vez:** Para cada ciudad $j$, se debe tomar exactamente un arco que llegue a ella.
            $$ \sum_{i \in V, i \neq j} x_{ij} = 1 \quad \forall j \in V $$

        El principal desafío es que estas restricciones por sí solas no impiden la formación de **sub-rutas** (ciclos desconectados más pequeños). Para evitarlo, se necesitan restricciones adicionales, como las propuestas por DFJ.
        """)
    
    # 2. Formulación DFJ
    with st.expander("Haz clic para ver la Formulación Dantzig-Fulkerson-Johnson (DFJ)"):
        st.markdown(r"""
        La formulación propuesta por Dantzig, Fulkerson y Johnson es el pilar de los solvers modernos para el TSP. Su aporte clave es la **Restricción de Eliminación de Sub-rutas (SEC)**.

        #### Complejidad Algorítmica
        El TSP es **NP-duro**. El método de *Branch and Cut* usado en esta app (basado en DFJ) tiene una complejidad en el peor de los casos de $O(n^2 2^n)$, lo que hace que el tiempo de cómputo crezca exponencialmente.

        #### Restricción de Eliminación de Sub-Rutas (SEC)
        Además de las restricciones de grado, se añade:
        $$ \sum_{i \in Q, j \in Q, i \neq j} x_{ij} \le |Q|-1 \quad \forall Q \subsetneq V, |Q| \ge 2 $$

        - **Explicación:** Esta inecuación es el corazón de la formulación DFJ. Impone que para cualquier subconjunto propio de ciudades `Q`, no se puede formar un ciclo cerrado internamente. Por ejemplo, si un subconjunto tiene 3 ciudades, no pueden existir 3 arcos entre ellas. Esto obliga al modelo a generar una única ruta que conecte todos los nodos.
        - **Implementación Práctica:** En lugar de añadir las $2^n - 2$ posibles restricciones SEC desde el inicio, se añaden dinámicamente como **"cortes perezosos"**. El solver encuentra una solución con sub-rutas, y solo entonces se añade la restricción específica que prohíbe ese ciclo, repitiendo el proceso hasta encontrar el tour único.
        """)

    # 3. Planteamiento Bioinspirado
    with st.expander("Haz clic para ver el Planteamiento Bioinspirado (Algoritmos Genéticos)"):
        st.markdown(r"""
        Un enfoque alternativo son los **algoritmos genéticos (AG)**, una heurística inspirada en la evolución natural. No garantizan la solución óptima, pero pueden encontrar soluciones de muy alta calidad en tiempos razonables, especialmente para problemas grandes.

        #### Componentes Clave del AG para el TSP:

        1.  **Representación (Cromosoma):**
            Una ruta se representa como una permutación de las ciudades. Por ejemplo, `[3, 1, 4, 0, 2]` es un "cromosoma" que codifica una ruta.
            $$ \text{Ruta} = [c_1, c_2, \dots, c_n] $$

        2.  **Función de Aptitud (Fitness):**
            La aptitud de una ruta es inversamente proporcional a su longitud total. El objetivo es encontrar el cromosoma con la menor distancia.
            $$ \text{Fitness} = \sum_{i=0}^{n-2} \text{dist}(c_i, c_{i+1}) + \text{dist}(c_{n-1}, c_0) $$

        3.  **Selección:**
            Se seleccionan las rutas "más aptas" (más cortas) para que sean "padres" de la siguiente generación. Un método común es la **selección por torneo**.

        4.  **Cruce (Crossover):**
            Dos rutas "padre" se combinan para crear una "hija". Para el TSP, se usan operadores como el **cruce de orden (OX)**, que preserva la estructura de las rutas para evitar soluciones inválidas (ciudades repetidas).

        5.  **Mutación:**
            Se introducen pequeños cambios aleatorios en las rutas hijas (ej. intercambiar dos ciudades) para mantener la diversidad genética y explorar nuevas áreas del espacio de soluciones.
        """)

    summary_placeholder = st.empty()
    plot_placeholder = st.empty()

    if n_ciudades != st.session_state.last_run_n:
        st.session_state.optimal_sequence = None
        st.session_state.selected_start_city_index = None

    if not st.session_state.optimal_sequence:
        plot_placeholder.plotly_chart(draw_map(df_ubicaciones), use_container_width=True, key="initial_map")

    if st.button('**Resolver por DFJ Simplex**'):
        st.session_state.last_run_n = n_ciudades
        st.session_state.optimal_sequence = None
        st.session_state.selected_start_city_index = None
        summary_placeholder.info("Calculando la ruta óptima...")
        
        m = Model()
        m.Params.lazyConstraints = 1
        m._df_ubicaciones, m._n, m._subtours, m._iterations = df_ubicaciones, n_ciudades, 0, 0
        m._summary, m._plot = summary_placeholder, plot_placeholder
        
        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
        for i, j in list(vars.keys()):
            vars[j, i] = vars[i, j]
        m._vars = vars
        m.addConstrs(vars.sum(i, '*') == 2 for i in range(n_ciudades))
        m.optimize(subtourelim)

        final_tour_edges = [k for k, v in m.getAttr('x', vars).items() if v > 0.5]
        final_tours = subtour(tuplelist(final_tour_edges), n_ciudades)
        st.session_state.optimal_sequence = final_tours[0]
        
        summary_placeholder.success("¡Optimización Completada!")
        st.write(f"**Distancia Óptima Total:** `{round(m.objVal):,} km`")
        st.write(f"**Tiempo de Ejecución:** `{m.Runtime:.2f} segundos`")
        st.write(f"**Total de Iteraciones (Callbacks):** `{m._iterations}`")
        
        final_fig = draw_map(df_ubicaciones, highlighted_tour=st.session_state.optimal_sequence)
        plot_placeholder.plotly_chart(final_fig, use_container_width=True, key="final_static_map")

    if st.session_state.optimal_sequence:
        st.markdown("---")
        st.markdown("### 3. Explora la Ruta Óptima")
        st.write("Selecciona una ciudad de inicio para ver la ruta detallada y su animación en el mapa.")

        cols_per_row = 8
        num_rows = math.ceil(n_ciudades / cols_per_row)
        city_index = 0
        for _ in range(num_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                if city_index < n_ciudades:
                    if cols[i].button(cities[city_index], key=f"city_{city_index}"):
                        st.session_state.selected_start_city_index = city_index
                    city_index += 1
        
        if st.session_state.selected_start_city_index is not None and st.session_state.selected_start_city_index < n_ciudades:
            start_node_index = st.session_state.selected_start_city_index
            optimal_sequence = st.session_state.optimal_sequence
            
            start_pos = optimal_sequence.index(start_node_index)
            reordered_forward = optimal_sequence[start_pos:] + optimal_sequence[:start_pos]
            
            reversed_sequence = optimal_sequence[::-1]
            start_pos_rev = reversed_sequence.index(start_node_index)
            reordered_backward = reversed_sequence[start_pos_rev:] + reversed_sequence[:start_pos_rev]

            cols_vis = st.columns([1, 1])
            with cols_vis[0]:
                st.markdown("#### Visualizar Recorrido")
                route_to_show = st.radio("Ruta a visualizar:", ("Ruta Óptima", "Ruta Inversa"), key="route_selector", horizontal=True)

            route_map_placeholder = st.empty()
            highlight_sequence = reordered_forward if route_to_show == "Ruta Óptima" else reordered_backward
            
            fig_anim = draw_map(df_ubicaciones)
            route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.5)

            route_with_end = highlight_sequence + [highlight_sequence[0]]
            for i in range(len(route_with_end) - 1):
                segment = [route_with_end[i], route_with_end[i+1]]
                segment_points = df_ubicaciones.iloc[segment]
                fig_anim.add_trace(go.Scattermapbox(lat=segment_points['Latitud'], lon=segment_points['Longitud'], mode='lines', line=go.scattermapbox.Line(width=4, color='crimson')))
                route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(2)
            
            with cols_vis[1]:
                route1_names = [cities[i] for i in reordered_forward] + [cities[start_node_index]]
                route2_names = [cities[i] for i in reordered_backward] + [cities[start_node_index]]

                st.markdown(f"#### Rutas desde **{cities[start_node_index]}**")
                st.markdown("**Ruta 1 (Sentido Óptimo):**")
                st.info(" ➔ ".join(route1_names))
                st.markdown("**Ruta 2 (Sentido Contrario):**")
                st.info(" ➔ ".join(route2_names))