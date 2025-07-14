import streamlit as st
import pandas as pd
import itertools
from gurobipy import Model, GRB, quicksum, tuplelist
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")
st.title('TSP: Ruta Óptima por las Capitales de Colombia')

@st.cache_data
def load_data():
    try:
        df_ubicaciones = pd.read_csv('ubicacion.csv')
        df_distancias = pd.read_csv('distancias.csv', index_col='Ciudad')
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}.")
        return None, None, None, None

    dist = {
        (i, j): df_distancias.iloc[i, j]
        for i in range(len(df_distancias))
        for j in range(i)
    }
    
    cities = df_ubicaciones['Capital'].tolist()
    return df_ubicaciones, df_distancias, dist, cities

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
            fig.add_trace(go.Scattermapbox(
                lat=tour_points['Latitud'],
                lon=tour_points['Longitud'],
                mode='lines',
                line=go.scattermapbox.Line(width=2, color='gray'),
                name=f"Sub-ruta {i+1}"
            ))

    if highlighted_tour:
        tour_copy = highlighted_tour[:]
        tour_copy.append(tour_copy[0])
        tour_points = df_ubicaciones.iloc[tour_copy]
        fig.add_trace(go.Scattermapbox(
            lat=tour_points['Latitud'],
            lon=tour_points['Longitud'],
            mode='lines',
            line=go.scattermapbox.Line(width=4, color='crimson'),
            name="Ruta Seleccionada"
        ))

    fig.update_layout(
        showlegend=False,
        mapbox_style="carto-positron",
        mapbox_zoom=4.5,
        mapbox_center={"lat": 4.5709, "lon": -74.2973},
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig

df_ubicaciones, df_distancias, dist, cities = load_data()

if 'optimal_sequence' not in st.session_state:
    st.session_state.optimal_sequence = None
if 'selected_start_city_index' not in st.session_state:
    st.session_state.selected_start_city_index = None

if df_ubicaciones is not None:
    n = len(df_ubicaciones)
    st.markdown(f"Optimizando la ruta para visitar las **{n} capitales** de Colombia.")
    st.markdown("---")
    st.subheader("Algoritmo Actual: Formulación de Dantzig-Fulkerson-Johnson (DFJ)")
    st.info("""
    Este método implementa la célebre formulación **Dantzig-Fulkerson-Johnson (DFJ)**, resuelta con Gurobi. El solver utiliza el **algoritmo Simplex** dentro de una estrategia general de **Branch and Cut**. Las restricciones de eliminación de sub-rutas se añaden dinámicamente como "cortes perezosos", lo cual se visualiza en cada iteración del mapa.
    """)
    
    summary_placeholder = st.empty()
    plot_placeholder = st.empty()

    if not st.session_state.optimal_sequence:
        plot_placeholder.plotly_chart(draw_map(df_ubicaciones), use_container_width=True, key="initial_map")

    if st.button('Iniciar Optimización'):
        st.session_state.optimal_sequence = None
        st.session_state.selected_start_city_index = None
        summary_placeholder.info("Calculando la ruta óptima... Este proceso puede tardar varios minutos.")
        
        m = Model()
        m.Params.lazyConstraints = 1
        
        m._df_ubicaciones = df_ubicaciones
        m._n = n
        m._subtours = 0
        m._iterations = 0
        m._summary = summary_placeholder
        m._plot = plot_placeholder
        
        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
        for i, j in list(vars.keys()):
            vars[j, i] = vars[i, j]

        m._vars = vars
        m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
        m.optimize(subtourelim)

        final_tour_edges = [k for k, v in m.getAttr('x', vars).items() if v > 0.5]
        final_tours = subtour(tuplelist(final_tour_edges), n)
        st.session_state.optimal_sequence = final_tours[0]
        
        summary_placeholder.success("¡Optimización Completada!")
        st.write(f"**Distancia Óptima Total:** `{round(m.objVal):,} km`")
        st.write(f"**Tiempo de Ejecución:** `{m.Runtime:.2f} segundos`")
        st.write(f"**Total de Iteraciones (Callbacks):** `{m._iterations}`")
        
        # Dibuja el mapa final estático en el placeholder de arriba
        final_fig = draw_map(df_ubicaciones, highlighted_tour=st.session_state.optimal_sequence)
        plot_placeholder.plotly_chart(final_fig, use_container_width=True, key="final_static_map")


    if st.session_state.optimal_sequence:
        st.markdown("---")
        st.subheader("Explora la Ruta Final")
        st.write("Selecciona una ciudad de inicio para ver la ruta detallada y su animación en el mapa.")

        city_index = 0
        for _ in range(4):
            cols = st.columns(8)
            for i in range(8):
                if city_index < n:
                    if cols[i].button(cities[city_index], key=f"city_{city_index}"):
                        st.session_state.selected_start_city_index = city_index
                    city_index += 1
        
        if st.session_state.selected_start_city_index is not None:
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
                route_to_show = st.radio(
                    "Selecciona la ruta a visualizar:",
                    ("Ruta Óptima", "Ruta Inversa"),
                    key="route_selector",
                    horizontal=True
                )

            route_map_placeholder = st.empty()
            
            highlight_sequence = reordered_forward if route_to_show == "Ruta Óptima" else reordered_backward
            
            # Animación de la ruta
            fig_anim = draw_map(df_ubicaciones) # Mapa base con ciudades
            route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.5)

            route_with_end = highlight_sequence + [highlight_sequence[0]]
            for i in range(len(route_with_end) - 1):
                segment = [route_with_end[i], route_with_end[i+1]]
                segment_points = df_ubicaciones.iloc[segment]
                fig_anim.add_trace(go.Scattermapbox(
                    lat=segment_points['Latitud'],
                    lon=segment_points['Longitud'],
                    mode='lines',
                    line=go.scattermapbox.Line(width=4, color='crimson'),
                ))
                route_map_placeholder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.25)
            
            # Rutas en texto
            with cols_vis[1]:
                route1_names = [cities[i] for i in reordered_forward] + [cities[start_node_index]]
                route2_names = [cities[i] for i in reordered_backward] + [cities[start_node_index]]

                st.markdown(f"#### Rutas desde **{cities[start_node_index]}**")
                st.markdown("**Ruta 1 (Sentido Óptimo):**")
                st.info(" ➔ ".join(route1_names))

                st.markdown("**Ruta 2 (Sentido Contrario):**")
                st.info(" ➔ ".join(route2_names))