import streamlit as st
import pandas as pd
import itertools
from gurobipy import Model, GRB, quicksum, tuplelist
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title('TSP: Ruta Óptima por las Capitales de Colombia')

@st.cache_data
def load_data():
    try:
        df_ubicaciones = pd.read_csv('ubicacion.csv')
        df_distancias = pd.read_csv('distancias.csv', index_col='Ciudad')
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que 'ubicacion.csv' y 'distancias.csv' estén en la misma carpeta que el script.")
        return None, None, None

    dist = {
        (i, j): df_distancias.iloc[i, j]
        for i in range(len(df_distancias))
        for j in range(i)
    }
    return df_ubicaciones, df_distancias, dist

def subtour(edges):
    unvisited = list(range(n))
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
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        
        tours = subtour(selected)
        if len(tours) > 1:
            model._subtours += 1
            for tour in tours:
                model.cbLazy(quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2)) <= len(tour) - 1)
        
        current_length = round(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        model._summary.markdown(f"""
        - **Distancia Actual:** `{current_length:,} km`
        - **Sub-rutas Encontradas:** `{len(tours)}`
        - **Restricciones Añadidas:** `{model._subtours}`
        """)
        
        fig = draw_map(model._df_ubicaciones, tours)
        model._plot.plotly_chart(fig, use_container_width=True)

def draw_map(df_ubicaciones, tours=None):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=df_ubicaciones['Latitud'],
        lon=df_ubicaciones['Longitud'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=10,
            color='darkblue'
        ),
        text=df_ubicaciones['Capital'],
        textposition='top right'
    ))

    if tours:
        for i, tour in enumerate(tours):
            tour.append(tour[0])
            tour_points = df_ubicaciones.iloc[tour]
            fig.add_trace(go.Scattermapbox(
                lat=tour_points['Latitud'],
                lon=tour_points['Longitud'],
                mode='lines',
                line=go.scattermapbox.Line(width=2),
                name=f"Ruta {i+1}"
            ))

    fig.update_layout(
        showlegend=False,
        mapbox_style="carto-positron",
        mapbox_zoom=4.5,
        mapbox_center={"lat": 4.5709, "lon": -74.2973},
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig

df_ubicaciones, df_distancias, dist = load_data()

if df_ubicaciones is not None:
    n = len(df_ubicaciones)
    st.write(f"Optimizando la ruta para visitar las **{n} capitales** de Colombia.")
    
    summary_placeholder = st.empty()
    plot_placeholder = st.empty()

    plot_placeholder.plotly_chart(draw_map(df_ubicaciones), use_container_width=True)

    if st.button('Iniciar Optimización'):
        summary_placeholder.info("Calculando la ruta óptima... Este proceso puede tardar varios minutos.")
        
        m = Model()
        m.Params.lazyConstraints = 1
        
        m._df_ubicaciones = df_ubicaciones
        m._subtours = 0
        m._summary = summary_placeholder
        m._plot = plot_placeholder
        
        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
        for i, j in vars.keys():
            vars[j, i] = vars[i, j]

        m._vars = vars

        m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
        
        m.optimize(subtourelim)

        final_tour_edges = [k for k, v in m.getAttr('x', vars).items() if v > 0.5]
        final_tours = subtour(tuplelist(final_tour_edges))
        
        final_fig = draw_map(df_ubicaciones, final_tours)
        plot_placeholder.plotly_chart(final_fig, use_container_width=True)

        summary_placeholder.success("¡Optimización Completada!")
        st.write(f"**Distancia Óptima Total:** `{round(m.objVal):,} km`")
        st.write(f"**Tiempo de Ejecución:** `{m.Runtime:.2f} segundos`")
        st.write(f"**Restricciones de sub-rutas añadidas:** `{m._subtours}`")
        