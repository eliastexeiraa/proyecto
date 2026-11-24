import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


# Configuración de la página
st.set_page_config(
    page_title="Análisis de Viviendas",
    page_icon="🏠",
    layout="wide"
)

# Título y descripción
st.title("Análisis Interactivo de Datos: California Housing")

st.write(
    "Esta aplicación permite explorar y analizar "
    "datos sobre viviendas en California."
)

# Cargar el dataset
housing = fetch_california_housing()

# Crear el DataFrame
df_california = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df_california["MedHouseVal"] = housing.target
# Barra lateral de filtros
st.sidebar.markdown("## Filtros de búsqueda")
st.sidebar.write("Seleccioná las características de las viviendas.")

# Filtro por antigüedad de la vivienda
edad_minima, edad_maxima = st.sidebar.slider(
    "Rango de antigüedad de las viviendas",
    min_value=float(df_california["HouseAge"].min()),
    max_value=float(df_california["HouseAge"].max()),
    value=(
        float(df_california["HouseAge"].min()),
        float(df_california["HouseAge"].max())
    )
)

# Filtro por latitud mínima
latitud_minima = st.sidebar.number_input(
    "Latitud mínima",
    min_value=float(df_california["Latitude"].min()),
    max_value=float(df_california["Latitude"].max()),
    value=float(df_california["Latitude"].min()),
    step=0.1
)

# Aplicar los filtros al DataFrame
df_filtrado = df_california[
    (df_california["HouseAge"] >= edad_minima) &
    (df_california["HouseAge"] <= edad_maxima) &
    (df_california["Latitude"] >= latitud_minima)
]

# Mostrar cantidad de registros
st.subheader("Resultados del filtrado")
st.write("Cantidad de viviendas:", len(df_filtrado))
st.dataframe(df_filtrado.head())
# Resumen descriptivo de las viviendas filtradas
st.subheader("Resumen descriptivo")

if len(df_filtrado) > 0:

    mediana = df_filtrado["MedHouseVal"].median()

    valor_maximo = df_filtrado["MedHouseVal"].max()
    valor_minimo = df_filtrado["MedHouseVal"].min()

    rango = valor_maximo - valor_minimo

    st.write("Mediana del valor de las viviendas:", round(mediana, 3))
    st.write("Rango del valor de las viviendas:", round(rango, 3))

else:
    st.warning("No hay viviendas que cumplan con los filtros seleccionados.")

# Histograma del valor de las viviendas
st.subheader("Distribución del valor de las viviendas")

if len(df_filtrado) > 0:
    st.pyplot(
        df_filtrado["MedHouseVal"].hist(
            bins=30,
            figsize=(10, 5)
        ).get_figure()
    )
else:
    st.warning("No hay datos disponibles para generar el gráfico.")

# Gráfico de dispersión: ingresos y valor de las viviendas
st.subheader("Relación entre ingresos y valor de las viviendas")

if len(df_filtrado) > 0:

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df_filtrado["MedInc"],
        df_filtrado["MedHouseVal"],
        alpha=0.5
    )

    ax.set_xlabel("Mediana de ingresos (MedInc)")
    ax.set_ylabel("Valor mediano de la vivienda (MedHouseVal)")
    ax.set_title("Relación entre ingresos y valor de las viviendas")

    st.pyplot(fig)

else:
    st.warning("No hay datos disponibles para generar el gráfico.")

# Mostrar las primeras cinco filas filtradas
st.subheader("Primeras cinco filas de las viviendas filtradas")
st.dataframe(df_filtrado.head().reset_index(drop=True))

# Mostrar tipos de datos
st.subheader("Tipos de datos de las columnas")
tipos_datos = df_california.dtypes.astype(str)
st.dataframe(tipos_datos.rename("Tipo de dato"))