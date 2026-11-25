"""Demostracion publica aislada del Analizador Raster.

NO se conecta a Supabase ni usa secretos. Los datos de cada sesion son temporales.
NO utilizar para datos confidenciales: el servicio web es accesible publicamente.
"""
import csv
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
from rasterio.windows import Window
import streamlit as st

from src import raster as geo

st.set_page_config(page_title="Analizador Ráster · Prueba pública", page_icon="🗺️", layout="wide")
LIMITE_MB = 8
MAX_PIXELES = 600_000
MAX_BANDAS = 12


def sesion_dir():
    if "directorio" not in st.session_state:
        st.session_state.directorio = tempfile.mkdtemp(prefix="raster_demo_")
    return Path(st.session_state.directorio)


def cargar_mapa(ruta):
    with rasterio.open(ruta) as ds:
        if ds.driver != "GTiff":
            raise ValueError("El archivo no es un GeoTIFF válido.")
        if ds.width * ds.height > MAX_PIXELES or ds.count > MAX_BANDAS:
            raise ValueError(f"Límite de esta prueba: {MAX_PIXELES:,} píxeles y {MAX_BANDAS} bandas.")
        if ds.width < 1 or ds.height < 1:
            raise ValueError("Dimensiones inválidas")
    st.session_state.mapa = str(ruta)
    st.session_state.indices = []
    st.session_state.csv = None


def exportar_csv(ruta, banda, destino):
    with rasterio.open(ruta) as ds, open(destino, "w", newline="", encoding="utf-8-sig") as f:
        t = ds.transform
        utm = bool(ds.crs and ds.crs.to_dict().get("proj") == "utm")
        coord = "UTM" if utm else "CRS original"
        writer = csv.writer(f)
        writer.writerow(["Pixel (F, C)", f"Coord. X ({coord})", f"Coord. Y ({coord})", "Valor pixel", "Estado Analítico/Diagnóstico"])
        for inicio in range(0, ds.height, 64):
            n = min(64, ds.height - inicio)
            a = ds.read(banda, window=Window(0, inicio, ds.width, n), masked=True)
            mask = np.ma.getmaskarray(a).reshape(-1)
            valores = np.asarray(a.data).reshape(-1)
            filas = np.repeat(np.arange(inicio, inicio+n), ds.width)
            cols = np.tile(np.arange(ds.width), n)
            for i, (fila, col, v, nodata) in enumerate(zip(filas, cols, valores, mask)):
                ok = not nodata and bool(np.isfinite(v))
                x, y = t * (float(col)+0.5, float(fila)+0.5)
                writer.writerow([f"({fila}, {col})", x if ds.crs else "", y if ds.crs else "", float(v) if ok else "", "No determinado" if ok else "Sin dato"])
    return destino


st.title("🗺️ Analizador Ráster — prueba pública aislada")
st.warning("Esta prueba es pública y no requiere cuenta. Usa solo archivos de prueba o no confidenciales. Los mapas no se guardan en Supabase ni se comparten con otros usuarios. Los archivos temporales pueden desaparecer al reiniciarse el servidor.")
st.caption(f"Límites de prueba: {LIMITE_MB} MiB por archivo, {MAX_PIXELES:,} píxeles, {MAX_BANDAS} bandas. Para más capacidad se necesita alojamiento y controles de seguridad adecuados.")

with st.sidebar:
    st.header("Mapa de trabajo")
    archivo = st.file_uploader("Subir GeoTIFF de prueba", type=["tif", "tiff"], max_upload_size=LIMITE_MB)
    if st.button("Generar mapa sintético", use_container_width=True):
        ruta = sesion_dir() / f"demo_{uuid4().hex}.tif"
        geo.crear_demo(ruta)
        cargar_mapa(ruta)
        st.success("Mapa sintético generado.")
    if archivo is not None and st.button("Utilizar GeoTIFF cargado", use_container_width=True):
        try:
            if archivo.size > LIMITE_MB * 1024 * 1024:
                raise ValueError("Archivo demasiado grande")
            ruta = sesion_dir() / f"subido_{uuid4().hex}.tif"
            ruta.write_bytes(archivo.getvalue())
            cargar_mapa(ruta)
            st.success("GeoTIFF cargado.")
        except Exception as e:
            st.error(f"No se pudo cargar: {e}")

if "mapa" not in st.session_state:
    st.info("Para comenzar, pulsa «Generar mapa sintético» o sube un GeoTIFF de prueba.")
    st.stop()

ruta = Path(st.session_state.mapa)
if not ruta.exists():
    st.session_state.pop("mapa", None)
    st.warning("El servidor eliminó el archivo temporal. Carga el mapa nuevamente.")
    st.stop()

try:
    meta, bandas = geo.metadata_y_bandas(ruta)
except Exception as e:
    st.error(f"No se pudo leer el mapa: {e}")
    st.stop()

st.caption(f"Mapa: {ruta.name} · {meta['width']} × {meta['height']} píxeles · {meta['count']} bandas · CRS: {meta['crs'] or 'sin definir'}")
pestanas = st.tabs(["Explorar píxeles", "Índices de vegetación", "Exportación CSV"])

with pestanas[0]:
    banda = st.selectbox("Banda a visualizar", range(1, meta["count"]+1), format_func=lambda b: f"Banda {b}")
    try:
        fig = px.imshow(geo.previsualizar(ruta, banda), aspect="equal", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error de visualización: {e}")
    col1, col2 = st.columns(2)
    fila = col1.number_input("Fila (base cero)", 0, meta["height"]-1, min(50, meta["height"]-1), step=1)
    col = col2.number_input("Columna (base cero)", 0, meta["width"]-1, min(60, meta["width"]-1), step=1)
    if st.button("Consultar píxel"):
        try:
            st.json(geo.consultar_pixel(ruta, fila, col))
        except Exception as e:
            st.error(str(e))

with pestanas[1]:
    st.caption("Solo calcular índices para imágenes con bandas espectrales válidas, no mapas de elevación.")
    roles = ["AZUL", "VERDE", "ROJO", "NIR", "SWIR1"]
    columnas = st.columns(5)
    asignacion = {}
    for j, rol in enumerate(roles):
        opciones = [0] + list(range(1, meta["count"]+1))
        predeterminado = j + 1 if j < meta["count"] else 0
        asignacion[rol] = columnas[j].selectbox(rol, opciones, index=opciones.index(predeterminado), format_func=lambda x: "No disponible" if x == 0 else f"Banda {x}")
    seleccion = st.multiselect("Índices", list(geo.FORMULAS), default=["NDVI"])
    factor = st.number_input("Factor de reflectancia", value=1.0, format="%.8f")
    offset = st.number_input("Offset", value=0.0, format="%.8f")
    confirmado = st.checkbox("Confirmo que las bandas y su escala corresponden a reflectancia")
    if st.button("Calcular índices", disabled=not (seleccion and confirmado)):
        try:
            resultados = geo.calcular_indices(ruta, sesion_dir() / f"indices_{uuid4().hex}", asignacion, seleccion, factor, offset)
            st.session_state.indices = resultados
            st.success(f"{len(resultados)} índice(s) calculado(s).")
        except Exception as e:
            st.error(f"No se pudo calcular: {e}")
    for resultado in st.session_state.get("indices", []):
        st.subheader(resultado["nombre"])
        st.caption(geo.FORMULAS[resultado["nombre"]])
        st.image(resultado["png"], width=520)
        with open(resultado["tif"], "rb") as fh:
            st.download_button("Descargar GeoTIFF " + resultado["nombre"], fh, file_name=Path(resultado["tif"]).name, mime="image/tiff", key="tif_" + resultado["nombre"])

with pestanas[2]:
    st.caption("CSV de cinco columnas, con una fila por píxel; el diagnóstico no se inventa.")
    banda_csv = st.selectbox("Banda para exportar", range(1, meta["count"]+1), key="banda_csv")
    if st.button("Generar CSV completo"):
        try:
            destino = sesion_dir() / f"pixeles_{uuid4().hex}.csv"
            exportar_csv(ruta, banda_csv, destino)
            st.session_state.csv = str(destino)
            st.success("CSV completo generado.")
        except Exception as e:
            st.error(f"Error al generar CSV: {e}")
    csv_path = st.session_state.get("csv")
    if csv_path and Path(csv_path).exists():
        st.dataframe(pd.read_csv(csv_path, nrows=15), hide_index=True, use_container_width=True)
        with open(csv_path, "rb") as fh:
            st.download_button("Descargar CSV", fh, file_name="pixeles_cinco_columnas.csv", mime="text/csv", use_container_width=True)

st.caption("Versión de prueba. Sin PostgreSQL, Storage, cuentas, secretos ni persistencia entre sesiones.")
