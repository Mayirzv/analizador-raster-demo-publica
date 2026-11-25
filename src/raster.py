import csv
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.warp import transform, transform_geom
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FORMULAS = {
    'NDVI': '(NIR - ROJO)/(NIR + ROJO)',
    'EVI': '2.5*(NIR - ROJO)/(NIR + 6*ROJO - 7.5*AZUL + 1)',
    'SAVI': '1.5*(NIR - ROJO)/(NIR + ROJO + 0.5)',
    'NDWI': '(VERDE - NIR)/(VERDE + NIR)',
    'NDMI': '(NIR - SWIR1)/(NIR + SWIR1)',
}
REQUERIDAS = {
    'NDVI': ('NIR','ROJO'),
    'EVI': ('NIR','ROJO','AZUL'),
    'SAVI': ('NIR','ROJO'),
    'NDWI': ('VERDE','NIR'),
    'NDMI': ('NIR','SWIR1'),
}


def crear_demo(ruta, alto=120, ancho=140):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:alto,0:ancho]
    v = (np.sin(x / 18)*np.cos(y / 20)+1)/2
    ruido = rng.normal(0,0.012,(alto,ancho))
    bandas = {
        'AZUL': np.clip(0.08 + 0.03*v + ruido,0.01,1),
        'VERDE': np.clip(0.14 + 0.1*v + ruido,0.01,1),
        'ROJO': np.clip(0.24 - 0.16*v + ruido,0.01,1),
        'NIR': np.clip(0.25 + 0.50*v + ruido,0.01,1),
        'SWIR1': np.clip(0.35 - 0.18*v + ruido,0.01,1),
    }
    with rasterio.open(ruta,'w',driver='GTiff',width=ancho,height=alto,
        count=5,dtype='float32',crs='EPSG:32615',
        transform=rasterio.transform.from_origin(500000,2000000,10,10),
        compress='deflate') as ds:
        for i,(nombre, valores) in enumerate(bandas.items(),1):
            ds.write(valores.astype('float32'),i)
            ds.set_band_description(i,nombre)
    return ruta.resolve()


def metadata_y_bandas(ruta):
    ruta=Path(ruta).resolve()
    with rasterio.open(ruta) as ds:
        # Polígono de las cuatro esquinas, admite transformaciones afines rotadas.
        t=ds.transform
        esquinas=[t*(0,0),t*(ds.width,0),t*(ds.width,ds.height),t*(0,ds.height)]
        huella=None
        if ds.crs:
            try:
                figura={'type':'Polygon','coordinates':[[list(p) for p in esquinas+[esquinas[0]]]]}
                huella=json.dumps(transform_geom(ds.crs,'EPSG:4326',figura,antimeridian_cutting=True))
            except Exception:
                huella=None
        meta={
            'driver': ds.driver,'width':ds.width,'height':ds.height,'count':ds.count,
            'dtype':ds.dtypes[0],'crs':ds.crs.to_string() if ds.crs else None,
            'epsg':ds.crs.to_epsg() if ds.crs else None,
            'res_x':abs(ds.res[0]) or None,'res_y':abs(ds.res[1]) or None,
            'nodata':float(ds.nodata) if ds.nodata is not None and np.isfinite(ds.nodata) else None,
            'huella':huella,'size':ruta.stat().st_size,
        }
        bandas=[]
        for i in ds.indexes:
            # Estadísticas sobre una muestra reducida para evitar saturar RAM.
            escala=min(1.0,512/max(ds.width,ds.height))
            h=max(1,round(ds.height*escala)); w=max(1,round(ds.width*escala))
            a=ds.read(i,out_shape=(h,w),masked=True,resampling=Resampling.nearest)
            valores=a.compressed().astype('float64')
            bandas.append({
                'numero':i,'nombre':ds.descriptions[i-1] or f'Banda {i}',
                'min':float(np.min(valores)) if valores.size else None,
                'max':float(np.max(valores)) if valores.size else None,
                'mean':float(np.mean(valores)) if valores.size else None,
                'std':float(np.std(valores)) if valores.size else None,
                'scale':float(ds.scales[i-1]),'offset':float(ds.offsets[i-1]),
            })
        return meta,bandas


def previsualizar(ruta, banda=1, max_size=600):
    with rasterio.open(ruta) as ds:
        factor=min(1,max_size/max(ds.width,ds.height))
        shape=(max(1,int(ds.height*factor)),max(1,int(ds.width*factor)))
        a=ds.read(banda,out_shape=shape,masked=True,resampling=Resampling.nearest)
        return a.astype("float32").filled(np.nan)


def consultar_pixel(ruta,fila,columna,indices=None):
    with rasterio.open(ruta) as ds:
        if not (0 <= fila < ds.height and 0 <= columna < ds.width):
            raise ValueError('Fila o columna fuera del mapa')
        vals=ds.read(window=Window(columna,fila,1,1),masked=True)
        bandas={f'B{i}':(None if np.ma.is_masked(vals[i-1,0,0]) else float(vals[i-1,0,0])) for i in ds.indexes}
        x,y=xy(ds.transform,fila,columna,offset='center')
        lonlat=None
        if ds.crs:
            lon,lat=transform(ds.crs,'EPSG:4326',[x],[y]); lonlat=(float(lon[0]),float(lat[0]))
        valores_indices={}
        if indices:
            for nombre,path in indices.items():
                with rasterio.open(path) as d:
                    v=d.read(1,window=Window(columna,fila,1,1),masked=True)[0,0]
                    valores_indices[nombre]=None if np.ma.is_masked(v) else float(v)
        return {'fila':fila,'columna':columna,'x':float(x),'y':float(y),
                'lonlat':lonlat,'bandas':bandas,'indices':valores_indices}


def _div(a,b):
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        return np.divide(a,b,out=np.full_like(a,np.nan,dtype='float32'),where=np.abs(b)>1e-8)


def _indices(b, nombres):
    resultados={}
    R=b.get('ROJO'); N=b.get('NIR'); G=b.get('VERDE'); A=b.get('AZUL'); S=b.get('SWIR1')
    if 'NDVI' in nombres: resultados['NDVI']=_div(N-R,N+R)
    if 'EVI' in nombres: resultados['EVI']=_div(2.5*(N-R),N+6*R-7.5*A+1)
    if 'SAVI' in nombres: resultados['SAVI']=_div(1.5*(N-R),N+R+0.5)
    if 'NDWI' in nombres: resultados['NDWI']=_div(G-N,G+N)
    if 'NDMI' in nombres: resultados['NDMI']=_div(N-S,N+S)
    return resultados


def calcular_indices(ruta, carpeta, asignacion, nombres, factor=1.0, offset=0.0):
    """Los datos deben representar reflectancia; factor y offset convierten valores almacenados."""
    carpeta=Path(carpeta).resolve(); carpeta.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(factor) or factor<=0 or not np.isfinite(offset):
        raise ValueError('Factor debe ser positivo y factor/offset finitos')
    nombres=list(dict.fromkeys(nombres))
    if not nombres: raise ValueError('Seleccione al menos un índice')
    requeridas=set().union(*(REQUERIDAS[n] for n in nombres))
    if any(not asignacion.get(b) for b in requeridas):
        raise ValueError(f'Faltan bandas: {requeridas-set(k for k,v in asignacion.items() if v)}')
    if not set(nombres).issubset(REQUERIDAS):
        raise ValueError("Índice no reconocido")
    if len(set(asignacion[b] for b in requeridas)) != len(requeridas):
        raise ValueError("Cada componente espectral requerido debe tener una banda distinta")
    with rasterio.open(ruta) as ds:
        if ds.width*ds.height>3_000_000:
            raise ValueError('Límite inicial: 3 millones de píxeles para índices; utilice un recorte o implemente procesamiento por bloques')
        b={}
        for nombre in requeridas:
            numero=int(asignacion[nombre])
            if numero<1 or numero>ds.count: raise ValueError(f'Banda inválida: {nombre}={numero}')
            datos=ds.read(numero,masked=True).astype('float32')
            b[nombre]=datos.filled(np.nan)*factor+offset
        indices=_indices(b,nombres)
        resultados=[]
        for nombre,arr in indices.items():
            arr[~np.isfinite(arr)]=np.nan
            valido=arr[np.isfinite(arr)]
            perfil=ds.profile.copy()
            perfil.update(driver='GTiff',count=1,dtype='float32',nodata=-9999.0,compress='deflate')
            tif=(carpeta/f'{nombre.lower()}.tif').resolve()
            png=(carpeta/f'{nombre.lower()}.png').resolve()
            with rasterio.open(tif,'w',**perfil) as out:
                out.write(np.where(np.isfinite(arr),arr,-9999).astype('float32'),1)
                out.set_band_description(1,nombre)
            fig,ax=plt.subplots(figsize=(8,6))
            img=ax.imshow(np.ma.masked_invalid(arr),cmap='RdYlGn',vmin=-1,vmax=1)
            ax.set_title(nombre); fig.colorbar(img,ax=ax); fig.tight_layout(); fig.savefig(png,dpi=120); plt.close(fig)
            resultados.append({'nombre':nombre,'formula':FORMULAS[nombre],
                'min':float(np.min(valido)) if valido.size else None,
                'max':float(np.max(valido)) if valido.size else None,
                'mean':float(np.mean(valido)) if valido.size else None,
                'std':float(np.std(valido)) if valido.size else None,
                'count':int(valido.size),'tif':str(tif),'png':str(png)})
        return resultados


def exportar_pixeles_csv(ruta, destino):
    """
    Exporta los pixeles de un raster a CSV.

    Incluye:
    - Fila y columna
    - Coordenadas X/Y originales
    - Longitud y latitud (EPSG:4326)
    - Valor del pixel de la primera banda
    - Valores de todas las bandas

    Procesa por bloques para ahorrar memoria.
    """

    import csv
    import numpy as np
    import rasterio

    from pathlib import Path
    from rasterio.windows import Window
    from rasterio.warp import transform

    ruta = Path(ruta)
    destino = Path(destino)

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el mapa: {ruta}"
        )

    destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(ruta) as src:

        ancho = src.width
        alto = src.height
        total_pixeles = ancho * alto

        # Limitar el numero de pixeles leidos
        # en cada bloque para ahorrar memoria.
        filas_bloque = max(
            1,
            min(32, 100000 // max(1, ancho))
        )

        nombres_bandas = [
            f"B{i}" for i in src.indexes
        ]

        encabezados = [
            "fila",
            "columna",
            "x",
            "y",
            "longitud",
            "latitud",
            "valor_pixel",
        ] + nombres_bandas

        print(
            f"Iniciando exportacion: "
            f"{total_pixeles:,} pixeles"
        )

        with open(
            destino,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as archivo:

            escritor = csv.writer(archivo)
            escritor.writerow(encabezados)

            procesados = 0

            for inicio in range(
                0, alto, filas_bloque
            ):

                n_filas = min(
                    filas_bloque,
                    alto - inicio
                )

                ventana = Window(
                    0,
                    inicio,
                    ancho,
                    n_filas
                )

                datos = src.read(
                    window=ventana,
                    masked=True
                )

                # Convertir los datos a flotantes
                # antes de rellenar con NaN.
                valores = datos.astype(
                    "float64"
                ).filled(np.nan)

                valores = valores.reshape(
                    src.count,
                    -1
                )

                filas = np.repeat(
                    np.arange(
                        inicio,
                        inicio + n_filas
                    ),
                    ancho
                )
                columnas = np.tile(
                    np.arange(ancho),
                    n_filas
                )
                # Coordenadas del centro
                # de cada pixel.
                tr = src.transform

                x = (
                    tr.c
                    + (columnas + 0.5) * tr.a
                    + (filas + 0.5) * tr.b
                )

                y = (
                    tr.f
                    + (columnas + 0.5) * tr.d
                    + (filas + 0.5) * tr.e
                )

                # Transformacion a EPSG:4326.
                if src.crs is not None:

                    lon, lat = transform(
                        src.crs,
                        "EPSG:4326",
                        x.tolist(),
                        y.tolist()
                    )

                else:
                    lon = [None] * len(x)
                    lat = [None] * len(y)

                def limpiar_valor(valor):
                    if not np.isfinite(valor):
                        return None
                    return float(valor)

                def generar_filas():

                    for i in range(len(filas)):

                        bandas = [
                            limpiar_valor(
                                valores[b, i]
                            )
                            for b in range(src.count)
                        ]

                        valor_pixel = bandas[0]

                        yield [
                            int(filas[i]),
                            int(columnas[i]),
                            round(float(x[i]), 6),
                            round(float(y[i]), 6),
                            (
                                round(float(lon[i]), 8)
                                if lon[i] is not None
                                and np.isfinite(lon[i])
                                else None
                            ),
                            (
                                round(float(lat[i]), 8)
                                if lat[i] is not None
                                and np.isfinite(lat[i])
                                else None
                            ),
                            valor_pixel,
                            *bandas
                        ]

                escritor.writerows(
                    generar_filas()
                )

                procesados += len(filas)

                porcentaje = (
                    procesados / total_pixeles
                ) * 100

                print(
                    f"Exportado: "
                    f"{porcentaje:.1f}%"
                )

    print("CSV generado correctamente")
    print(f"Ubicacion: {destino.resolve()}")

    return destino.resolve()


import rasterio
from rasterio.warp import transform


def obtener_coordenadas_pixel(
    ruta_raster,
    fila,
    columna
):
    with rasterio.open(ruta_raster) as src:

        if not (0 <= fila < src.height):
            raise ValueError("Fila fuera del ráster")

        if not (0 <= columna < src.width):
            raise ValueError("Columna fuera del ráster")

        # Coordenadas del centro del píxel
        x, y = src.xy(fila, columna)

        longitud = None
        latitud = None

        if src.crs is not None:
            lon, lat = transform(
                src.crs,
                "EPSG:4326",
                [x],
                [y]
            )

            longitud = lon[0]
            latitud = lat[0]

        valores = src.read(
            window=((fila, fila + 1),
                    (columna, columna + 1))
        )

        return {
            "fila": fila,
            "columna": columna,
            "x": x,
            "y": y,
            "longitud": longitud,
            "latitud": latitud,
            "valores_bandas": valores[:, 0, 0].tolist()
        }


def generar_tabla_analitica(
    ruta,
    destino,
    roles=None,
    factor=1.0,
    offset=0.0
):
    """
    Analiza automaticamente todos los pixeles.

    roles: diccionario con numeros de bandas.
    Ejemplo:
    {
        "ROJO": 3,
        "NIR": 4,
        "VERDE": 2,
        "PH": 0,
        "ARCILLA": 0
    }

    El valor 0 significa no disponible.
    """

    import csv
    import numpy as np
    import rasterio

    from pathlib import Path
    from rasterio.windows import Window
    from rasterio.warp import transform

    roles = roles or {}
    ruta = Path(ruta)
    destino = Path(destino)

    destino.parent.mkdir(
        parents=True, exist_ok=True
    )

    campos = [
        "Pixel (F,C)",
        "Fila",
        "Columna",
        "Coord. X",
        "Coord. Y",
        "Longitud",
        "Latitud",
        "Valor pixel",
        "NDVI",
        "Valor pH",
        "Indice NDWI",
        "% Arcilla",
        "Estado Analitico"
    ]

    def numero(valor):
        try:
            n = float(valor)
            return n if np.isfinite(n) else None
        except (TypeError, ValueError):
            return None

    def indice(a, b):
        if a is None or b is None:
            return None

        denominador = a + b
        if abs(denominador) < 1e-10:
            return None

        return (a - b) / denominador

    def diagnostico(ndvi):
        if ndvi is None:
            return "Sin datos para interpretar NDVI"
        if ndvi < 0:
            return "NDVI negativo: revisar tipo de superficie"
        if ndvi < 0.2:
            return "NDVI muy bajo: posible cobertura escasa"
        if ndvi < 0.4:
            return "NDVI bajo: cobertura vegetal limitada"
        if ndvi < 0.6:
            return "NDVI moderado: cobertura apreciable"

        return "NDVI alto: cobertura vegetal aparente"

    with rasterio.open(ruta) as src:

        for rol, banda in roles.items():
            if banda is None or banda == 0:
                continue
            if not 1 <= int(banda) <= src.count:
                raise ValueError(
                    f"Banda invalida para {rol}: {banda}"
                )

        campos_bandas = [
            f"B{i}" for i in src.indexes
        ]

        campos_csv = (
            campos[:-1]
            + campos_bandas
            + ["Estado Analitico"]
        )

        total = src.width * src.height

        bloque = max(
            1,
            min(64, 50000 // max(1, src.width))
        )

        procesados = 0

        with open(
            destino, "w",
            encoding="utf-8-sig",
            newline=""
        ) as archivo:

            escritor = csv.writer(archivo)
            escritor.writerow(campos_csv)

            for inicio in range(0, src.height, bloque):

                cantidad = min(
                    bloque, src.height - inicio
                )

                ventana = Window(
                    0, inicio, src.width, cantidad
                )

                datos = src.read(
                    window=ventana,
                    masked=True
                ).astype("float32").filled(np.nan)

                filas, columnas = np.meshgrid(
                    np.arange(inicio, inicio + cantidad),
                    np.arange(src.width),
                    indexing="ij"
                )

                filas = filas.ravel()
                columnas = columnas.ravel()

                tr = src.transform

                xs = (
                    tr.c
                    + (columnas + 0.5) * tr.a
                    + (filas + 0.5) * tr.b
                )

                ys = (
                    tr.f
                    + (columnas + 0.5) * tr.d
                    + (filas + 0.5) * tr.e
                )

                if src.crs:
                    lon, lat = transform(
                        src.crs,
                        "EPSG:4326",
                        xs.tolist(),
                        ys.tolist()
                    )
                else:
                    lon = [None] * len(xs)
                    lat = [None] * len(xs)

                datos = datos.reshape(src.count, -1)

                def valor_rol(rol, posicion):
                    banda = roles.get(rol, 0)
                    if not banda:
                        return None
                    return numero(datos[banda - 1, posicion])

                for i in range(len(filas)):

                    valores = [
                        numero(datos[b, i])
                        for b in range(src.count)
                    ]

                    rojo = valor_rol("ROJO", i)
                    nir = valor_rol("NIR", i)
                    verde = valor_rol("VERDE", i)

                    # Se asume reflectancia correctamente
                    # escalada, configurada por el usuario.
                    def reflectancia(v):
                        if v is None:
                            return None
                        return v * factor + offset

                    rojo = reflectancia(rojo)
                    nir = reflectancia(nir)
                    verde = reflectancia(verde)

                    ndvi = indice(nir, rojo)
                    ndwi = indice(verde, nir)

                    ph = valor_rol("PH", i)
                    arcilla = valor_rol("ARCILLA", i)

                    # Evitar convertir NoData en mediciones.
                    fila_csv = [
                        f"({filas[i]},{columnas[i]})",
                        int(filas[i]),
                        int(columnas[i]),
                        numero(xs[i]),
                        numero(ys[i]),
                        numero(lon[i]),
                        numero(lat[i]),
                        valores[0],
                        ndvi,
                        ph,
                        ndwi,
                        arcilla,
                        *valores,
                        diagnostico(ndvi)
                    ]

                    escritor.writerow(fila_csv)

                procesados += len(filas)

                print(
                    f"Analizados {procesados:,} "
                    f"de {total:,} pixeles",
                    flush=True
                )

    return destino