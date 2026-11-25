# Analizador Ráster — prueba pública aislada

## Qué hace
- Abre sin cuenta de Streamlit cuando se despliega como aplicación pública.
- Genera mapa sintético multiespectral o admite GeoTIFF de prueba.
- Previsualiza y consulta píxeles por fila y columna.
- Calcula índices NDVI/EVI/SAVI/NDWI/NDMI, si las bandas son adecuadas.
- Exporta CSV completo de cinco columnas para el mapa cargado, con límites de prueba.

## Qué NO hace
- NO conecta con Supabase PostgreSQL ni Supabase Storage; no guarda proyectos compartidos.
- NO protege la privacidad de archivos cargados contra todos los riesgos de un hosting público.
- NO restringe qué dispositivos entran; cualquiera con el enlace puede abrirla.
- NO está dimensionada ni probada para 30 análisis simultáneos.
- Los archivos se alojan temporalmente en el servidor y pueden eliminarse al reiniciar. La carpeta temporal puede acumular datos durante la vida de la instancia; esta demo NO debe usarse para datos sensibles y necesita limpieza periódica para uso prolongado.

## Desplegar sin afectar la aplicación original
1. Crea un **repositorio nuevo** en GitHub. NO sobrescribas `analizador-raster-nube`.
2. Sube el contenido de esta carpeta a la raíz del repositorio nuevo, incluido `.streamlit/config.toml`.
3. En Streamlit Community Cloud, crea **New app**, elige ese repositorio y `app.py`.
4. En Sharing configura Public **solo para esta demo aislada**. No agregues ningún Secret.
5. Prueba con el mapa sintético y un GeoTIFF de prueba, desde una segunda PC.
6. Cuando conozcas la URL real, genera un instalador Inno Setup que abra la URL.

## Advertencia sobre la versión centralizada
Para conservar PostgreSQL, Storage y proyectos compartidos y además impedir que dispositivos no autorizados accedan sin login, hace falta un servidor o gateway administrado que distinga los dispositivos. Una web pública y un instalador que contiene solo un enlace no aportan esa propiedad. El usuario final puede seguir teniendo experiencia de un clic, pero la configuración/autorización se hace durante la distribución por un administrador.
