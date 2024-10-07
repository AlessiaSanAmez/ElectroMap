import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx
from pyproj import Transformer

import shapely.ops as ops

# Centros de carga en Puebla
centros_carga = [
    # Agencia de Energia
    {"nombre": "Parque Ecológico Zona 1", "lat": 19.023934964947465, "lon": -98.19086182319907},
    {"nombre": "Parque Ecológico Zona 2", "lat": 19.02645263253028, "lon": -98.18615931766742},
    {"nombre": "Centro de Convenciones", "lat": 19.043288, "lon": -98.191392},
    {"nombre": "Zona Los Fuertes", "lat": 19.057078, "lon": -98.183774},
    {"nombre": "Centro Expositor", "lat": 19.059195795750288, "lon": -98.18113773541947},

    #Google Maps
    {"nombre": "Mercedes-Benz", "lat": 19.015830605910985, "lon": -98.25505568911856}, # Tlaxcalancingo 
    {"nombre": "Tesla Charger - Solesta", "lat": 19.037843139644565, "lon": -98.22920118256935},
    {"nombre": "Chevrolet", "lat": 19.02163244026228, "lon": -98.25267738876323}, # Tlaxcalancingo
    {"nombre": "Tesla Charger - Angelopolis", "lat": 19.013140490360847, "lon": -98.24553245644337},
    {"nombre": "Tesla Charger - Animas", "lat": 19.050579007621728, "lon": -98.23450884657841},
    {"nombre": "Tesla Charger - Analco", "lat": 19.040263080452775, "lon": -98.18920286277289},
    {"nombre": "Tesla Charger - Ikea", "lat": 19.027432446573208, "lon": -98.23637255184909},
    {"nombre": "Tesla Charger - Centro", "lat": 19.04616300185802, "lon": -98.18942555533113},
    {"nombre": "Tesla Charger - Convento", "lat": 19.045596717489577, "lon": -98.18925990138358},
    {"nombre": "Tesla Charger - Triangulo", "lat": 19.043126732422195, "lon": -98.23628403630079},
    {"nombre": "BMW Station", "lat": 19.034603796048074, "lon": -98.22919598684334}
]

# GeoDataFrame del mapa de Puebla con el geoJson del municipio de puebla.
gdf_puebla = gpd.read_file('Recursos\Zona_Urbana_2023.geojson')
gdf_puebla = gdf_puebla.to_crs(epsg=3857)
# GeoDataFrame de los centros de carga
gdf_centros = gpd.GeoDataFrame(centros_carga, geometry=[Point(xy) for xy in zip([c['lon'] for c in centros_carga], [c['lat'] for c in centros_carga])], crs="EPSG:4326")
gdf_centros = gdf_centros.to_crs(epsg=3857)

# Creación del diagrama Voronoi con las coordenadas de los centros de carga.
centros_coords = np.array([[geom.x, geom.y] for geom in gdf_centros.geometry])
vor = Voronoi(centros_coords)
fig, ax = plt.subplots(figsize=(8, 8))

# Contorno del municipio de puebla, graficación con los puntos Voronoi.
# Gráfica Voronoi con mapa de fondo, almacenada en formato png.
gdf_puebla.plot(ax=ax, edgecolor='black', facecolor='none')
plt.title('Diagrama de Voronoi para Centros de Carga')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
voronoi_plot_2d(vor, ax=ax, line_colors='blue')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.savefig('voronoi_diagram.png', bbox_inches='tight')

# Vertices del diagrama Voronoi
# Se guardan en EPSG:3957, para poder usarlo en mapas se necesita transformar a EPSG:3857
vertices = vor.vertices
vertices_4326 = []
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
for x, y in vertices:
    xNew, yNew = transformer.transform(x,y)
    vertices_4326.append((xNew, yNew))
# --- Imprimir en formato para codigo ---
print(",\n".join([f"[{x:.5f}, {y:.5f}]" for x, y in vertices_4326]))


# Obtener el polígono para aplicarlo al mapa del municipio de Puebla
voronoi_polygons = [Polygon([vor.vertices[i] for i in region]) for region in vor.regions if -1 not in region]
voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_polygons, crs=gdf_puebla.crs)
clipped_voronoi = gpd.clip(voronoi_gdf, gdf_puebla)

fig, ax = plt.subplots(figsize=(8, 8))
gdf_puebla.plot(ax=ax, edgecolor='black', facecolor='none')
clipped_voronoi.plot(ax=ax, color='blue', alpha=0.5)
gdf_centros.plot(ax=ax, color='red', markersize=50, label="Centros de carga")

# Para agregar el mapa de fondo
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

plt.title('Diagrama de Voronoi para Centros de Carga en Puebla de Zaragoza')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()