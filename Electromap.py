import os
os.environ["OMP_NUM_THREADS"] = "1"
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx
from pyproj import Transformer
from sklearn.cluster import KMeans

# --------------- VORONOI CON CENTROS DE CARGA EXISTENTES ---------------

# --- Centros de carga en Puebla ---
def read_centros_from_txt(file_path):
    centros_carga = []
    with open(file_path, 'r') as file:
        for line in file:
            nombre, lat, lon = line.strip().split(',')
            centros_carga.append({"nombre": nombre.strip(), "lat": float(lat), "lon": float(lon)})
    return centros_carga

centros_carga = read_centros_from_txt('Recursos/centros_carga.txt')

# GeoDataFrame del mapa de Puebla con el geoJson del municipio de puebla.
gdf_puebla = gpd.read_file('Recursos/Zona_Urbana_2023.geojson')
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
# plt.savefig('voronoi_diagram.png', bbox_inches='tight')
plt.savefig('Web/Icons/VoronoiCentrosExistentes.png', bbox_inches='tight')

# Vertices del diagrama Voronoi
# Se guardan en EPSG:3957, para poder usarlo en mapas se necesita transformar a EPSG:3857
vertices = vor.vertices
vertices_4326 = []
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
for x, y in vertices:
    xNew, yNew = transformer.transform(x,y)
    vertices_4326.append((xNew, yNew))
# --- Imprimir en formato para codigo ---
print(",\n".join([f"[{x}, {y}]" for x, y in vertices_4326]))


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
plt.savefig('Web/Icons/VoronoiPuebla.png', bbox_inches='tight')
plt.show()


# --------------- K-MEANS CON CANDIDATOS Y VÉRTICES ---------------

# Centros de carga candidatos a implementar.
def read_puntos_candidatos(file_path):
    puntos_candidatos = []
    with open(file_path, 'r') as file:
        for line in file:
            lat, lon = line.strip().split(',')
            puntos_candidatos.append([float(lat), float(lon)])
    return np.array(puntos_candidatos)

# Centros de carga planteados por ubicación (plazas)
puntos_candidatos = read_puntos_candidatos('Recursos/puntos_candidatos.txt')
# Centros de carga planteados por vértices de Voronoi
puntos_candidatos = np.vstack((puntos_candidatos, vertices_4326))

# Número de centros de carga que se planean implementar
k = 10

# Algoritmo k-means
# Mostrar los centroides (ubicaciones para nuevos centros de carga)
kmeans = KMeans(n_clusters=k,random_state=42)
kmeans.fit(puntos_candidatos)
nuevos_centros = kmeans.cluster_centers_
labels = kmeans.labels_

print("Nuevas ubicaciones para centros de carga:", nuevos_centros)
# --- Imprimir en formato para codigo ---
centros_candidatos = []
for i, centro in enumerate(nuevos_centros, start=1):
    centros_candidatos.append({"nombre": str(i), "lat": float(centro[0]), "lon": float(centro[1])})  

colors = ['blue', 'green', 'red', 'purple', 'brown', 'orange', 'pink', 'gray', 'olive', 'cyan']
plt.scatter(puntos_candidatos[:, 0], puntos_candidatos[:, 1], c=[colors[label] for label in labels], label='Puntos de Demanda')
plt.scatter(nuevos_centros[:, 0], nuevos_centros[:, 1], c='red', marker='x', label='Centros de Carga')

plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Agrupación de Puntos de Demanda y Centros de Carga')
plt.legend()
plt.savefig('Web/Icons/Kmeans.png', bbox_inches='tight')
plt.show()


# --------------- ALGORITMO TSP - CENTROS CONSOLIDADOS ---------------

# Función para determinar la distancia Harversine.
# Distancia entre dos puntos de la tierra.
# El resultado se presenta en km.
def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Estimar el tiempo de viaje de acuerdo con la distancia.
# Regresa el tiempo en minutos.
def tiempoEstimado(distancia, velocidad_avg=40):
    return (distancia / velocidad_avg) * 60 

# Calcular peso de trayectoria.
def calc_peso(distancia, tiempoTrayectoria, pesoDistancia=0.5, pesoTiempo=0.5):
    return (pesoDistancia * distancia) + (pesoTiempo * tiempoTrayectoria)

combinations = list(itertools.combinations(centros_candidatos, 5))

best_combination = None
peso_best = float('inf')
for combination in combinations:
    total_centros = centros_carga + list(combination)
    
    peso_total = 0
    for i in range(len(total_centros)):
        for j in range(i + 1, len(total_centros)):
            coord_a = (total_centros[i]['lat'], total_centros[i]['lon'])
            coord_b = (total_centros[j]['lat'], total_centros[j]['lon'])
            
            distancia = haversine(coord_a, coord_b)
            tiempo_estimado = tiempoEstimado(distancia)
            
            peso = calc_peso(distancia, tiempo_estimado)
            peso_total += peso

    if peso_total < peso_best:
        peso_best = peso_total
        best_combination = combination

print(f"Mejor combinación: {', '.join([c['nombre'] for c in best_combination])}")
print(f"Con peso total: {peso_best:.2f}")

# --------------- VORONOI CON CENTROS DE CARGA CONSOLIDADOS ---------------

centros_consolidados = centros_carga + list(best_combination)

# GeoDataFrame del mapa de Puebla con el geoJson del municipio de puebla.
gdf_puebla = gpd.read_file('Recursos/Zona_Urbana_2023.geojson')
gdf_puebla = gdf_puebla.to_crs(epsg=3857)
# GeoDataFrame de los centros de carga
gdf_centros = gpd.GeoDataFrame(centros_consolidados, geometry=[Point(xy) for xy in zip([c['lon'] for c in centros_consolidados], [c['lat'] for c in centros_consolidados])], crs="EPSG:4326")
gdf_centros = gdf_centros.to_crs(epsg=3857)

# Creación del diagrama Voronoi con las coordenadas de los centros de carga.
centros_coords = np.array([[geom.x, geom.y] for geom in gdf_centros.geometry])
vor = Voronoi(centros_coords)
fig, ax = plt.subplots(figsize=(8, 8))

# Contorno del municipio de puebla, graficación con los puntos Voronoi.
gdf_puebla.plot(ax=ax, edgecolor='black', facecolor='none')
plt.title('Diagrama de Voronoi con nuevos Centros de Carga')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
voronoi_plot_2d(vor, ax=ax, line_colors='blue', show_vertices=False)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.savefig('Web/Icons/VoronoiConsolidados.png', bbox_inches='tight')
plt.show()