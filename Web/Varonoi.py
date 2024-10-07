import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx

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

# Crear GeoDataFrame para los centros de carga
gdf_centros = gpd.GeoDataFrame(centros_carga, geometry=[Point(xy) for xy in zip([c['lon'] for c in centros_carga], [c['lat'] for c in centros_carga])], crs="EPSG:4326")

# Convertir a un CRS adecuado para las distancias métricas (por ejemplo, EPSG:3857)
gdf_centros = gdf_centros.to_crs(epsg=3857)

# Extraer las coordenadas proyectadas para el diagrama de Voronoi
centros_coords = np.array([[geom.x, geom.y] for geom in gdf_centros.geometry])
vor = Voronoi(centros_coords)

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar el diagrama de Voronoi
voronoi_plot_2d(vor, ax=ax, show_vertices=False)


# Añadir los centros de carga al gráfico
gdf_centros.plot(ax=ax, color='red', markersize=50, label="Centros de carga")

# Añadir mapa base de contexto
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Configurar etiquetas y mostrar el gráfico
plt.title('Diagrama de Voronoi para Centros de Carga en Puebla')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()
