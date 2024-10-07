import math

# Función para calcular la distancia utilizando la fórmula del Haversine
def haversine(coord1, coord2):
    R = 6371  # Radio de la Tierra en kilómetros
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Devuelve la distancia en kilómetros

# Función para calcular un tiempo estimado basado en la distancia
def estimate_travel_time(distancia, average_speed=40):  # velocidad promedio en km/h
    return (distancia / average_speed) * 60  # Devuelve el tiempo en minutos

# Función para calcular el peso del trayecto
def calculate_weight(distancia, travel_time, distance_weight=0.5, time_weight=0.5):
    return (distance_weight * distancia) + (time_weight * travel_time)

# Datos de los centros de carga
centros_carga = [
    {"nombre": "Parque Ecológico Zona 1", "lat": 19.023934964947465, "lon": -98.19086182319907},
    {"nombre": "Parque Ecológico Zona 2", "lat": 19.02645263253028, "lon": -98.18615931766742},
    {"nombre": "Centro de Convenciones", "lat": 19.043288, "lon": -98.191392},
    {"nombre": "Zona Los Fuertes", "lat": 19.057078, "lon": -98.183774},
    {"nombre": "Centro Expositor", "lat": 19.059195795750288, "lon": -98.18113773541947},
    {"nombre": "Mercedes-Benz", "lat": 19.015830605910985, "lon": -98.25505568911856},  # Tlaxcalancingo
    {"nombre": "Tesla Charger - Solesta", "lat": 19.037843139644565, "lon": -98.22920118256935},
    {"nombre": "Chevrolet", "lat": 19.02163244026228, "lon": -98.25267738876323},  # Tlaxcalancingo
    {"nombre": "Tesla Charger - Angelopolis", "lat": 19.013140490360847, "lon": -98.24553245644337},
    {"nombre": "Tesla Charger - Animas", "lat": 19.050579007621728, "lon": -98.23450884657841},
    {"nombre": "Tesla Charger - Analco", "lat": 19.040263080452775, "lon": -98.18920286277289},
    {"nombre": "Tesla Charger - Ikea", "lat": 19.027432446573208, "lon": -98.23637255184909},
    {"nombre": "Tesla Charger - Centro", "lat": 19.04616300185802, "lon": -98.18942555533113},
    {"nombre": "Tesla Charger - Convento", "lat": 19.045596717489577, "lon": -98.18925990138358},
    {"nombre": "Tesla Charger - Triangulo", "lat": 19.043126732422195, "lon": -98.23628403630079},
    {"nombre": "BMW Station", "lat": 19.034603796048074, "lon": -98.22919598684334}
]

# Crear el grafo de pesos
peso_grafo = {}

for i in range(len(centros_carga)):
    for j in range(i + 1, len(centros_carga)):
        coord_a = (centros_carga[i]['lat'], centros_carga[i]['lon'])
        coord_b = (centros_carga[j]['lat'], centros_carga[j]['lon'])
        
        distancia = haversine(coord_a, coord_b)
        tiempo_estimado = estimate_travel_time(distancia)
        
        peso = calculate_weight(distancia, tiempo_estimado)
        peso_grafo[(centros_carga[i]['nombre'], centros_carga[j]['nombre'])] = peso

# Imprimir los pesos del grafo
for trayecto, peso in peso_grafo.items():
    print(f"Peso del trayecto de {trayecto[0]} a {trayecto[1]}: {peso:.2f}")