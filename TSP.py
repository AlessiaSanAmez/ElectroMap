import itertools
import math

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

# Centros de carga existentes.
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

centros_candidatos = [
    {"nombre": "1", "lat": 19.01940, "lon": -98.24643},
    {"nombre": "2", "lat": 19.03198, "lon": -98.20178},
    {"nombre": "3", "lat": 19.06135, "lon": -98.21327},
    {"nombre": "4", "lat": 19.06340, "lon": -98.16420},
    {"nombre": "5", "lat": 19.03806, "lon": -98.23783},
    {"nombre": "6", "lat": 19.04223, "lon": -98.16517},
    {"nombre": "7", "lat": 19.05624, "lon": -98.27949},
    {"nombre": "8", "lat": 19.01017, "lon": -98.21495},
    {"nombre": "9", "lat": 19.09426, "lon": -98.21494},
    {"nombre": "10", "lat": 19.04500, "lon": -98.18949}
]


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

print(f"Best combination: {', '.join([c['nombre'] for c in best_combination])}")
print(f"With total peso: {peso_best:.2f}")
