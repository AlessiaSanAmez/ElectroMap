import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Centros de carga candidatos a implementar.
puntos_candidatos = np.array([
    # Centros de carga planteados por ubicación (plazas)
    [19.0433, -98.1984],
    [19.0289, -98.2046],
    [19.0739, -98.2196],
    [19.0634, -98.1642],
    [19.0279, -98.1915],

    # Centros de carga planteados por vértices de Voronoi
    [19.01712, -98.24946],
    [19.02168, -98.24339],
    [19.03927, -98.15768],
    [19.09426, -98.21494],
    [19.00562, -98.21534],
    [19.03235, -98.19280],
    [19.03331, -98.20339],
    [19.05158, -98.21207],
    [19.05730, -98.21099],
    [19.06263, -98.21040],
    [19.04541, -98.22863],
    [19.05624, -98.27949],
    [19.03534, -98.24883],
    [19.01472, -98.21456],
    [19.03529, -98.23756],
    [19.03621, -98.23631],
    [19.03318, -98.20881],
    [19.03625, -98.20960],
    [19.04294, -98.18850],
    [19.04534, -98.19141],
    [19.04841, -98.17966],
    [19.04313, -98.16784],
    [19.04429, -98.17000]
])

# Número de centros de carga que se planean implementar
k = 10

# Algoritmo k-means
# Mostrar los centroides (ubicaciones para nuevos centros de carga)
kmeans = KMeans(n_clusters=k)
kmeans.fit(puntos_candidatos)
nuevos_centros = kmeans.cluster_centers_
labels = kmeans.labels_

print("Nuevas ubicaciones para centros de carga:", nuevos_centros)
# --- Imprimir en formato para codigo ---
for i, centro in enumerate(nuevos_centros, start=1):
    print(f'{{"nombre": "{i}", "lat": {centro[0]:.5f}, "lon": {centro[1]:.5f}}},')

colors = ['blue', 'green', 'red', 'purple', 'brown', 'orange', 'pink', 'gray', 'olive', 'cyan']
plt.scatter(puntos_candidatos[:, 0], puntos_candidatos[:, 1], c=[colors[label] for label in labels], label='Puntos de Demanda')
plt.scatter(nuevos_centros[:, 0], nuevos_centros[:, 1], c='red', marker='x', label='Centros de Carga')

plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Agrupación de Puntos de Demanda y Centros de Carga')
plt.legend()
plt.show()
