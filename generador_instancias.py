#%%
import os
import numpy as np
from sklearn.datasets import make_blobs

def generar_instancia(n, m, rango, tipo='uniforme', archivo_salida='instancia.txt', k_clusters=3):
    # Generar puntos según tipo
    if tipo == 'uniforme':
        puntos = np.random.uniform(low=rango[0], high=rango[1], size=(m, n))

    elif tipo == 'clusters':
        if k_clusters > m:
            raise ValueError("La cantidad de clusters no puede superar la cantidad de puntos")
        std = 1000
        rango_centros = (rango[0] + std*3, rango[1] - std*3) 
        puntos, _ = make_blobs(n_samples=m, centers=k_clusters, n_features=n, cluster_std=std, center_box= rango_centros)

    elif tipo == 'lineal':
        a = np.random.uniform(*rango, size=n)
        b = np.random.uniform(*rango, size=n)
        t = np.linspace(0, 1, m).reshape(-1, 1)
        puntos = (1 - t) * a + t * b

    else:
        raise ValueError(f"Tipo de instancia no reconocido: {tipo}")

    pesos = np.random.randint(1, 50, size=(m, 1)) # PESOS ENTRE 1 Y 50
    datos = np.hstack([puntos, pesos])

    with open(archivo_salida, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{m}\n")
        for fila in datos:
            coord = " ".join(f"{x:.4f}" for x in fila[:-1])
            peso = int(fila[-1])
            f.write(f"{coord} {peso}\n")

    print(f"Instancia guardada en: {archivo_salida}")

#%% Iniciar instancias variadas

# Parámetros
dimensiones = [2, 5, 10]
cant_puntos = [2000, 5000, 10000]
tipos = ['uniforme', 'clusters', 'lineal']
k_clusters = 6  # número de clusters para el tipo clusters
rango = (-10000, 10000)

# Carpeta donde guardar las instancias
folder = 'instancias\distribucion'

if not os.path.exists(folder):
    os.makedirs(folder)

# Generar todas las combinaciones
for n in dimensiones:
    for m in cant_puntos:
        for tipo in tipos:
            # Para clusters: evitar que k_clusters > m
            k_c = min(k_clusters, m)
            nombre_archivo = f"{folder}/inst_n{n}_m{m}_{tipo}.txt"
            generar_instancia(n=n, m=m, rango = rango, tipo=tipo, archivo_salida=nombre_archivo, k_clusters=k_c)


#%% Iniciar mas instancias

dimensiones = [2, 5, 10, 15, 20, 30]
cant_puntos = [2000, 5000, 10000]
tipos = ['uniforme']
k_clusters = 6  # número de clusters para el tipo clusters
rango = (-10000, 10000)

# Carpeta donde guardar las instancias
folder = 'instancias\dimensional'

if not os.path.exists(folder):
    os.makedirs(folder)

# Generar todas las combinaciones
for n in dimensiones:
    for m in cant_puntos:
        for tipo in tipos:
            # Para clusters: evitar que k_clusters > m
            k_c = min(k_clusters, m)
            nombre_archivo = f"{folder}/inst_n{n}_m{m}_{tipo}_{3}.txt"
            generar_instancia(n=n, m=m, rango = rango, tipo=tipo, archivo_salida=nombre_archivo, k_clusters=k_c)

# %%
