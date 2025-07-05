#%%

from scipy.optimize import minimize_scalar
import numpy as np
import time

TOLERANCE = 1e-07

#data 
rng = np.random.default_rng()
puntos = np.round(rng.uniform(low=0, high=1000000, size=(500, 20)), decimals=4)
pesos = rng.integers(1, 25, size=500)
punto_inicial = np.average(puntos, axis=0, weights=pesos) ## para comparar despues 

#%%
# solo graficar si estamos en R2
import matplotlib.pyplot as plt

plt.scatter(puntos[:, 0], puntos[:, 1], label="Puntos", color='blue')
plt.scatter(punto_inicial[0], punto_inicial[1], color='red', marker='x', s=100, label="Promedio ponderado")
#%%

def descenso_coordenado(xk, k, pesos, puntos):
    def f_univariable(a):
        xk_temp = np.copy(xk)
        xk_temp[k] += a
        return sum(w * np.linalg.norm(xk_temp - p) for w, p in zip(pesos, puntos))

    res = minimize_scalar(f_univariable)
    return res.x  #valor óptimo de a


def resolver_descenso(puntos, pesos, max_iter=100):
    dim = len(puntos[0])

    xk = np.average(puntos, axis=0, weights=pesos) #punto inicial, promedio con pesos
    
    for iter in range(max_iter): 
        xk_tmp = np.copy(xk)

        for k in range(dim): #iteraciones ciclicas
            a_k = descenso_coordenado(xk, k, pesos, puntos)
            xk[k] += a_k
            # print(k) # para ver los pasos
        if np.linalg.norm(xk - xk_tmp) < TOLERANCE:
            break
    return xk

start = time.time()
solucion = resolver_descenso(puntos, pesos)
end = time.time()
tiempo = end - start
print(f'Tiempo de computo para instancia de {len(puntos)} puntos, de dimension {len(puntos[0])}:') 
print(f'{tiempo:.5f} segundos')

#%%
# solo graficar si estamos en R2
plt.scatter(puntos[:, 0], puntos[:, 1], label="Puntos", color='blue')
plt.scatter(solucion[0], solucion[1], color='red', marker='x', s=100, label="Promedio ponderado")

# %%
dist_recorrida = np.linalg.norm(solucion - punto_inicial)
print(dist_recorrida)
