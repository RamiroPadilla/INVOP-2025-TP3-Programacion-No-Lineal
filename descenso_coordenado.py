#%%
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
import time

def descenso_coordenado(xk, k, pesos, puntos):
    def f_univariable(a):
        xk_temp = np.copy(xk)
        xk_temp[k] += a
        return sum(w * np.linalg.norm(xk_temp - p) for w, p in zip(pesos, puntos))

    res = minimize_scalar(f_univariable)
    return res.x  #valor óptimo de a


def resolver_descenso(puntos, pesos, max_iter=5000, tol=1e-6,):
    t0 = time.time()
    dim = len(puntos[0])

    xk = np.average(puntos, axis=0, weights=pesos) #punto inicial, promedio con pesos
    
    for iter in range(max_iter): 
        xk_tmp = np.copy(xk)

        for k in range(dim): #iteraciones ciclicas
            a_k = descenso_coordenado(xk, k, pesos, puntos)
            xk[k] += a_k
            # print(k) # para ver los pasos
        if np.linalg.norm(xk - xk_tmp) < tol:
            break

    tiempo = time.time() - t0
    return xk, tiempo, iter

#%%

# Ejemplo de uso
datos = np.loadtxt('instancias\distribucion\inst_n2_m5000_clusters.txt', skiprows=2)
P = datos[:, :-1]
w = datos[:, -1]

punto_optimo, tiempo_ejecucion, cant_iteraciones = resolver_descenso(P, w)

print("Punto óptimo:", punto_optimo)
print("Tiempo de ejecución:", tiempo_ejecucion)
print("Cantidad de iteraciones:", cant_iteraciones)
# print("Valor de la función objetivo:", f(punto_optimo, P, w))
# %%

# solo graficar si estamos en R2

punto_inicial = np.average(P, axis=0, weights=w)

plt.scatter(P[:, 0], P[:, 1], s=8, alpha=0.6, c='blue', label="Puntos")
plt.scatter(punto_inicial[0], punto_inicial[1], c='black', marker='x', s=100,label='Punto inicial')
plt.scatter(punto_optimo[0], punto_optimo[1], c='red', marker='x', s=100, label='Punto optimo')
plt.legend()
plt.xlim(-12000, 12000)
plt.ylim(-12000, 12000)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
# %%
