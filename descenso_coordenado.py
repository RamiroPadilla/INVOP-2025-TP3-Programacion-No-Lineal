from scipy.optimize import minimize_scalar
import numpy as np
import time

def descenso_coordenado(xk, k, pesos, puntos):
    def f_univariable(a):
        xk_temp = np.copy(xk)
        xk_temp[k] += a
        return sum(w * np.linalg.norm(xk_temp - p) for w, p in zip(pesos, puntos))

    res = minimize_scalar(f_univariable)
    return res.x


def resolver_descenso_coordenado(puntos, pesos, max_iter=5000, tol=1e-4,):
    t0 = time.time()
    dim = len(puntos[0])

    xk = np.average(puntos, axis=0, weights=pesos) # punto inicial, promedio con pesos
    
    for iter in range(max_iter): 
        xk_tmp = np.copy(xk)

        for k in range(dim): # iteraciones ciclicas
            a_k = descenso_coordenado(xk, k, pesos, puntos)
            xk[k] += a_k
        if np.linalg.norm(xk - xk_tmp) < tol:
            break

    tiempo = time.time() - t0
    return xk, tiempo, iter