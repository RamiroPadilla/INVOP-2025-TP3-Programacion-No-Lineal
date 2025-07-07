#%%
import numpy as np
import time
import matplotlib.pyplot as plt
# 1 - Funcion objetivo (Fermat–Weber)
def funcion_objetivo(x, P, w):
    return np.sum(w * np.linalg.norm(P - x, axis=1))

# 2 - Gradiente de la FO
def gradiente_funcion(x, P, w):
    diff = x - P                          
    normas = np.linalg.norm(diff, axis=1)
    normas = np.where(normas < 1e-12, 1e-12, normas) #  Evito dividir por cero si x == pi
    return np.sum((w[:, None] * diff) / normas[:, None], axis=0)

# 3 - Direccion de descenso por gradiente, usando el criterio de armijo (para el alpha)
# Selecciono alphak 0.5 (entre 0,1 y 0,9 alphak) y c1 para poder cambiarlos si hace falta
def descenso_gradiente_armijo(P, w, x0=None, tol=1e-6, max_iter=5000, alpha0=1.0, alphak=0.5, c1=1e-4):
    t0 = time.time()
    # Inicializo a x
    if x0 is None:
        xk = np.average(P, axis=0, weights=w)
    else:
        xk = x0.copy()
    
    for k in range(1, max_iter+1):
        grad = gradiente_funcion(xk, P, w)
        if np.linalg.norm(grad) < tol:
            break
        
        d = -grad  # ddireccion descenso
        f0 = funcion_objetivo(xk, P, w)
        alpha = alpha0
        
        # itero c/armijo
        while True:
            x_nuevo = xk + alpha * d
            if funcion_objetivo(x_nuevo, P, w) <= f0 + c1 * alpha * grad.dot(d):
                break
            alpha *= alphak
            
        if np.linalg.norm(x_nuevo - xk) < tol:
            break
        
        xk = x_nuevo
    
    f_opt = funcion_objetivo(xk, P, w)
    tiempo = time.time() - t0
    return xk, tiempo, k

#%%
# Ejemplo de uso
datos = np.loadtxt('instancias\distribucion\inst_n2_m5000_clusters.txt', skiprows=2)
P = datos[:, :-1]
w = datos[:, -1]

punto_optimo, tiempo_ejecucion, cant_iteraciones = descenso_gradiente_armijo(P, w)

print("Punto óptimo:", punto_optimo)
print("Tiempo de ejecución:", tiempo_ejecucion)
print("Cantidad de iteraciones:", cant_iteraciones)
# %%
