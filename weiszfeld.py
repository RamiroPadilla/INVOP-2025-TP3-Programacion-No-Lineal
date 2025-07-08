import numpy as np
import time

#funcion objetivo
def f(x, P, w): 
    return np.sum(w * np.linalg.norm(P - x, axis=1))

#funcion gradiente
def Rj(pj, P, w, j):
    # R(p_j) = ∑_{i ≠ j} w_i * (p_j - p_i) / ||p_j - p_i||
    
    mask = np.arange(len(P)) != j # Saco el punto j

    resta = pj - P[mask] # Resto el punto pj a los demás puntos
    
    normas = np.linalg.norm(resta, axis=1)
    normas = np.where(normas < 1e-12, 1e-12, normas)  # Para evitar división por cero, por si pj y pi estan muy cerca
    
    return np.sum((w[mask, None] * resta) / normas[:, None], axis=0)

# operador S
def S(pj, P, w, j):
    #Modificación 2: Paso S(p_j), cuando x^(k) = p_j y ||R_j|| > w_j
    # S(pj) = p_j + d_j * t_j

    R_j = Rj(pj, P, w, j)
    norma_R_j = np.linalg.norm(R_j)

    if norma_R_j <= w[j]: #Condicion de optimalidad
        return pj
    
    #Si no es optimo:
    dj = -R_j / norma_R_j  # Direccion del gradiente

    mask = np.arange(len(P)) != j  
    normas = np.linalg.norm(pj - P[mask], axis=1)
    normas = np.where(normas < 1e-12, 1e-12, normas)

    den = np.sum(w[mask] / normas)
    tj = (norma_R_j - w[j]) / den  # Paso de Weiszfeld

    return pj + dj * tj  # Actualizo el punto pj


#funcion T; paso de Weiszfeld
def T(xk, P, w):
    #Version estandar, sin modificaciones
    resta = xk - P
    normas = np.linalg.norm(resta, axis=1)
    normas = np.where(normas < 1e-12, 1e-12, normas)
    num = np.sum((w[:, None] * P) / normas[:, None], axis=0)
    den = np.sum(w / normas)
    return num / den

# T̂(x^(k))
def T_somb(xk, P, w):
    #Modificación 1: 
    # T̂(x^(k)) =    T(x^(k))  si x^(k) ∉ P
    #               pⱼ        si x^(k) = pⱼ (1 ≤ j ≤ m) y ||Rⱼ|| ≤ wⱼ
    #               S(pⱼ)     si x^(k) = pⱼ (1 ≤ j ≤ m) y ||Rⱼ|| > wⱼ
    
    for j in range(len(P)):
        pj = P[j]
        if np.linalg.norm(xk - pj) < 1e-12: # Si xk es igual a pj
            R_j = Rj(P[j], P, w, j)

            if np.linalg.norm(R_j) <= w[j]: # Condición de optimalidad
                return pj # Si es óptimo, devuelvo pj
            
            else:
                # Si no es optimo, aplico S(pj)
                spj = S(pj, P, w, j)
                if f(spj, P, w) < f(xk, P, w):
                    return spj
                else:
                    return pj
                
    return T(xk, P, w) # Sino, xk no es igual a pj
            

# Algoritmo de Weiszfeld
def weiszfeld(P, w, tol=1e-6, max_iter=5000):
    #Con ambas modificaciones
    # Devuelve el punto optimo y el tiempo de ejecución

    m, n = P.shape
    w = w.reshape(-1)
    t0 = time.time()
    iteraciones = 0 
    
    # --------- Modificación 2: Starting Point ----------
    for j in range(m):
        R_j = Rj(P[j], P, w, j)
        if np.linalg.norm(R_j) <= w[j]: #Cndicion de optimalidad
            tiempo = time.time() - t0
            return P[j], tiempo, iteraciones
    # Ningun pj es optimo
   
    j_mejor = np.argmin([f(P[j], P, w) for j in range(m)])
    xk = S(P[j_mejor], P, w, j_mejor)  # Punto inicial usando S(p_j)

    # ----------------------------------------------------

    for k in range(max_iter):
        iteraciones += 1
        # ----------- Modificación 1: T̂(xk) ------------
        xk_nuevo = T_somb(xk, P, w)
        if np.linalg.norm(xk_nuevo - xk) < tol:
            tiempo = time.time() - t0

            return xk_nuevo, tiempo, iteraciones  
        # ----------------------------------------------

        xk = xk_nuevo

    tiempo = time.time() - t0
    return xk, tiempo, iteraciones

#%%
# """
# # Ejemplo de uso
# if __name__ == "__main__":
#     ruta_archivo = "INVOP-2025-TP3-Programacion-No-Lineal\instancias\dimensional\inst_n2_m20_uniforme_2.txt" 
#     P, w = leer_instancia(ruta_archivo)
    
#     punto_optimo, tiempo_ejecucion, cant_iteraciones = weiszfeld(P, w)
    
#     print("Punto óptimo:", punto_optimo)
#     print("Tiempo de ejecución:", tiempo_ejecucion)
#     print("Cantidad de iteraciones:", cant_iteraciones)
#     print("Valor de la función objetivo:", f(punto_optimo, P, w))
# """
