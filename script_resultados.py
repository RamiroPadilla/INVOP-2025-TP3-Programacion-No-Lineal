import os
import numpy as np
import pandas as pd

from weiszfeld import weiszfeld
from descenso_coordenado import resolver_descenso_coordenado
from direccion_descenso_armijo import descenso_gradiente_armijo

def resultados(ruta_carpeta): 
    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.txt')]
    resultados = []

    for txt in archivos:
        ruta_instancia = os.path.join(ruta_carpeta, txt)
        
        datos = np.loadtxt(ruta_instancia, skiprows=2)
        P = datos[:, :-1]
        w = datos[:, -1]  

        p_optimo_weiszfeld, t_weisz, it_weisz = weiszfeld(P, w)
        resultados.append({
            'nombre_instancia': txt,
            'algoritmo': 'weiszfeld',
            'tiempo_computo': t_weisz,
            'cant_iteraciones': it_weisz,
            'punto_optimo' : p_optimo_weiszfeld
        })

        p_optimo_desc_coord, t_desc_coord, it_desc_coord = resolver_descenso_coordenado(P, w)
        resultados.append({
            'nombre_instancia': txt,
            'algoritmo': 'desc_coord',
            'tiempo_computo': t_desc_coord,
            'cant_iteraciones': it_desc_coord,
            'punto_optimo' : p_optimo_desc_coord
        })
        
        p_optimo_desc_grad, t_desc_grad, it_desc_grad = descenso_gradiente_armijo(P, w)
        resultados.append({
            'nombre_instancia': txt,
            'algoritmo': 'desc_grad_armijo',
            'tiempo_computo': t_desc_grad,
            'cant_iteraciones': it_desc_grad,
            'punto_optimo' : p_optimo_desc_grad
        })

    df = pd.DataFrame(resultados)
    tipo = os.path.basename(ruta_carpeta)
    df.to_csv(f'resultados_{tipo}.csv', index=False)


def main():
    ruta_dis = 'instancias\distribucion'
    resultados(ruta_dis)

    ruta_dim = 'instancias\dimensional'
    resultados(ruta_dim)

if __name__ == "__main__":
    main()