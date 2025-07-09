#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#%%
df = pd.read_csv('../resultados_dimensional.csv')

df = df.sort_values('dimension')

cmap = cm.get_cmap('viridis', 3)

for i, algo in enumerate(["weiszfeld", "desc_coord", "desc_grad_armijo"]):
    datos_alg = df[df['algoritmo'] == algo]
    plt.plot(datos_alg['dimension'], datos_alg['tiempo_computo'], color=cmap(i),
             marker='o', label=algo)

plt.xlabel('Dimensión')
plt.ylabel('Tiempo de cómputo (s)')
plt.xticks(np.arange(5, 51, 5))
plt.yticks(np.arange(0, 91, 10)) 
plt.legend(title='Algoritmo')
plt.grid(True)
plt.tight_layout()
plt.savefig(fname = f'grafico_dimensiones_cluster',dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()

# %%
df = pd.read_csv('../resultados_lineal.csv')

df = df.sort_values('dimension')

cmap = cm.get_cmap('viridis', 3)

for i, algo in enumerate(["weiszfeld", "desc_coord", "desc_grad_armijo"]):
    datos_alg = df[df['algoritmo'] == algo]
    plt.plot(datos_alg['dimension'], datos_alg['tiempo_computo'], color=cmap(i),
             marker='o', label=algo)

plt.xlabel('Dimensión')
plt.ylabel('Tiempo de cómputo (s)')
plt.xticks(np.arange(5, 51, 5))
plt.legend(title='Algoritmo')
plt.grid(True)
plt.tight_layout()
plt.savefig(fname = f'grafico_dimensiones_lineal',dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()
#
# %%
df = pd.read_csv('../resultados_uniforme.csv')

df = df.sort_values('dimension')

cmap = cm.get_cmap('viridis', 3)

for i, algo in enumerate(["weiszfeld", "desc_coord", "desc_grad_armijo"]):
    datos_alg = df[df['algoritmo'] == algo]
    plt.plot(datos_alg['dimension'], datos_alg['tiempo_computo'], color=cmap(i),
             marker='o', label=algo)

plt.xlabel('Dimensión')
plt.ylabel('Tiempo de cómputo (s)')
plt.xticks(np.arange(5, 51, 5))
plt.legend(title='Algoritmo')
plt.grid(True)
plt.tight_layout()
plt.savefig(fname = f'grafico_dimensiones_uniforme',dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()
# %%
