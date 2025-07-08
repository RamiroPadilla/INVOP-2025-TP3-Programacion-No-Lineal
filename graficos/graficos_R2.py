# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})
#%%

datos = np.loadtxt('..\instancias\distribucion\inst_dim2_n500_clusters.txt', skiprows=2)
puntos = datos[:, :2]
pesos = datos[:, 2]

plt.scatter(puntos[:, 0], puntos[:, 1], s=8, alpha=0.6, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname=f'{n}_pts_cluster', dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()

# %%

datos = np.loadtxt('..\instancias/distribucion/inst_dim2_n500_lineal.txt', skiprows=2)
puntos = datos[:, :2]
pesos = datos[:, 2]

plt.scatter(puntos[:, 0], puntos[:, 1], s=8, alpha=0.6, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname = f'{n}_pts_lineal',dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()
# %%

datos = np.loadtxt('..\instancias/distribucion/inst_dim2_n500_uniforme.txt', skiprows=2)
puntos = datos[:, :2]
pesos = datos[:, 2]

plt.scatter(puntos[:, 0], puntos[:, 1], s=8, alpha=0.6, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname = f'{n}_pts_uniforme',dpi=250, bbox_inches='tight', pad_inches=0.2)
plt.close()
# %%
