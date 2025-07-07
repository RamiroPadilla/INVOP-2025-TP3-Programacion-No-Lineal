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

plt.scatter(puntos[:, 0], puntos[:, 1], s=5, alpha=0.5, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.title(f'{n} puntos con distribucion en clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname = f'{n}_pts_cluster',dpi=200)
plt.close()

# %%

datos = np.loadtxt('..\instancias/distribucion/inst_dim2_n500_lineal.txt', skiprows=2)
puntos = datos[:, :2]
pesos = datos[:, 2]

plt.scatter(puntos[:, 0], puntos[:, 1], s=8, alpha=0.5, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.title(f'{n} puntos con distribucion lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname = f'{n}_pts_lineal',dpi=200)
plt.close()
# %%

datos = np.loadtxt('..\instancias/distribucion/inst_dim2_n500_uniforme.txt', skiprows=2)
puntos = datos[:, :2]
pesos = datos[:, 2]

plt.scatter(puntos[:, 0], puntos[:, 1], s=8, alpha=0.5, c='purple')
plt.xlim(-65000, 65000)
plt.ylim(-65000, 65000)

n = len(puntos)
plt.title(f'{n} puntos con distribucion uniforme')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig(fname = f'{n}_pts_uniforme',dpi=200)
plt.close()
# %%
