import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

nc = Dataset("100_years_1_member.nc")

time = nc.variables["time"][:200].copy()
grid = nc.variables["grid"][:].copy()
data = nc.variables["0"][:200,:].copy()

fig,ax = plt.subplots()

im = ax.pcolor(time,grid,data.transpose(),cmap="PuOr")
ax.set_xlabel("Zeit [d]")
ax.set_ylabel("Gitter")
plt.colorbar(im, label="Variable")
plt.savefig("LorenzOriginal.pdf")
