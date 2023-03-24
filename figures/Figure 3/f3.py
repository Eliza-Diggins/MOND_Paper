import matplotlib.pyplot as plt
from core import *
import numpy as np
import matplotlib as mpl
from scipy.optimize import fsolve

mpl.rcParams["text.usetex"] = True
mpl.rcParams[
    "text.latex.preamble"] = r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"

m = np.logspace(14,16,200)
a = np.logspace(2,4,200)
R = np.logspace(1,4,1000)

RT = np.zeros((a.size,m.size))
for j,ms in enumerate(m):
    for i,AS in enumerate(a):
        z = 2*(1-(R/(AS+R))) - (1/(((G_mass*ms)/(a_0*(AS+R)**2))+1))
        try:
            RM = R[np.where(z<0)][0]
        except:
            RM = R[-1]

        RT[i,j] = RM

c=plt.imshow(np.log10(RT),extent=[1e2,1e4,1e14,1e16],origin="lower",cmap=plt.cm.binary_r)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.binary_r,norm=mpl.colors.LogNorm(vmin=np.amin(RT),vmax=np.amax(RT))),pad=0.01,fraction=0.05,label=r"$\mathcal{R}_v$ / [$\mathrm{kpc}$]")
CS = plt.contour(np.log10(RT),extent=[1e2,1e4,1e14,1e16],colors="red",levels=np.log10(np.array([275,300,400,600,800,1200,1500])),linestyles=":")
plt.xlabel(r"Hernquist Scale Length ($a$) / [$\mathrm{kpc}$]")
plt.ylabel(r"Hernquist Scale Mass ($M_0$) / [$\mathrm{M}_{\odot}$]")
plt.suptitle("Maximal Viable Radii $\mathcal{R}_v$ For Hernquist Profiles")
plt.title(r"$\left(\alpha = 1\right)$")
plt.xscale("log")
fmt = {k:r"%s $\mathrm{kpc}$"%np.round(10**k,decimals=0) for k in CS.levels}
cs = plt.gca().clabel(CS, CS.levels, inline=False, fontsize=8,fmt=fmt,colors="black")
cs = [c.set_bbox({"ec": "k", "fc": "white", "alpha": 0.5}) for c in cs]
plt.yscale("log")
plt.show()