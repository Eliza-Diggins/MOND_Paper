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

m = np.logspace(np.log10(1e14),np.log10(1e16),100)
a = np.logspace(2,4,100)
M,A = np.meshgrid(m,a)
c = 133
r = np.logspace(1,4,1000)
QP = np.zeros(M.shape)
xq = np.zeros(M.shape)
for i,ms in enumerate(m):
    for j,a_s in enumerate(a):
        rho = (ms * a_s) / (2 * np.pi * r * (r + a_s) ** 3)
        r_500 = np.amin(r[np.where(rho<2500*c)])
        rho = rho[np.where(r==r_500)]
        gamma = (G_mass*ms)/(a_0*(a_s+r_500)**2)

        if gamma > 1:
            #- This can always be managed -#
            alpha = 1
            n = 0
            while n == 0:
                if rho >=(4*np.pi*ms*r_500**3 )/((gamma**alpha + 1)*(a_s+r_500)**2):
                    n = 1
                    xq[i,j] = alpha
                else:
                    alpha += 1
        else:
            if rho>=(4*np.pi*ms*r_500**3 )/((gamma + 1)*(a_s+r_500)**2):
                xq[i,j] = 0
            else:
                xq[i,j] = np.inf


        QP[i,j] = gamma

cmap = plt.cm.plasma.copy()
cmap.set_bad("black")
fig,axes = plt.subplots(nrows=1,ncols=2,sharey=True,gridspec_kw={"wspace":0},figsize=(11,6))
i1 = axes[0].imshow(np.log10(xq),extent=[1e2,1e4,1e14,1e16],origin="lower",cmap=cmap)
axes[0].contour(np.log10(QP),extent=[1e2,1e4,1e14,1e16],levels=[0],colors="red")
axes[0].set_yscale("log")
axes[0].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xscale("log")
axes[0].set_ylabel(r"Hernquist Mass $\left(\mathrm{M}_0\right)$ / [$\mathrm{M}_{\odot}$]")
axes[1].set_xlabel(r"Hernquist Scale Length $\left(a\right)$ / [$\mathrm{kpc}$]")
axes[0].set_xlabel(r"Hernquist Scale Length $\left(a\right)$ / [$\mathrm{kpc}$]")
axes[0].set_yticks([1e14,5e14,1e15,5e15,1e16])
axes[0].set_xticks([1e2,5e2,1e3,5e3])
axes[1].set_yticks([1e14,5e14,1e15,5e15,1e16])
axes[1].set_xticks([1e2,5e2,1e3,5e3])
i2 = axes[1].contourf(np.log10(QP),extent=[1e2,1e4,1e14,1e16],levels=30,cmap=plt.cm.RdBu_r,vmin=-2,vmax=2)
CS = axes[1].contour(np.log10(QP),extent=[1e2,1e4,1e14,1e16],levels=[0],colors="red")
axes[0].text(750,2.8e15,"$\gamma = 0$",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].text(550,1.3e14,r"Clusters Not Viable at $r_{500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].text(550,8e15,r"Clusters Viable at $r_{500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[1].text(750,2.8e15,"$\gamma = 0$",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].set_title(r"Minimum Interpolation $\alpha$ For $\mathcal{R}_v > r_{500}$")
axes[1].set_title(r"Relative Acceleration ($\gamma$)")
plt.suptitle(r"Minimal Interpolation Parameter ($\alpha$) and Associated $\gamma$ For Hernquist Profiles")
cb1= plt.colorbar(i1,ax=axes,fraction=0.05,pad=0.07,label=r"Minimal Interp. Scale / $\left[\log_{10}\left(\alpha\right)\right]$")
cb2 = plt.colorbar(i2,ax=axes,location="right",fraction=0.05,pad=0.0,label=r"Relative Acceleration / $\left[\log_{10}\left(\gamma\right)\right]$")
plt.show()
