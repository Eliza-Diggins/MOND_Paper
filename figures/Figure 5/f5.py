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

m = np.logspace(np.log10(1e14),np.log10(1e16),400)
a = np.logspace(2,4,400)
M,A = np.meshgrid(m,a)
c = 133
r = np.logspace(1,4,1000)
QP_500 = np.zeros(M.shape)
xq_500 = np.zeros(M.shape)
QP_200 = np.zeros(M.shape)
xq_200 = np.zeros(M.shape)
QP_1000 = np.zeros(M.shape)
xq_1000 = np.zeros(M.shape)
QP_2500 = np.zeros(M.shape)
xq_2500 = np.zeros(M.shape)
for i,ms in enumerate(m):
    print(i)
    for j,a_s in enumerate(a):
        rho = (ms * a_s) / (2 * np.pi * r * (r + a_s) ** 3)
        r_200,r_500,r_1000,r_2500 = np.amin(r[np.where(rho<200*c)]),np.amin(r[np.where(rho<500*c)]),np.amin(r[np.where(rho<1000*c)]),np.amin(r[np.where(rho<2500*c)])
        rhos = [rho[np.where(r==i)] for i in [r_200,r_500,r_1000,r_2500]]
        gammas = [(G_mass*ms)/(a_0*(a_s+i)**2) for i in [r_200,r_500,r_1000,r_2500]]

        for gamma,rs,rho_i,p,Q in zip(gammas,[r_200,r_500,r_1000,r_2500],
                                     rhos,
                                    [xq_200,xq_500,xq_1000,xq_2500],
                                     [QP_200,QP_500,QP_1000,QP_2500]):
            if gamma > 1:
                #- This can always be managed -#
                alpha = 1
                n = 0
                while n == 0:
                    if rho_i >=(4*np.pi*ms*rs**3 )/((gamma**alpha + 1)*(a_s+rs)**2):
                        n = 1
                        p[i,j] = alpha
                    else:
                        alpha += 1
            else:
                if rho_i>=(4*np.pi*ms*rs**3 )/((gamma + 1)*(a_s+rs)**2):
                    p[i,j] = 0
                else:
                    p[i,j] = np.inf


            Q[i,j] = gamma

cmap = plt.cm.plasma.copy()
cmap.set_bad("black")
vmin_Q = np.amin([np.amin(np.log10(i[np.where(i != np.inf)])) for i in [QP_200,QP_500,QP_1000,QP_2500]])
vmax_Q = np.amax([np.amax(np.log10(i[np.where(i != np.inf)])) for i in [QP_200,QP_500,QP_1000,QP_2500]])
vmin_P = np.amin([np.amin(np.log10(i[np.where(i != np.inf)])) for i in [xq_200,xq_500,xq_1000,xq_2500]])
vmax_P = np.amax([np.amax(np.log10(i[np.where(i != np.inf)])) for i in [xq_200,xq_500,xq_1000,xq_2500]])

PNorm = mpl.colors.Normalize(vmin=vmin_P,vmax=vmax_P)
QNorm = mpl.colors.Normalize(vmin=-2,vmax=2)
fig,axes = plt.subplots(nrows=4,ncols=2,sharey=True,sharex=True,gridspec_kw={"wspace":0,"hspace":0},figsize=(7.5,12))

for i,data in enumerate(zip([QP_200,QP_500,QP_1000,QP_2500],
               [xq_200,xq_500,xq_1000,xq_2500])):
    #- Left plots -#
    X,P = data
    i1 = axes[i,0].imshow(np.log10(P),extent=[1e2,1e4,1e14,1e16],origin="lower",cmap=cmap)
    axes[i,0].contour(np.log10(X), extent=[1e2, 1e4, 1e14, 1e16], levels=[0], colors="red")
    #- Right Plot -#
    i2 = axes[i,1].contourf(np.log10(X), extent=[1e2, 1e4, 1e14, 1e16], levels=30, cmap=plt.cm.RdBu_r, vmin=-2,
                          vmax=2)
    CS = axes[i,1].contour(np.log10(X), extent=[1e2, 1e4, 1e14, 1e16], levels=[0], colors="red")

    #- Ticks and extra -#
    axes[i,0].text(2e3,1.3e14,[r"$\mathcal{R}_v \ge r_{200}$",r"$\mathcal{R}_v \ge r_{500}$",r"$\mathcal{R}_v \ge r_{1000}$",r"$\mathcal{R}_v \ge r_{2500}$"][i],bbox={"ec": "k", "fc": "white", "alpha": 0.8})
    axes[i,0].set_yticks([1e14, 5e14, 1e15, 5e15])
    axes[i,0].set_xticks([1e2, 5e2, 1e3, 5e3])
    axes[i,1].set_yticks([1e14, 5e14, 1e15, 5e15])
    axes[i,1].set_xticks([1e2, 5e2, 1e3, 5e3])
    axes[i,0].set_yscale("log")
    axes[i,0].set_xscale("log")
    axes[i,1].set_yscale("log")
    axes[i,1].set_xscale("log")
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=PNorm,cmap=plt.cm.plasma), ax=axes[:2,:], fraction=0.05, pad=0.0,
                       label=r"Minimal Interp. Scale / $\left[\log_{10}\left(\alpha\right)\right]$")
cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=QNorm,cmap=plt.cm.coolwarm), ax=axes[2:,:], location="right", fraction=0.05, pad=0.0,
                       label=r"Relative Acceleration / $\left[\log_{10}\left(\gamma\right)\right]$")
axes[0,0].text(130,1.3e14,"Clusters Not Viable at $r_{200}$." ,bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="red")
axes[1,0].text(130,1.3e14,"Clusters Not Viable at $r_{500}$." ,bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="red")
axes[2,0].text(130,1.3e14,"Clusters Not Viable at $r_{1000}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="red")
axes[3,0].text(1.7e3,2.1e14,"Clusters Not Viable\n at $r_{2500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8,color="red")
axes[0,0].text(130,6e15,"Clusters Viable at $r_{200}$." ,bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="green")
axes[1,0].text(130,6e15,"Clusters Viable at $r_{500}$." ,bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="green")
axes[2,0].text(130,6e15,"Clusters Viable at $r_{1000}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8    ,color="green")
axes[3,0].text(130,6e15,"Clusters Viable at $r_{2500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8,    color="green")
axes[0,1].text(200,4.2e15,r"$\gamma = 1$",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8)
axes[1,1].text(2e3,4e15,r"$\gamma = 1$",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8)
axes[2,1].text(2e3,4e15,r"$\gamma = 1$",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8)
axes[3,1].text(3e3,4e15,r"$\gamma = 1$",bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8)
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("common X")
plt.ylabel("common Y")
plt.ylabel(r"Hernquist Mass $\left(\mathrm{M}_0\right)$ / [$\mathrm{M}_{\odot}$]")
plt.xlabel(r"Hernquist Scale Length $\left(a\right)$ / [$\mathrm{kpc}$]")
axes[0,0].set_title(r"Minimum Interpolation $\alpha$ For $\mathcal{R}_v > r_{i}$")
axes[0,1].set_title(r"Relative Acceleration ($\gamma$)")
plt.suptitle(r"Minimal Interpolation Parameter ($\alpha$) and Associated $\gamma$ For Hernquist Profiles")
plt.subplots_adjust(bottom=0.048,right=0.84,left=0.125,top=0.94)
plt.savefig("test.png",dpi=400)
exit()
i1 = axes[0].imshow(np.log10(xq_500),extent=[1e2,1e4,1e14,1e16],origin="lower",cmap=cmap)
axes[0].contour(np.log10(QP_500),extent=[1e2,1e4,1e14,1e16],levels=[0],colors="red")
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
i2 = axes[1].contourf(np.log10(QP_500),extent=[1e2,1e4,1e14,1e16],levels=30,cmap=plt.cm.RdBu_r,vmin=-2,vmax=2)
CS = axes[1].contour(np.log10(QP_500),extent=[1e2,1e4,1e14,1e16],levels=[0],colors="red")
axes[0].text(750,2.8e15,"$\gamma = 0$",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].text(550,1.3e14,r"Clusters Not Viable at $r_{500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].text(550,8e15,r"Clusters Viable at $r_{500}$.",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[1].text(750,2.8e15,"$\gamma = 0$",bbox={"ec": "k", "fc": "white", "alpha": 0.8})
axes[0].set_title(r"Minimum Interpolation $\alpha$ For $\mathcal{R}_v > r_{500}$")
axes[1].set_title(r"Relative Acceleration ($\gamma$)")
plt.suptitle(r"Minimal Interpolation Parameter ($\alpha$) and Associated $\gamma$ For Hernquist Profiles")
cb1= plt.colorbar(i1,ax=axes,fraction=0.05,pad=0.07,label=r"Minimal Interp. Scale / $\left[\log_{10}\left(\alpha\right)\right]$")
cb2 = plt.colorbar(i2,ax=axes,location="right",fraction=0.05,pad=0.0,label=r"Relative Acceleration / $\left[\log_{10}\left(\gamma\right)\right]$")
plt.savefig("test.png")