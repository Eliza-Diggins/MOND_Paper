from core import *
clusters = load_clusters()
r_disp = [(90, 1000), (20, 800), (60, 1000), (60, 2000), (60, 1200), (20, 1100), (80, 1100), (20, 1100),
          (30, 2000), (90, 3000), (10, 1000), (10, 1000), (40, 300)]
cs = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#f46a9b",
      "#e60049", "#0bb4ff", "#50e991"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams[
    "text.latex.preamble"] = r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={"hspace":0,"height_ratios":[0.8,0.2]},figsize=(5,6))

for cluster, rs, c in zip(clusters, r_disp, cs):
    r = np.linspace(rs[0],rs[1],1000)
    y1,y2 = cluster.nfw(r),cluster.gas_density(r)
    axes[0].loglog(r,y1,color=c,ls=":")
    axes[0].loglog(r,y2,label=cluster.name,color=c)

    res = (y1-y2)/y2

    axes[1].semilogx(r[::10],res[::10],color=c,ls="-.",lw=1)
    axes[1].semilogx(r,np.zeros(r.size),color="k")

axes[0].set_ylabel(r"Gas Density / $\left[\mathrm{M}_\odot \; \mathrm{kpc}^{-3}\right]$")
axes[1].set_xlabel(r"Cluster Radius / $\left[\mathrm{kpc}\right]$")
axes[1].set_ylim([-1,1])
axes[1].set_xlim([10,3000])
axes[1].set_ylabel("Scaled Res.")
axes[0].set_title(r"NFW Profile Fits to $\rho_g(r)$")
axes[0].legend(fontsize=8)
plt.savefig("Figure_6.png")