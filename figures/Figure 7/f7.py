from core import *
from itertools import product
clusters = load_clusters()
r_disp = [(90, 1000), (20, 800), (60, 1000), (60, 2000), (60, 1200), (20, 1100), (80, 1100), (20, 1100),
          (30, 2000), (90, 3000), (10, 1000), (10, 1000), (40, 300)]
cs = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#f46a9b",
      "#e60049", "#0bb4ff", "#50e991"]
pos = product(np.arange(0,4),np.arange(0,3))
mpl.rcParams["text.usetex"] = True
mpl.rcParams[
    "text.latex.preamble"] = r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
fig,axes = plt.subplots(nrows=4,ncols=3,sharex=True,sharey=True,gridspec_kw={"hspace":0,"wspace":0},figsize=(8.5,11))
for cluster, rs, c,p in zip(clusters, r_disp, cs,pos):
    i,j = p
    print(i,j)
    r = np.linspace(10, 4 * rs[1], 2000)
    gt, gd = cluster.gas_temperature(r), cluster.gas_density(r)

    Mg = 4 * np.pi * cumulative_trapezoid(gd * r ** 2, r, initial=0)
    MN = Ndynamical_mass(gamma(gt, gd, r), r)
    MM = Mdynamical_mass(gamma(gt, gd, r), r, lambda x: x / (1 + x))
    Mdm = MN - Mg
    MMdm = MM - Mg
    dMdm = np.gradient(Mdm, r)
    MMdm[np.where(dMdm < 0)] = 0

    T = get_temperature_profile(Mdm, Mg, r, lambda x: x / (1 + x), T0=gt[-1])
    T2 = get_temperature_profile(np.zeros(Mg.shape), Mg, r, lambda x: x / (1 + x), mode="MOND")
    T3 =get_temperature_profile(MMdm, Mg, r, lambda x: x / (1 + x), mode="MOND")



    #- Gas temp -#
    axes[i, j].semilogx(r[np.where(r >= rs[1])], gt[np.where(r >= rs[1])], color="k", ls=":"  ,lw=1.5)
    axes[i, j].semilogx(r[np.where((r <= rs[1])&(r>=rs[0]))], gt[np.where((r <= rs[1])&(r>=rs[0]))], color="k", ls="-"  ,lw=4)
    axes[i, j].semilogx(r[np.where(r <= rs[0])], gt[np.where(r <= rs[0])], color="k", ls=":"  ,lw=1.5)

    #- Newtonian Solution -#
    axes[i, j].semilogx(r[np.where(r >= rs[1])], T[np.where(r >= rs[1])], color=cs[0], ls=":",lw=1.5)
    axes[i, j].semilogx(r[np.where((r <= rs[1])&(r>=rs[0]))], T[np.where((r <= rs[1])&(r>=rs[0]))], color=cs[0], ls="-",lw=2)
    axes[i, j].semilogx(r[np.where(r <= rs[0])], T[np.where(r <= rs[0])], color=cs[0], ls=":",lw=1.5)

    #- MOND SOLO -#
    axes[i, j].semilogx(r[np.where(r >= rs[1])], T2[np.where(r >= rs[1])], color=cs[1], ls=":",lw=1.5)
    axes[i, j].semilogx(r[np.where((r <= rs[1])&(r>=rs[0]))], T2[np.where((r <= rs[1])&(r>=rs[0]))], color=cs[1], ls="-",lw=1)
    axes[i, j].semilogx(r[np.where(r <= rs[0])], T2[np.where(r <= rs[0])], color=cs[1], ls=":",lw=1.5)

    #- MOND -#
    axes[i, j].semilogx(r[np.where(r >= rs[1])], T3[np.where(r >= rs[1])], color=cs[2], ls=":",lw=1.5)
    axes[i, j].semilogx(r[np.where((r <= rs[1])&(r>=rs[0]))], T3[np.where((r <= rs[1])&(r>=rs[0]))], color=cs[2], ls="-",lw=2.5,alpha=0.8)
    axes[i, j].semilogx(r[np.where(r <= rs[0])], T3[np.where(r <= rs[0])], color=cs[2], ls=":",lw=1.5)
    axes[i,j].grid()
    axes[i,j].text(100,10,cluster.name,bbox={"ec": "k", "fc": "white", "alpha": 0.8},fontsize=8)

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Cluster Radius / $\left[\mathrm{kpc}\right]$")
plt.ylabel(r"Gas Temperature / $\left[\mathrm{keV}\right]$")
plt.title(r"Gas Temperature Solutions In Different Paradigms")

legend_handles = [Line2D([],[],color=cs[2],lw=2.5,label="MOND + DM"),
                  Line2D([],[],color="k",lw=4,label="Observed"),
                  Line2D([],[],color=cs[1],lw=1,label="MOND"),
                  Line2D([],[],color=cs[0],lw=2,label="CDM"),
                  Line2D([],[],color="k",lw=1,ls=":",label="Outside of Fit Zone"),
                  Line2D([],[],color="k",lw=1,ls="-",label="Inside of Fit Zone")]
plt.legend(legend_handles,["MOND + DM","Observed","MOND","CDM","Outside of Fit Zone","Inside of Fit Zone"],bbox_to_anchor=(0.0,-0.08,1.0,0.01),mode="expand",ncol=4)

plt.show()