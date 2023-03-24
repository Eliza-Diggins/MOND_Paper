from core import *

mpl.rcParams["text.usetex"] = True
mpl.rcParams[
    "text.latex.preamble"] = r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"

clusters = load_clusters()
r_disp = [(90, 1000), (20, 800), (60, 1000), (60, 2000), (60, 1200), (20, 1100), (80, 1100), (20, 1100),
          (30, 2000), (90, 3000), (10, 1000), (10, 1000), (40, 300)]
cs = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#f46a9b",
      "#e60049", "#0bb4ff", "#50e991"]

for id, p in enumerate(zip(clusters, r_disp, cs)):
    cluster, rd, cs = p
    r = np.linspace(rd[0], 5000, 4000)
    T, rho = cluster.gas_temperature(r), cluster.gas_density(r)
    M, N = Mdynamical_mass(gamma(T, rho, r), r, lambda x: x / (1 + x)), Ndynamical_mass(gamma(T, rho, r), r)
    try:
        rm = r[np.where(np.gradient(N, r) < 0)][0]
    except:
        rm = 5000

    r = np.linspace(rd[0], rm, 4000)
    T, rho = cluster.gas_temperature(r), cluster.gas_density(r)
    M, N = Mdynamical_mass(gamma(T, rho, r), r, lambda x: x / (1 + x)), Ndynamical_mass(gamma(T, rho, r), r)
    rM, rN = (1 / (4 * np.pi * (r ** 2))) * np.gradient(M, r), (1 / (4 * np.pi * (r ** 2))) * np.gradient(N, r)
    rB = cluster.gas_density(r)
    diff = rM - rB

    r500 = r[np.where(rN >= 500 * 133)][-1]
    r200 = r[np.where(rN >= 200 * 133)][-1]
    print(r[np.where(diff < 0)][0])
    rx_max = np.amin(r[np.where(diff < 0)])
    rt_max = np.amin(r[np.where(np.gradient(np.log(N), r) < 1 / (r * ((G_mass * N / (a_0 * r ** 2)) + 1)))])
    plt.bar(id + 1, rm - rd[0], 0.8, rd[0], log=True, alpha=0.2, color=cs)
    plt.semilogy([id + 1, id + 1], [rx_max, rt_max], c=cs)
    plt.semilogy([id + 0.75, id + 1.25], [rx_max, rx_max], c=cs)
    plt.semilogy([id + 0.75, id + 1.25], [rt_max, rt_max], c=cs)
    plt.semilogy([id + 0.5, id + 1.5], [r200, r200], "k-.")
    plt.semilogy([id + 0.5, id + 1.5], [r500, r500], "k:")

plt.title(r"Best Case and Observed Maximal Viable Radius $\mathcal{R}_v$")
plt.ylabel(r"Cluster Radius / $[\mathrm{kpc}]$")
plt.xlabel(r"Cluster")
plt.xticks([i + 1 for i in range(len(clusters))], [cluster.name for cluster in clusters], rotation="vertical")

# - Custom Legends -#
b1 = LineCollection(np.empty((2, 2, 2)), colors="k")
line = Line2D([], [], ls="", c="k")
cl = [Line2D([], [], color="k", ls=":"),
      Line2D([], [], color="k", ls="-."),
      ErrorbarContainer((line, [line], [b1]), has_yerr=True),
      Rectangle((0, 0), 0, 0, color="black", alpha=0.25)]

plt.legend(cl, [r"$\mathrm{r}_{200}$", r"$\mathrm{r}_{500}$", r"Best Case and Observed $\mathcal{R}_v$",
                "Fit Range"])

plt.show()
# ax.loglog(r, np.gradient(N-MB,r) / (4 * np.pi * r ** 2))
# ax.loglog(r, -np.gradient(N-MB,r) / (4 * np.pi * r ** 2),"r")
# ax.loglog(r,(a_0/(2*np.pi*G_mass))/r)