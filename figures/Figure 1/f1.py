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
for rs, c, cluster in zip(r_disp[::2], cs[::2], clusters[::2]):
    r = np.linspace(rs[0], rs[1], 2000)
    g = gamma(cluster.gas_temperature(r), cluster.gas_density(r), r)

    MM = Mdynamical_mass(g, r, lambda x: x / (1 + x))
    MN = Ndynamical_mass(g, r)

    bcg_density = rho_bcg(r, MN[-1], 30)
    MB = cumulative_trapezoid((cluster.gas_density((r)) + bcg_density) * (4 * np.pi * r ** 2), r, initial=MN[0])

    plt.loglog(r[1:], MB[1:] / MM[1:], color=c, ls=":", alpha=0.7)
    plt.loglog(r[1:], MB[1:] / MN[1:], color=c, ls="-", alpha=0.7, label=cluster.name)

    plt.title("Dynamical Cluster Mass Profiles in MOND and Newtonian Gravity")
    plt.xlabel(r"Cluster Radius / [$\mathrm{kpc}$]")
    plt.ylabel(r"Ratio of Baryonic to Dynamic Mass within $r$ / [$\mathrm{M}_{\odot}$]")
    plt.grid()
    plt.legend(title="Cluster")

plt.show()
