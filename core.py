"""

Core functions for the project.

"""
# Standard Units: Msol,kpc,keV
#
#
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import matplotlib as mpl
import os
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
#----------------------------------------------------------------------------------------------------------------------#
#   Constants                                                                                                          #
#----------------------------------------------------------------------------------------------------------------------#
proton_mass = 1.672621911e-27 # KG
G = 6.674e-11
G_mass = 1.39367222e-19
a_0 = 1.2e-10
#- Conversions -#
inv_cmToinv_kpc = 3.08546e21
kgTomsol = 5.029806e-31
eta = 0.6
keVToWeird = 5.1911860012961762799740764744005184705119896305897602073882e-36
#----------------------------------------------------------------------------------------------------------------------#
#   Classes                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------#
class Cluster:
    def __init__(self,name,parameters=None):
        self.name = name

        if parameters:
            self.parameters = parameters

            for k,v in self.parameters.items():
                setattr(self,k,v)

    def __repr__(self):
        return "Cluster %s"%self.name
    def __str__(self):
        return "Cluster %s"%self.name

    def gas_temperature(self,r):
        params = ['T_0','a','b','c','r_t','r_cool','a_cool','T_min']


        return gas_temperature(r,*[getattr(self,p) for p in params])
    def gas_density(self,r):
        params = ['r_c','n_0','ALPHA','BETA','r_s','GAMMA','EPSILON','n_02','r_c2','BETA_2']

        return gas_density(r,*[getattr(self,p) for p in params])

#----------------------------------------------------------------------------------------------------------------------#
#   Functions                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#
def load_clusters():
    os.chdir("C:\\Users\\13852\\PycharmProjects\\MOND_Paper")
    df = pd.read_csv("datasets/Vikhlinin.csv")

    clusters = []
    for cluster in df["Cluster"]:
        clusters.append(Cluster(cluster,parameters={col.split(",")[0]:df.loc[df["Cluster"] == cluster,col].values[0] for col in df.columns[1:]}))

    return clusters
#----------------------------------------------------------------------------------------------------------------------#
#   Profiles                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#
def gas_temperature(r,T_0,a,b,c,r_t,r_cool,a_cool,T_min):
    return (T_0 * (
            ((r / r_t) ** (-a)) / (
            (1 + ((r / r_t) ** (b))) ** (c / b))) * ((((r / r_cool) ** (a_cool)) + (T_min / T_0)) / (((r / r_cool) ** (a_cool)) + 1)))

def gas_density(r,rc,n_0,alpha,beta,rs,gamma,epsilon,n_02,rc2,beta2):
    return 1.252 * proton_mass*kgTomsol * np.sqrt((((((n_0*inv_cmToinv_kpc**3) ** 2) * (
            (r / rc) ** (-alpha)) / ((1 + (
            r / rc) ** 2) ** ((3 * beta) - (alpha / 2)))) * ( 1 / ((1 + ((r /rs) ** (gamma))) ** (
                                                                             epsilon /
                                                                             gamma)))) + (
                                                                    (n_02*inv_cmToinv_kpc**3)**2 / (
                                                                    (1 + ((r / rc2) ** 2)) ** (3 * beta2)))))

def gamma(T,rho,r):
    dT,drho = (1/T)*np.gradient(T,r),(1/rho)*np.gradient(rho,r)

    return -keVToWeird*(T/(proton_mass*eta))*(dT + drho)

def Ndynamical_mass(gamma,r):
    return ((r**2)*gamma/G_mass)

def Mdynamical_mass(gamma,r,interp):
    return  ((r**2)*gamma/G_mass)*interp(np.abs(gamma)/a_0)

def rho_bcg(r,M,h):
    return (5.3e11*(M/1e14)**(0.42))*h/(2*np.pi*r*(r+h)**3)
def get_temperature_profile(m_dm,
                            m_g,
                            r,
                                interpolation_function,
                                mode="Newtonian",
                                 independent_units=None,
                                 dependent_units=None,
                                 output_units=None,
                                 sample_frequency=1
                                 ):
    #- Quantity Calculations -#
    m_tot = m_g+m_dm
    rho_dm,rho_g,rho = (1/(4*np.pi*r**2))*np.gradient(m_dm,r), (1/(4*np.pi*r**2))*np.gradient(m_g,r), (1/(4*np.pi*r**2))*np.gradient(m_tot,r)

    #------------------------------------------------------------------------------------------------------------------#
    #  Computing the gravitational field strength
    #------------------------------------------------------------------------------------------------------------------#
    # Notes: We utilize that dphi/a_0 = x, then we need to solve eta(x)*x = -G/(r^2a_0) * M(<r)
    #
    #
    #------------------------------------------------------------------------------------------------------------------#
    #- Unit Manipulation -#
    G_unit = G_mass
    a_0 = 1.2e-10

    if mode == "MOND":
        #- Setting up the solver -#
        solving_array = r[::sample_frequency]
        solving_mass = m_tot[::sample_frequency]
        solver_func = lambda x: interpolation_function(np.sqrt(x ** 2 + 1e-5)) * x + (G_unit / (a_0 * solving_array ** 2)) * solving_mass

        #- Solving -#
        alph = (G_unit/(a_0*solving_array**2))*solving_mass
        guess = -(alph/2) - np.sqrt(alph**2+4*alph)/2
        temp_field = a_0*fsolve(solver_func,guess,xtol=1e-7)

        #- interpolating -#
        interp = interp1d(solving_array,temp_field,fill_value="extrapolate")
        field = interp(r)
    else:
        field = -G_unit*m_tot/(r**2)

    #------------------------------------------------------------------------------------------------------------------#
    # Solving the temperature equation!
    #------------------------------------------------------------------------------------------------------------------#
    integrand = np.flip(cumulative_trapezoid(np.flip(rho_g*field),np.flip(r),initial=0))
    #T## = *(m_p.in_units("kg")*mu)/(rho_g*(1*pyn.units.Unit("m").in_units(CONFIG["units"]["default_length_unit"]))) * integrand

    #return T*(pyn.units.Unit("keV").in_units(output_units))


if __name__ == '__main__':
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{graphicx,amsmath,amssymb,amsfonts,algorithmicx,algorithm,algpseudocodex}"
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['xtick.direction'] = "in"
    plt.rcParams['ytick.direction'] = "in"

    clusters = load_clusters()
    r_disp = [(90,1000),(20,800),(60,1000),(60,2000),(60,1200),(20,1100),(80,1100),(20,1100),
              (30,2000),(90,3000),(10,1000),(10,1000),(40,300)]
    cs = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#f46a9b",
          "#e60049", "#0bb4ff", "#50e991"]

    for id,p in enumerate(zip(clusters,r_disp,cs)):
        cluster, rd, cs = p
        r = np.linspace(rd[0],5000,4000)
        T,rho = cluster.gas_temperature(r),cluster.gas_density(r)
        M,N = Mdynamical_mass(gamma(T,rho,r),r,lambda x: x/(1+x)),Ndynamical_mass(gamma(T,rho,r),r)
        try:
            rm = r[np.where(np.gradient(N,r)<0)][0]
        except:
            rm = 5000

        r = np.linspace(rd[0],rm,4000)
        T,rho = cluster.gas_temperature(r),cluster.gas_density(r)
        M,N = Mdynamical_mass(gamma(T,rho,r),r,lambda x: x/(1+x)),Ndynamical_mass(gamma(T,rho,r),r)
        rM,rN = (1/(4*np.pi*(r**2)))*np.gradient(M,r),(1/(4*np.pi*(r**2)))*np.gradient(N,r)
        rB = cluster.gas_density(r)
        diff = rM-rB

        r500 = r[np.where(rN>=500*133)][-1]
        r200 = r[np.where(rN>=200*133)][-1]
        print(r[np.where(diff<0)][0])
        rx_max = np.amin(r[np.where(diff<0)])
        rt_max = np.amin(r[np.where(np.gradient(np.log(N),r)<1/(r*((G_mass*N/(a_0*r**2))+1)))])
        plt.bar(id+1,rm-rd[0],0.8,rd[0],log=True,alpha=0.2,color=cs)
        plt.semilogy([id+1,id+1],[rx_max,rt_max],c=cs)
        plt.semilogy([id+0.75,id+1.25],[rx_max,rx_max],c=cs)
        plt.semilogy([id + 0.75, id + 1.25], [rt_max, rt_max],c=cs)
        plt.semilogy([id+0.5,id+1.5],[r200,r200],"k-.")
        plt.semilogy([id + 0.5, id + 1.5], [r500, r500], "k:")


    plt.title(r"Best Case and Observed Maximal Viable Radius $\mathcal{R}_v$")
    plt.ylabel(r"Cluster Radius / $[\mathrm{kpc}]$")
    plt.xlabel(r"Cluster")
    plt.xticks([i+1 for i in range(len(clusters))],[cluster.name for cluster in clusters],rotation="vertical")

    #- Custom Legends -#
    b1 = LineCollection(np.empty((2,2,2)),colors="k")
    line = Line2D([],[],ls="",c="k")
    cl = [Line2D([],[],color="k",ls=":"),
          Line2D([],[],color="k",ls="-."),
          ErrorbarContainer((line,[line],[b1]),has_yerr=True),
          Rectangle((0,0),0,0,color="blue",alpha=0.25)]

    plt.legend(cl,[r"$\mathrm{r}_{200}$",r"$\mathrm{r}_{500}$",r"Best Case and Observed $\mathcal{R}_v$",
                   "Fit Range"])

    plt.show()
        #ax.loglog(r, np.gradient(N-MB,r) / (4 * np.pi * r ** 2))
        #ax.loglog(r, -np.gradient(N-MB,r) / (4 * np.pi * r ** 2),"r")
        #ax.loglog(r,(a_0/(2*np.pi*G_mass))/r)
