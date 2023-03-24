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
from scipy.optimize import curve_fit,brentq
#----------------------------------------------------------------------------------------------------------------------#
#   Constants                                                                                                          #
#----------------------------------------------------------------------------------------------------------------------#
proton_mass = 1.672621911e-27 # KG
G = 6.674e-11
G_mass = 1.39367222e-19
G_kpc2_m_kg_s2 = 4.3
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

    def get_NFW(self,r):
        gas_density = self.gas_density(r) #- grabbing the gas density so that we can fit it.
        scale = brentq(lambda x: x+x**3 - (gas_density[0]/gas_density[-1]),0,15)
        RS = r[-1]/scale

        rs = r[np.where(r<RS)][-1]
        print(rs,RS)
        po = 4*gas_density[np.where(r==rs)][0]
        print(po)
        fit = curve_fit(NFW,r,gas_density,p0=[po,200])

        return fit

    def nfw(self,r):
        return NFW(r,self.NFW_RHO0,self.NFW_RS)

    def get_aqual_potential(self,r, A):

        # Setup
        #--------------------------------------------------------------------------------------------------------------#
        #- Newtonian poptential -#
        # -4 pi G rho Rs^3/r * ln(1+r/rs)
        phi_n = -np.flip(((4 * np.pi * G_kpc2_m_kg_s2 *self.NFW_RHO0* (self.NFW_RS)**3)/(r))*np.log(1+(r/self.NFW_RS)))
        dphi_n = np.gradient(phi_n,r)/(3.086e19)


        #- Setting up the Euler Run. -#
        r = np.flip(r)
        phi,dphi = np.zeros((r.size,)),np.zeros((r.size,))
        dr = [np.abs(r[i]-r[i-1]) for i in range(1,len(r))]

        # Running
        #--------------------------------------------------------------------------------------------------------------#
        for i,rs in enumerate(r):
            if i > 0:
                print(i,rs,phi[i-1],A(phi[i-1]))

                dphi[i] = ((dphi_n[i])/2) - np.sqrt(dphi_n[i]**2 -(4*dphi_n[i]*A(phi[i-1])))/2
                print(dphi[i])
                phi[i] = phi[i-1] + (dphi[i]*dr[i-1])

        r, phi,dphi = np.flip(r),np.flip(phi),np.flip(-dphi)

        M = ((r**2)/G_mass)*dphi*(np.abs(dphi)/(A(phi)+np.abs(dphi)))

        plt.loglog(r[:-1],M[:-1])
        plt.loglog(r,4*np.pi*cumulative_trapezoid(self.nfw(r)*r**2,r,initial=M[-1]))
        plt.show()
#----------------------------------------------------------------------------------------------------------------------#
#   Functions                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#
def load_clusters():
    os.chdir("/home/ediggins/Documents/ediggins_personal/MOND_Paper")
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
                                 sample_frequency=1,
                            T0=0
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
    T = ((proton_mass*eta)/(rho_g) * integrand * 3.086e19 * 6.242e15)

    return T

def NFW(x,rho,rs):
    return rho/((x/rs)*(1+(x/rs))**2)

if __name__ == '__main__':

    clusters = load_clusters()
    r_disp = [(90,1000),(20,800),(60,1000),(60,2000),(60,1200),(20,1100),(80,1100),(20,1100),
              (30,2000),(90,3000),(10,1000),(10,1000),(40,300)]
    cs = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#f46a9b",
          "#e60049", "#0bb4ff", "#50e991"]

    for cluster,rs,c in zip(clusters,r_disp,cs):
        r = np.linspace(rs[0],4*rs[1],1000)
        gt,gd = cluster.gas_temperature(r),cluster.gas_density(r)

        Mg = 4*np.pi*cumulative_trapezoid(gd*r**2,r,initial=0)
        MN = Ndynamical_mass(gamma(gt,gd,r),r)
        MM = Mdynamical_mass(gamma(gt,gd,r),r,lambda x: x/(1+x))
        Mdm = MN-Mg
        MMdm = MM-Mg
        dMdm = np.gradient(Mdm,r)
        MMdm[np.where(dMdm<0)] = 0

        T = get_temperature_profile(Mdm,Mg,r,lambda x: x/(1+x),T0=gt[-1])
        T2 = get_temperature_profile(MMdm,Mg,r,lambda x: x/(1+x),mode="MOND")
        plt.semilogx(r,T)
        plt.semilogx(r,T2)
        plt.semilogx(r,gt)
        plt.show()


        plt.show()