from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import jv
import xraydb 
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

############################################     CONSTANTS   #########################################################

# Fundamental constants
c = 3e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant in J/K
e = 1.6e-19  # Elementary charge in C
Na = 6.022e23  # Avogadro's number


############################################     FUNCTIONS   #########################################################


def calculate_sigma0(E0, Ie, Ig, alpha):
    return 2.446e-15 / (E0 / 1e3) ** 2 * (2 * Ie + 1) / (2 * Ig + 1) * 1 / (1 + alpha)

def calculate_mu_e(element, energy):
    return xraydb.mu_elam(element, energy)

def calculate_recoilless_fraction(T, Td, E0, M):
    integral = quad(lambda x: x / (np.exp(x) - 1), 0, Td / T)[0]
    return np.exp(-(3 * (E0 * e) ** 2) / (kB * Td * M * c ** 2) * (1 / 4 + (T / Td) ** 2 * integral))

def get_current_activity(half_life_days, activity, date, nowdate=None):
    if nowdate is None:
        nowdate = datetime.now()
    else:
        nowdate = datetime.strptime(nowdate, '%Y%m%d')
    tdiff_seconds = (nowdate - datetime.strptime(date, '%Y%m%d')).total_seconds()
    return activity*((0.5)**(tdiff_seconds/(3600*24)/half_life_days))

def calculate_photon_rate(activity_Ci,relative_intensity):
    return 3.7e10 * activity_Ci * relative_intensity

def calculate_thickness_gcm2(thickness_m, rho, abundance):
    return thickness_m * rho/10 * abundance

def _lorentzian_s(x, x0, gamma):
    return gamma/(2*np.pi)/( ((x-x0))**2 + (gamma/2)**2)

def _lorentzian_a(x, x0, gamma):
    return (gamma/2)**2/( ((x-x0))**2 + (gamma/2)**2)


# point source
def compute_solid_angle(r, d):
    return 2*np.pi*(1-d/np.sqrt(r**2 + d**2))

def E(x,y,D):
    return D/(x**2 + y**2 + D**2)**(3/2)

def thickness_correction(x, y, D):
    return np.sqrt(x**2 + y**2 + D**2)/D

def quartz_transmission(x, y, D, t_min=200e-6):

    t_eff = t_min*thickness_correction(x, y, D)
    mu = xraydb.material_mu('Quartz', 14.4e3)  * t_eff * 100  # 100 is for conversion units

    return np.exp(-mu) 



############################################     SOURCE CLASSES   #########################################################
class CobaltRhodium:
    def __init__(self):
        self.T = 292
        self.Td = 510
        self.E0 = 14.4e3
        self.Gamma_ev = 4.55e-9
        self.Eres = [0]
        self.split_ratio = [1]
        self.activity_Ci = 50e-3
        self.production_date = '20250112'
        self.date = None
        self.half_life = 272
        self.mossbauer_relative_intensity = 0.0916
        self.M = 57e-3 / Na
        self.element = 'Fe'
        self.alpha = 8.17
        self.update_params()
    
    def update_params(self):
        self.Gamma = self.Gamma_ev * c / self.E0 * 1000
        self.current_activity_Ci = get_current_activity(self.half_life, self.activity_Ci, self.production_date, self.date)
        self.mossbauer_photon_rate = calculate_photon_rate(self.current_activity_Ci, self.mossbauer_relative_intensity)
        self.fs = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float) / np.sum(self.split_ratio)


class CobaltFe:
    def __init__(self):
        self.T = 292
        self.Td = 470
        self.E0 = 14.4e3
        self.Gamma_ev = 4.55e-9
        self.Eres = [-5, -3, -1, 1, 3, 5]
        self.split_ratio = [3, 2, 1, 1, 2, 3]
        self.activity_Ci = 50e-3
        self.production_date = '20250112'
        self.date = None
        self.half_life = 272
        self.mossbauer_relative_intensity = 0.0916
        self.M = 57e-3 / Na
        self.element = 'Fe'
        self.alpha = 8.17
        self.update_params()
    
    def update_params(self):
        self.Gamma = self.Gamma_ev * c / self.E0 * 1000
        self.current_activity_Ci = get_current_activity(self.half_life, self.activity_Ci, self.production_date, self.date)
        self.mossbauer_photon_rate = calculate_photon_rate(self.current_activity_Ci, self.mossbauer_relative_intensity)
        self.fs = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float) / np.sum(self.split_ratio)


############################################     ABSORBER CLASSES   #########################################################


class alphaFe:
    def __init__(self):
        self.thickness_m = 25e-6
        self.T = 292
        self.Td = 470
        self.E0 = 14.4e3
        self.Gamma_ev = 4.55e-9
        self.Eres = [-5, -3, -1, 1, 3, 5]
        self.split_ratio = [3, 2, 1, 1, 2, 3]
        self.abundance = 0.0212
        self.M = 57e-3 / Na
        self.nM = Na / 57
        self.rho = 7.87e3
        self.element = 'Fe'
        self.alpha = 8.17
        self.update_params()
    
    def update_params(self):
        self.Gamma = self.Gamma_ev * c / self.E0 * 1000
        self.mu_e = calculate_mu_e(self.element, self.E0)
        self.fa = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float) / np.sum(self.split_ratio)
        self.sigma0 = calculate_sigma0(self.E0, 3/2, 1/2, self.alpha)
        self.thickness_gcm2 = calculate_thickness_gcm2(self.thickness_m, self.rho, 1)
        self.thickness_gcm2_Fe57 = calculate_thickness_gcm2(self.thickness_m, self.rho, self.abundance)
        self.thickness_normalized = self.fa * self.nM * self.sigma0 * self.thickness_gcm2_Fe57


class KFeCy:
    def __init__(self):
        self.thickness_gcm2_Fe = 6e-3
        self.E0 = 14.4e3
        self.Gamma_ev = 4.55e-9
        self.Eres = [0]
        self.split_ratio = [1]
        self.abundance = 0.0212
        self.M = 57e-3 / Na
        self.nM = Na / 57
        self.rho = 7.87e3
        self.elements = {'Fe': 1, 'K': 4, 'C': 6, 'N': 6, 'H': 6, 'O': 3}
        self.alpha = 8.17
        self.sigma0 = 2.1e-18 # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.4.2915
        self.fa = 0.311 # https://www.sciencedirect.com/science/article/abs/pii/0029554X81900987
        self.update_params()
    
    def update_params(self):
        self.Gamma = self.Gamma_ev * c / self.E0 * 1000
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float) / np.sum(self.split_ratio)
        self.thickness_gcm2_Fe57 = self.thickness_gcm2_Fe * self.abundance
        self.thickness_normalized = self.fa * self.nM * self.sigma0 * self.thickness_gcm2_Fe57
        self.mass_sum = np.sum([val * xraydb.atomic_mass(element) for element, val in self.elements.items()])
        self.mu_e = np.sum([xraydb.mu_elam(element, self.E0) * multiplicity * xraydb.atomic_mass(element) / self.mass_sum 
                            for element, multiplicity in self.elements.items()])
        self.thickness_gcm2 = self.thickness_gcm2_Fe / (self.elements['Fe'] * xraydb.atomic_mass('Fe')) * self.mass_sum

        
    


############################################     DETECTOR CLASSES   #########################################################
class epix_camera:
    def __init__(self):

        self.H = 35.2e-3  # camera height
        self.L = 38.4e-3  # camera width
        self.NpixelsH = 352  # number of pixels in height
        self.NpixelsL = 384  # number of pixels in width
        
        self.distance_sd = 32.6e-3  # source-detector distance in meters
        self.distance_sa = 21.8e-3  # source-absorber distance in meters
        self.source_radius = 3e-3  # source radius in meters
        self.absorber_radius = 25.4e-3  # absorber radius in meters

        self.efficiency_14keV = 0.9  # detector efficiency at 14.4 keV
        self.quartz_thickness_m = 200e-6  # quartz window thickness in meters
        self.snr14keV = 10  # signal to noise ratio at 14.4 keV
        
        self.update_params()
    
    def update_params(self):
        self.overlap_radious = self.source_radius*(self.distance_sd/self.distance_sa-1)
        self.x = np.linspace(-self.L/2, self.L/2, self.NpixelsL)
        self.y = np.linspace(-self.H/2, self.H/2, self.NpixelsH)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        #geometry
        self.detector_area = self.L * self.H
        self.detector_radius = np.sqrt(self.detector_area/np.pi)
        self.source_detector_solid_angle = compute_solid_angle(self.detector_radius, self.distance_sd)
        self.source_absorber_solid_angle = compute_solid_angle(self.absorber_radius, self.distance_sa)
        self.absorber_detector_solid_angle = compute_solid_angle(self.detector_radius, self.distance_sd - self.distance_sa)
        self.pixel_solid_angle_map = E(self.X, self.Y, self.distance_sd)*self.dx*self.dy
        self.source_detector_solid_angle2 = self.pixel_solid_angle_map.sum()
        
        #Quartz attenuation
        self.quartz_transmission_map = quartz_transmission(self.X, self.Y, self.distance_sd, self.quartz_thickness_m)
        self.weighted_quartz_attenuation = np.sum(self.quartz_transmission_map*self.pixel_solid_angle_map)/self.pixel_solid_angle_map.sum()
        
        #Efficiency map
        self.solid_angle_efficiency = self.pixel_solid_angle_map.sum()/(4*np.pi) 
        self.total_efficiency = self.solid_angle_efficiency * self.efficiency_14keV * self.weighted_quartz_attenuation
        
       
        #Cosine broadening
        self.photon_angle_map = np.arctan(np.sqrt(self.X**2 + self.Y**2)/self.distance_sd)
        self.relative_line_broadening_map = 1/np.cos(self.photon_angle_map)
        self.weighted_broadening = np.sum(self.relative_line_broadening_map*self.pixel_solid_angle_map)/self.pixel_solid_angle_map.sum()


class SDD_detector:
    def __init__(self):

        self.active_area = 17e-6  # active area in m^2
        self.distance_sa = 95e-3  # source-absorber distance in meters
        self.distance_sd = 100-3  # source-detector distance in meters
        self.absorber_radius = 20e-3  # absorber radius in meters
        
        self.efficiency_14keV = 0.8  # detector efficiency at 14.4 keV
        self.snr14keV = 7  # signal to noise ratio at 14.4 keV
        self.quartz_thickness_m = 0e-6  # quartz window thickness in meters
        
        self.update_params()
    
    def update_params(self):
        self.pixel_solid_angle = self.active_area/ (self.distance_sd ** 2)
        self.quartz_transmission = quartz_transmission(0, 0, self.distance_sd, self.quartz_thickness_m)
        self.total_efficiency = self.pixel_solid_angle * self.efficiency_14keV * self.quartz_transmission
        self.detector_area = self.active_area
        self.detector_radius = np.sqrt(self.detector_area/np.pi)
        self.source_detector_solid_angle = compute_solid_angle(self.detector_radius, self.distance_sd)
        self.source_absorber_solid_angle = compute_solid_angle(self.detector_radius, self.distance_sd)
        self.absorber_detector_solid_angle = compute_solid_angle(self.detector_radius, self.distance_sd - self.distance_sa)

    
###########################################      MOSSBAUER    ######################################################

c=3e8

class Mossbauer:
    
    def __init__(self, source, absorber, detector = SDD_detector()):
    
        self.source = source
        self.absorber = absorber
        self.detector = detector

    def source_specrtrum(self, E, v):
        spectrum = 0
        for coef, Eres in zip(self.source.transition_coefficients, self.source.Eres):
                spectrum += coef *_lorentzian_s(
                    E,
                    Eres - v,
                    self.source.Gamma
                )
        
        return self.source.fs  * spectrum
    
    def cross_section(self, E):
        spectrum = 0
        for coef, Eres in zip(self.absorber.transition_coefficients, self.absorber.Eres):
            spectrum += coef *_lorentzian_a(
                E,
                Eres,
                self.absorber.Gamma
            )
            
        return self.absorber.sigma0 * spectrum
    
    def non_resonant_attenuation(self):
        return np.exp(-self.absorber.mu_e * self.absorber.thickness_gcm2) 
    
    def resonant_transmission_fraction(self, E, v):
        
        res = ( self.source_specrtrum(E, v) 
                * np.exp(-self.cross_section(E) * self.absorber.fa * self.absorber.nM * self.absorber.thickness_gcm2_Fe57 )
              )
        return res

    def resonant_transmission_rate(self, v):
        res = quad_vec(lambda E: self.resonant_transmission_fraction(E,v), -np.inf, np.inf)[0] * self.non_resonant_attenuation()
        return res
    
    def nonresonant_transmission_rate(self):
        return (1 - self.source.fs)* self.non_resonant_attenuation()

    def total_transmission_rate(self, v):
        return self.nonresonant_transmission_rate() + self.resonant_transmission_rate(v)

    
    #here add reemission and compton offset detected rates

    def compton_offset(self):
        1*self.non_resonant_attenuation()/self.detector.snr14keV

    def absorbtion_rate(self,v):
        return (self.source.fs-self.resonant_transmission_rate(v)) * self.detector.source_absorber_solid_angle/(4*np.pi)

    def readiative_branching(self):
        return 1/(1+self.absorber.alpha)
    
    def escape_probability(self):
        return (1-np.exp(-self.absorber.thickness_normalized))/self.absorber.thickness_normalized
    
    def reemission_rate(self,v):
        return (self.absorbtion_rate(v) *
                self.readiative_branching() *
                self.escape_probability() *
                self.non_resonant_attenuation() *
                self.detector.absorber_detector_solid_angle/self.detector.source_detector_solid_angle)
    
    def compton_offset(self):
        return 1*self.non_resonant_attenuation()/self.detector.snr14keV
    
    def total_spectrum(self, v) :
        return ( self.total_transmission_rate(v) 
                + self.reemission_rate(v) 
                + self.compton_offset()
               )

    def epsilon(self,t):
        return 1-np.exp(-t/2)*jv(0,1j*t/2).real


if __name__ == "__main__":
    source = CobaltRhodium()

    absorber = KFeCy()
    absorber.update_params()

    moss = Mossbauer(source, absorber)

    v = np.linspace(-10,10,1000)
    t = moss.total_transmission_rate(v)
    norm = moss.source.mossbauer_photon_rate


    plt.plot(v, t/norm, label = 'Fe')

    plt.axhline((1-((t.max()-t.min())/2/t.max()))*moss.non_resonant_attenuation())
    plt.axvline(-(source.Gamma + absorber.Gamma)/2)
    plt.axvline((source.Gamma + absorber.Gamma)/2)

    plt.axhline((1 - source.fs)*moss.non_resonant_attenuation(),color = 'r')
    plt.axhline((1 - source.fs*moss.epsilon(absorber.thickness_normalized))*moss.non_resonant_attenuation(),color = 'g')



    plt.ylim(0,1)
    plt.xlim(-1,1)


    vis = moss.epsilon(absorber.thickness_normalized)*source.fs
    vis, (t.max()-t.min())/t.max(), source.mossbauer_photon_rate/1e6, moss.non_resonant_attenuation()