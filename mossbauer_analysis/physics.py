import numpy as np

planck_mass = 1.22e19  # GeV
m_times_eV = 0.2*1e9*1e-15
kappa_d = 45./4.7  # idk
Au_density = 19.32  # g/cm3
Au_mm = 197.  # g/mol
nAu = Au_density / (Au_mm) * 6e23 * (100.**3) * 197  # number of nucleONS(!!) per m^3

def alpha_to_yukawa(alpha):
    """For down quark model"""
    return np.sqrt(4*np.pi*alpha) * (1/planck_mass) / kappa_d

def yukawa_to_alpha(yukawa):
    """For down quark model"""
    return (yukawa / (1/planck_mass) * kappa_d)**2/(4*np.pi)

def get_limits(r, deltaE):
    """Exactly reproduces fig. 3 in PRD (2020)"""
    epsilon = 2.5/kappa_d  # fudge factor
    bkg = 1e-14*((1e-8/r)**4)  # eV
    sensi = deltaE + bkg  # eV
    gnucsolve = np.sqrt(2*np.e/epsilon) * np.sqrt(sensi / (nAu * (r**2) * m_times_eV))
    alpha_ann = (planck_mass)**2 / 4 / np.pi * (gnucsolve**2)
    yukawa_modulus = alpha_to_yukawa(alpha_ann)
    return yukawa_modulus, alpha_ann
