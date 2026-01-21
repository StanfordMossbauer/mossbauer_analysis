from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import jv
import xraydb 
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


############################################     CONSTANTS   #########################################################



############################################     FUNCTIONS   #########################################################


# point source
def compute_solid_angle(r, d):
    return 2*np.pi*(1-d/np.sqrt(r**2 + d**2))

def E(x,y,D):
    return D/(x**2 + y**2 + D**2)**(3/2)


# distributed source
def E_disk(x, y, D, r_s, q=1.0, N=256):
    """
    Irradiance at (x,y) on plane z=D from a uniform circular disk source (radius r_s) at z=0.
    Isotropic emission. Per-unit source power density q.
    v is integrated analytically; u via Gauss–Legendre (N nodes).
    """
    X, Y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))

    xi, wi = leggauss(N)         # nodes/weights on [-1,1]
    u = r_s * xi                 # map to [-r_s, r_s]
    w = r_s * wi

    # Broadcast to (N, *X.shape)
    u_b = u.reshape(N, *([1]*X.ndim))
    w_b = w.reshape(N, *([1]*X.ndim))
    Xb = X[None, ...]
    Yb = Y[None, ...]

    s = np.sqrt(r_s*r_s - u_b*u_b)         # √(r_s^2 - u^2)
    A = D*D + (Xb - u_b)**2

    w2 =  s - Yb
    w1 = -s - Yb

    term = (w2 / np.sqrt(A + w2*w2)) - (w1 / np.sqrt(A + w1*w1))
    integrand = (D / A) * term

    E = (q / (4.0 * np.pi)) * np.sum(w_b * integrand, axis=0)
    return E.item() if E.shape == () else E

def thickness_effective(x, y, D):
    return np.sqrt(x**2 + y**2 + D**2)/D

def quartz_transmission(x, y, D, t=200e-6):

    t_eff = t*np.sqrt(x**2 + y**2 + D**2)/D
    mu = xraydb.material_mu('Quartz', 14.4e3)  * thickness_effective(x, y, D) * 100  # 100 is for conversion units

    return np.exp(-mu) 

