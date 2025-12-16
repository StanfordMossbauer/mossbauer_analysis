# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:50:08 2025

@author: Albert
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

s_param_data = np.loadtxt(r"C:\Users\Albert\PythonRepoLand\ZNL20\AGM_TN02_5MHz_PDMS.csv", delimiter=",")

freq_data = s_param_data[:,0]
s11_data = np.power(10, s_param_data[:,1]/20)
s12_data = np.power(10, s_param_data[:,2]/20)

#IDT Parameters for AGM TN02
Nf = 200 #Finger pairs per transducer
W = 970E-6 #Transducer width in meters
Z = 12E-3 #Distance center-to-center of IDTs
#Cs approximately, W*50E-12 in F/m, is fit directly 
freq0 = 97.9*10**6 #Resonance frequency, experimentally determined
k2 = 0.0014 #From calculated value in Slobodnik, 1976 "Surface Acoustic Waves and SAW Materials"

G0 = 4/np.pi*k2*(2*np.pi*freq0)*Nf**2 #SAW conductance at resonance, divided by the capacitance per finger times width
Z1 = 50 #VNA Impedance

def X(freq):
    return Nf*np.pi*(freq-freq0)/freq0

def Ga(x, Cs):
    return Cs*G0*(np.sin(x)/x)**2

def Ba(x, Cs):
    return 1j*Cs*G0*(np.sin(2*x)-2*x)/(2*x**2)

def Z2(x, Cs):
    return 1.0/Ga(x, Cs)

def ZL(freq, RL, Cs):
    return 1/(1/RL + 1j*2*np.pi*freq*Cs*Nf + Ba(X(freq), Cs))

def propagation_loss(freq):
    #Calculated using Slobodnik 1976
    VAC = 2.62 #dB/microsecond
    AIR = 0.47 #dB/microsecond
    propagation_loss = VAC*(freq*1E-9)**2 + AIR*(freq*1E-9) #dB/microsecond
    time_in_mode = Z/(32E-6*freq)*1E6 #in microseconds
    return propagation_loss*time_in_mode

diffraction_loss = 1.41 #dB, interpolated from Slobodnik 1976

def S11(freq, RL, Cs):
    return (np.abs((ZL(freq, RL, Cs)*Z2(X(freq), Cs)-np.conjugate(Z1)*(ZL(freq, RL, Cs)+Z2(X(freq), Cs))))/np.abs((ZL(freq, RL, Cs)*Z2(X(freq), Cs)+Z1*(ZL(freq, RL, Cs)+Z2(X(freq), Cs)))))

def S12(freq, RL, Cs, alpha, noise_floor):
    return (np.abs(noise_floor) + (10**(-diffraction_loss/20)) * (10**(-propagation_loss(freq)/20)) * 1/2 * (np.sqrt(1-alpha**2)) * (np.abs(np.sqrt(np.real(Z1))/np.sqrt(np.real(Z2(X(freq), Cs)))*ZL(freq, RL, Cs)*(Z2(X(freq), Cs)+np.conjugate(Z2(X(freq), Cs)))/(Z1*Z2(X(freq), Cs)+ZL(freq, RL, Cs)*(Z1+Z2(X(freq), Cs)))))**2)

def S12_source_to_SAW(freq, RL, Cs, alpha, noise_floor):
    #Calculates right traveling SAW amplitude that makes it to the center of the iron rectangle
    return np.sqrt((10**(-diffraction_loss/20)) * (10**(-propagation_loss(freq)/20)) * 1/2  * (np.abs(np.sqrt(np.real(Z1))/np.sqrt(np.real(Z2(X(freq), Cs)))*ZL(freq, RL, Cs)*(Z2(X(freq), Cs)+np.conjugate(Z2(X(freq), Cs)))/(Z1*Z2(X(freq), Cs)+ZL(freq, RL, Cs)*(Z1+Z2(X(freq), Cs)))))**2)

def s_fit_function(freq_array, RL, C, alpha, noise_floor):
    midpoint_array = len(freq_array)//2
    freq_s11 = freq_array[:midpoint_array]
    freq_s12 = freq_array[midpoint_array:]
    s11_array = S11(freq_s11, RL, C)
    s12_array = S12(freq_s12, RL, C, alpha, noise_floor)
    return np.hstack((s11_array, s12_array))
    
    return 

fit_freq_data = np.hstack((freq_data, freq_data))
fit_y_data = np.hstack((s11_data, s12_data))

fit_param, fit_param_pcov = scipy.optimize.curve_fit(s_fit_function, fit_freq_data, fit_y_data, p0=(1E3, 50E-15, 0.01, 1E-7))

print("RL (ohms), Capacitance per finger pair (F), SAW reflection Coefficient (alpha), noise floor ")
print(fit_param)
print(np.sqrt(np.diag(fit_param_pcov)))

plt.title("AGM TN02 S11")
plt.plot(freq_data, 20*np.log10(s11_data))
plt.plot(freq_data, 20*np.log10(S11(freq_data, fit_param[0], fit_param[1])))
plt.xlabel("Frequency (Hz)")
plt.ylabel("S11 (dB)")
plt.legend(("Measured","Theory"))
plt.ylim(-3,0)
plt.show()

plt.title("AGM TN02 S21")
plt.plot(freq_data, 20*np.log10(s12_data))
plt.plot(freq_data, 20*np.log10(S12(freq_data, *fit_param)))
plt.xlabel("Frequency (Hz)")
plt.ylabel("S12 (dB)")
plt.legend(("Measured","Theory"))
plt.ylim(-80,0)
plt.show()

source_to_SAW_S12 = 20*np.log10(S12_source_to_SAW(freq_data, *fit_param))
print("Source to SAW S21:")
print(np.max(source_to_SAW_S12))

plt.title("AGM TN02 Source to SAW S12")
plt.plot(freq_data, source_to_SAW_S12, color="tab:orange")
plt.xlabel("Frequency (Hz)")
plt.ylabel("S12 (dB)")
plt.ylim(-80,0)
plt.show()
