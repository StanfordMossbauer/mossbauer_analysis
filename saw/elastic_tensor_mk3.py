# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:11:14 2026

@author: Albert
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

W = 970E-6 #Transducer width in meters
freq_0 = 97.9E6 #Resonance frequency in Hz, experimentally determined
rho = 2650.0 #density quartz, kg/m^3

c = np.zeros((3,3,3,3))

#values for left handed alpha quartz, taken from Coquin and Tiersten 1967
c11 = 86.74E9
c12 = 6.99E9
c13 = 11.91E9
c14 = -17.91E9
c22 = c11
c23 = c13
c24 = -c14
c33 = 107.2E9
c34 = 0.0
c44 = 57.94E9
c55 = c44
c56 = c14
c66 = 0.5*(c11-c12)


# key 1 -> (0,0), 2 -> (1,1), 3 -> (2,2), 4 -> (1,2), 5 -> (0,2), 6 -> (1,0): add 1 to all (#,#) pairs to get the notation in Coquin and Tiersten
c[0,0,0,0] = c11

c[0,0,1,1] = c12
c[1,1,0,0] = c12

c[0,0,2,2] = c13
c[2,2,0,0] = c13

c[0,0,1,2] = c14
c[1,2,0,0] = c14
c[2,1,0,0] = c14
c[0,0,2,1] = c14

c[1,1,1,1] = c22

c[1,1,2,2] = c23
c[2,2,1,1] = c23

c[1,1,1,2] = c24
c[1,1,2,1] = c24
c[1,2,1,1] = c24
c[2,1,1,1] = c24

c[2,2,2,2] = c33

c[2,2,1,2] = c34
c[2,2,2,1] = c34
c[1,2,2,2] = c34
c[2,1,2,2] = c34

c[1,2,1,2] = c44
c[1,2,2,1] = c44
c[2,1,1,2] = c44
c[2,1,2,1] = c44

c[0,2,0,2] = c55
c[0,2,2,0] = c55
c[2,0,0,2] = c55
c[2,0,2,0] = c55

c[0,2,0,1] = c56
c[0,2,1,0] = c56
c[2,0,0,1] = c56
c[2,0,1,0] = c56
c[0,1,0,2] = c56
c[0,1,2,0] = c56
c[1,0,0,2] = c56
c[1,0,2,0] = c56

c[0,1,0,1] = c66
c[0,1,1,0] = c66
c[1,0,0,1] = c66
c[1,0,1,0] = c66

def rotation_array(theta):
    rotation_array_ = np.zeros((3,3))
    rotation_array_[0,0] = 1
    rotation_array_[1,1] = np.cos(theta)
    rotation_array_[1,2] = np.sin(theta)
    rotation_array_[2,1] = -np.sin(theta)
    rotation_array_[2,2] = np.cos(theta)
    return rotation_array_

def calculate_elastic_tensor(theta):
    rotation = rotation_array(theta)
    c_prime_ = np.zeros(np.shape(c))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for r in range(3):
                        for s in range(3):
                            for t in range(3):
                                for u in range(3):
                                    c_prime_[i,j,k,l] = c_prime_[i,j,k,l] + rotation[i,r]*rotation[j,s]*rotation[k,t]*rotation[l,u]*c[r,s,t,u]
    return c_prime_


#THIS IS Where idefine what points to calculate!
theta_array = np.linspace(-np.pi/2, np.pi/2, 100)  # Multiple angles from -90 to +90 degrees
theta_array = (np.linspace(-90, 90, 46)+0.75)/180*np.pi  # Multiple angles from -90 to +90 degrees with 0.75 degree increments
# theta_array = [42.0/180*np.pi]  # Single angle for testing




velocity_array = np.zeros(len(theta_array))

# Initialize arrays to store decay coefficients for plotting
decay_coeff_real_0 = []
decay_coeff_imag_2 = []
decay_coeff_real_1 = []
decay_coeff_real_2 = []
angle_degrees = []

a = sp.symbols('a')
a2 = sp.symbols('a2')

for index in range(len(theta_array)):
    c_prime = calculate_elastic_tensor(theta_array[index])
    
    def results_maker(vs):
        vs = vs[0]
        matrix_21 = sp.Matrix([[(rho*vs**2 + c_prime[0,1,0,1]*a**2 - c_prime[0,0,0,0]), 1.0j*a*(c_prime[0,0,1,1] + c_prime[0,1,0,1]), 1.0j*a*(c_prime[0,0,1,2] + c_prime[0,2,0,1])], [1.0j*a*(c_prime[0,0,1,1] + c_prime[0,1,0,1]), rho*vs**2 + c_prime[1,1,1,1]*a**2 - c_prime[0,1,0,1], c_prime[1,1,1,2]*a**2 - c_prime[0,2,0,1]],[1.0j*a*(c_prime[0,0,1,2] + c_prime[0,2,0,1]), c_prime[1,1,1,2]*a**2 - c_prime[0,2,0,1], rho*vs**2 + c_prime[1,2,1,2]*a**2 - c_prime[0,2,0,2]]])
        equation_21_det = matrix_21.det()
        equation_21_det = equation_21_det.subs(a**2, a2)
        solutions_system = sp.polys.polytools.nroots(equation_21_det, n=40)        
        a_array = np.sqrt(np.array(solutions_system).astype(np.complex128))
        
        beta = np.zeros((3,3)).astype(np.complex128)
            
        for j in range(len(a_array)):
            matrix_21_j = np.array(matrix_21.subs(a, a_array[j])).astype(np.complex128)
            beta[:,j] = scipy.linalg.null_space(matrix_21_j)[:,0]
            
        L_matrix = np.zeros((3,3)).astype(np.complex128)
        
        for j in range(3):
            L_matrix[0,j] = c_prime[0,1,0,1]*a_array[j]*beta[0,j] + 1.0j*c_prime[0,1,0,1]*beta[1,j] + 1.0j*c_prime[0,2,0,1]*beta[2,j]
            L_matrix[1,j] = 1.0j*c_prime[0,0,1,1]*beta[0,j] + c_prime[1,1,1,1]*a_array[j]*beta[1,j] + c_prime[1,1,1,2]*a_array[j]*beta[2,j]
            L_matrix[2,j] = 1.0j*c_prime[0,0,1,2]*beta[0,j] + c_prime[1,1,1,2]*a_array[j]*beta[1,j] + c_prime[1,2,1,2]*a_array[j]*beta[2,j]
        
        B = np.zeros(3).astype(np.complex128)
        B = scipy.linalg.null_space(L_matrix, rcond = 1E-7) #rcond makes the null_space algorithm happy
        
        remainder = np.abs(np.linalg.det(L_matrix))*1E-28 #rescaling the remainder makes the scipy.optimize.minimize algorithm happy
        return remainder, a_array, beta, B.reshape(-1)
    
    def velocity_finder(vs_guess):
        return results_maker(vs_guess)[0]
    
    vs = scipy.optimize.minimize(velocity_finder, x0=3200, method = "SLSQP", bounds=[(3100,3330)]).x
    
    velocity_array[index] = vs[0]

    final_remainer, a_actual, beta_actual, B_actual = results_maker(vs)
    
    # Store data for plotting
    angle_degrees.append(180/np.pi*theta_array[index])
    decay_coeff_real_0.append(np.real(a_actual[0]))
    decay_coeff_imag_2.append(np.imag(a_actual[2]))
    decay_coeff_real_1.append(np.real(a_actual[1]))
    decay_coeff_real_2.append(np.real(a_actual[2]))
    
    C = np.zeros((3,3)).astype(np.complex128)
    
    for i in range(3):
        for j in range(3):
            C[i,j] = B_actual[j]*beta_actual[i,j]
    
    P1 = 0
    for j in range(3):
        for k in range(3):
            P1 = P1 + np.pi*freq_0/(a_actual[j] + np.conjugate(a_actual[k]))*(np.conjugate(C[0,k])*(c_prime[0,0,0,0]*C[0,j] - 1.0j*a_actual[j]*c_prime[0,0,1,1]*C[1,j] -1.0j*a_actual[j]*c_prime[0,0,1,2]*C[2,j])
            + np.conjugate(C[1,k])*(-1.0j*a_actual[j]*c_prime[0,1,0,1]*C[0,j] + c_prime[0,1,0,1]*C[1,j] + c_prime[0,2,0,1]*C[2,j])
            + np.conjugate(C[2,k])*(-1.0j*a_actual[j]*c_prime[0,2,0,1]*C[0,j] + c_prime[0,2,0,1]*C[1,j] + c_prime[0,2,0,2]*C[2,j]))
            
    print("++++++")
    amplitude = np.abs(np.sum(C[1,:])) #amplitude of motion normal to surface
    print("Angle:", 180/np.pi*theta_array[index], "Degrees")
    print("Velocity:", vs[0], "m/s")
    print("Cperp:", np.real(amplitude/np.sqrt(W*P1)), "m/sqrt(W)")
    print("------")

# Plot decay coefficients
plt.figure(figsize=(10, 6))
plt.scatter(angle_degrees, decay_coeff_real_0, color="black", label="Real(a[0])")
plt.scatter(angle_degrees, decay_coeff_imag_2, color="blue", label="Imag(a[2])")
plt.scatter(angle_degrees, decay_coeff_real_1, color="red", label="Real(a[1])")
plt.scatter(angle_degrees, decay_coeff_real_2, color="green", label="Real(a[2])")
plt.xlabel("Angle (Deg)")
plt.ylabel("Decay Coefficient")
plt.legend()
plt.title("Decay Coefficients vs Angle")
plt.grid(True, alpha=0.3)
plt.show()

# Plot velocity vs angle
plt.figure(figsize=(10, 6))
plt.scatter(180/np.pi*theta_array, velocity_array, color="black", s=50)
plt.xlabel("Angle (Deg)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Angle")
plt.grid(True, alpha=0.3)
plt.show()

# Save results to dataframe
results_df = pd.DataFrame({
    'angle_degrees': angle_degrees,
    'velocity_ms': velocity_array,
    'decay_coeff_real_a0': decay_coeff_real_0,
    'decay_coeff_imag_a2': decay_coeff_imag_2,
    'decay_coeff_real_a1': decay_coeff_real_1,
    'decay_coeff_real_a2': decay_coeff_real_2
})

# Save to CSV file
results_df.to_csv('elastic_tensor_results2.csv', index=False)
print(f"Results saved to elastic_tensor_results.csv")
print(f"Dataframe shape: {results_df.shape}")
print(results_df.head())