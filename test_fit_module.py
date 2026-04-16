"""
Test script for the new mossbauer_fit_model module
"""

from mossbauer_analysis.mossbauer_fit_model import profile_scan_t
from mossbauer_analysis.ironanalytics_load import read_ironanalytics_data  
from mossbauer_analysis.mossbauer_theory import CobaltRhodium, alphaFe
import numpy as np
import matplotlib.pyplot as plt

# Load data
directory = "C:/Users/magrini/Documents/programming/mossbauer_analysis/data/SpectraA/"
data_dep10 = read_ironanalytics_data(directory, "A00152", offset=-3, plot=False)

x_meas = data_dep10.velocity_list
y_meas = data_dep10.data_folded/max(data_dep10.data_folded)

# Setup source
source = CobaltRhodium()
source.Eres = [-0.11]  # from pipcorn
source.Gamma_ev = 7.9e-9
source.update_params()

# Setup absorber
absorber = alphaFe()
absorber.thickness_m = 1000e-9
absorber.abundance = 0.96
absorber.Eres = [-5.48, -3.25, -1.01, 0.66, 2.90, 5.13]  # from pipcorn
absorber.update_params()

# Define thickness grid
t_grid = np.linspace(100e-9, 1000e-9, 50)

# Run the fit
params, sse, x_calibrated, y_fit = profile_scan_t(x_meas, y_meas, t_grid, absorber, source)

print("Best fit parameters:")
print(params)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot SSE vs thickness
ax1.plot(t_grid * 1e9, sse, '-o', ms=3)
ax1.set_xlabel("Thickness (nm)")
ax1.set_ylabel("SSE")
ax1.grid(alpha=0.3)
ax1.set_title("SSE vs Thickness")

# Plot data and fit
ax2.plot(x_calibrated, y_meas, '.', ms=3, label="Data")
ax2.plot(x_calibrated, y_fit, '-', lw=2, label="Best Fit")
ax2.set_xlabel("Velocity (mm/s)")  
ax2.set_ylabel("Transmission")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("Data vs Fit")

plt.tight_layout()
plt.show()