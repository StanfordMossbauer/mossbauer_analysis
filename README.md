# Mossbauer Analysis

This repository provides tools and examples for analyzing Mossbauer spectroscopy data, including theoretical modeling and experimental data analysis.

## 1. Getting Started

Install dependencies and the package in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Contents

- **`mossbauer_analysis/`**: Main package containing classes for theory modeling and data analysis
- **`examples/`**: Data analysis and theory calculation examples, including SAW paper plot examples  
- **`data/`**: Raw experimental spectra and calibration data
- **`notebooks/`**: Analysis notebooks, and experiment theory calculations 
- **`mossbauer_theory_old_module/`**: Legacy theory code from Joey's original implementation



## 3. Theory Example
```python
from mossbauer_analysis.mossbauer_theory import CobaltRhodium, alphaFe, Mossbauer
import numpy as np

# Create source and absorber
source = CobaltRhodium()
absorber = alphaFe()
absorber.thickness_m = 25e-6  # 25 micron thickness
absorber.abundance = 0.0212   # natural Fe57 abundance

# Create Mossbauer measurement
moss = Mossbauer(source, absorber)

# Calculate transmission spectrum
v = np.linspace(-10, 10, 1000)  # velocity range in mm/s
transmission = moss.total_transmission_rate1(v)
```

## 4. Experimental Data Analysis Example

```python
from mossbauer_analysis.ironanalytics_load import read_ironanalytics_data
from mossbauer_analysis.ironanalytics_analyze import fit_and_calibrate
from mossbauer_analysis.fit_functions import six_peak_lorentzian_poly2

# Load experimental data
dir = 'data/SAW_spectra/calibration_spectra/'
data = read_ironanalytics_data(dir, 'A00051', offset=-3)

# Set up fitting parameters
fit_guess = {
    'contrasts': [-1e2, -1e2, -1e2],
    'resonances': [-4.32, -2.55, -0.78, 0.55, 2.31, 4.09],
    'fullwidth': 0.2,
    'poly': [3e6, 1e2, 1e1]
}

params = {
    'directory': dir,
    'id': 'A00051',
    'fitfunction': six_peak_lorentzian_poly2,
    'calibration_resonances': [-5.48, -3.25, -1.01, 0.66, 2.90, 5.13]
}

# Fit and calibrate spectrum
results = fit_and_calibrate(params=params, fit_guess=fit_guess, plot=True)
```

## Citation
If you use this code or data in your research, please cite appropriately.

## License
See `LICENSE` for details.


