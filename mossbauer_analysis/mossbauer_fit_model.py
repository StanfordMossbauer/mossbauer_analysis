

import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import leggauss
from .mossbauer_theory import Mossbauer, _lorentzian_s
import mossbauer_analysis.utils as u



Nq = 1600  # number of quadrature points for integration
W = 20     # width of integration region in mm/s (should cover the source spectrum
# Create integration grids for this calculation
xq, wq = leggauss(Nq)
Egrid = W * xq
Wgrid = W * wq



def source_spectrum_matrix(Egrid, v, source):
    v = np.asarray(v)
    Emat = Egrid[:, None]
    vmat = v[None, :]

    spec = 0.0
    for coef, Eres in zip(source.transition_coefficients, source.Eres):
        spec = spec + coef * _lorentzian_s(Emat, (Eres - vmat), source.Gamma_mms)
    return source.fs * spec 


def S_fast(v, absorber, source):

    moss = Mossbauer(source, absorber)

    sigma = moss.cross_section(Egrid)  
    tau = sigma * absorber.fa * absorber.nM * absorber.thickness_gcm2_Fe57
    att = np.exp(-tau)                     
    src = source_spectrum_matrix(Egrid, v, source)

    # integral over Egrid: sum_k w_k * src_k,i * att_k
    resonant = (Wgrid[:, None] * (src * att[:, None])).sum(axis=0)

    resonant *= moss.non_resonant_attenuation()

    return moss.nonresonant_transmission_rate() + resonant


def S_slow(v, absorber, source):
    moss = Mossbauer(source, absorber)
    return moss.total_transmission_rate(v)



def optimize_linewidth_s(xdata, ydata, gamma_s_grid, absorber, source, x_model, fast=True):

    params_best = None
    params_all = []
    sse_all = []

    for gamma_s in gamma_s_grid:
        source.Gamma_ev = gamma_s
        source.update_params()
    

        # compute theory curve for this gamma_s on x_model
        if fast:
            y_model = S_fast(x_model, absorber, source)
        else:
            y_model = S_slow(x_model, absorber, source)

        # interpolation of the model curve
        model_interp = interp1d(x_model, y_model, kind="cubic")

        # Fit parameters: A, B, x0, s
        def fit_model(p, x):
            A, B, x0, s = p
            return A * model_interp(s * (x - x0)) + B

        # initial guess (use previous best as warm start when available)
        p0 = [1.0, 1, 0.15, 1.3]
        p, dp = u.fit(fit_model, xdata, ydata, p0=p0, fullout=False)
        A, B, x0, s = p

        # compute SSE / chi^2-like score for this gamma_s
        r = ydata - fit_model(p, xdata)
        SSE = np.sum(r*r)
        params = dict(gamma_s=gamma_s, A=A, B=B, x0=x0, s=s, SSE=SSE)
        
        params_all.append(params)
        sse_all.append(SSE)

        if (params_best is None) or (SSE < params_best["SSE"]):
            params_best = params

     # Compute calibrated data and fitted values using best parameters
    x_calibrated = params_best["s"] * (xdata - params_best["x0"])
    
    source.Gamma_ev = float(params_best["gamma_s"])
    source.update_params()
    y_model = S_slow(x_calibrated, absorber, source)
    y_fit = params_best["A"] * y_model + params_best["B"]
    
    return params_best, np.array(sse_all), x_calibrated, y_model, y_fit


def optimize_thickness(xdata, ydata, t_grid, absorber, source, x_model, fast=True):

    params_best = None
    params_all = []  # store results for plotting (chi2 vs t)
    sse_all = []

    for t in t_grid:
        absorber.thickness_m = float(t)
        absorber.update_params()

        # compute theory curve for this thickness on x_model
        if fast:
            y_model = S_fast(x_model, absorber, source)
        else:
            y_model = S_slow(x_model, absorber, source)

        # interpolation of the model curve
        model_interp = interp1d(x_model, y_model, kind="cubic")

        # Fit parameters: A, B, x0, s
        def fit_model(p, x):
            A, B, x0, s = p
            return A * model_interp(s * (x - x0)) + B
        # initial guess (use previous best as warm start when available)
        p0 = [1.0, 1,-0.1, 0.05, 1.3]
        p, dp = u.fit(fit_model, xdata, ydata, p0=p0, fullout=False)
        A, B, x0, s = p

        # compute SSE / chi^2-like score for this t
        r = ydata - fit_model(p, xdata)
        SSE = np.sum(r*r)

        rec = dict(t=t, A=A, B=B, C=C, x0=x0, s=s, SSE=SSE)
        params_all.append(rec)
        sse_all.append(SSE)

        if (params_best is None) or (SSE < params_best["SSE"]):
            params_best = rec

    
    # Compute calibrated data and fitted values using best parameters
    x_calibrated = params_best["s"] * (xdata - params_best["x0"])
    
    absorber.thickness_m = float(params_best["t"])
    absorber.update_params()
    y_model = S_slow(x_calibrated, absorber, source)
    y_fit = params_best["A"] * y_model + params_best["B"] + params_best["C"] * xdata
    
    return params_best, np.array(sse_all), x_calibrated, y_model, y_fit


def fit_gamma_and_thickness(
    xdata, ydata,
    absorber, source,
    gamma_center, gamma_span, gamma_N,
    t_center, t_span, t_N,
    x_model,
    n_iter=2,
    fast=True
):
    """
    Alternating grid search for gamma_s and thickness.
    Starts from (gamma_center, t_center). Each iteration shrinks spans.
    Returns (best_gamma, best_t, best_records) where best_records contains
    last iteration's calibrated x and fitted curve for quick plotting.
    """
    best_records = {}

    gamma = float(gamma_center)
    t = float(t_center)

    for it in range(n_iter):
        # --- Step 1: fit gamma on a grid ---
        gamma_grid = np.linspace(gamma - gamma_span, gamma + gamma_span, gamma_N)
        absorber.thickness_m = t
        absorber.update_params()

        best_g, sse_g, x_cal_g, y_model_g, y_fit_g = optimize_linewidth_s(
            xdata, ydata, gamma_grid, absorber, source, x_model, fast=fast
        )
        gamma = float(best_g["gamma_s"])

        # --- Step 2: fit thickness on a grid ---
        t_grid = np.linspace(t - t_span, t + t_span, t_N)
        # keep thickness positive
        t_grid = t_grid[t_grid > 0]

        source.Gamma_ev = gamma
        source.update_params()

        best_t, sse_t, x_cal_t, y_model_t, y_fit_t = optimize_thickness(
            xdata, ydata, t_grid, absorber, source, x_model, fast=fast
        )
        t = float(best_t["t"])

        # store last iteration outputs for plotting
        best_records = dict(
            iter=it,
            gamma=gamma,
            t=t,
            x_calibrated=x_cal_t,
            y_model=y_model_t,
            y_fit=y_fit_t,
            sse_gamma=sse_g,
            gamma_grid=gamma_grid,
            sse_t=sse_t,
            t_grid=t_grid,
        )

        # shrink search windows for the next iteration
        gamma_span *= 0.3
        t_span *= 0.3

    return gamma, t, best_records



import numpy as np
