import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from datetime import datetime, timedelta

from mossbauer_analysis.mossbauer_theory import CobaltRhodium, CobaltFe, KFeCy, alphaFe, Mossbauer    #import classes for sources, absorbers, and the mossbauer transmission spectrum
from mossbauer_analysis.fit_functions import linear, six_peak_lorentzian_poly2, single_peak_lorentzian_poly2, poly5
from mossbauer_analysis.ironanalytics_load import read_ironanalytics_data, print_ironanalytics_metadata 
import mossbauer_analysis.utils as u


def update_fitparams_from_list(fitparams, p):
    """
    Update fit parameters from a flat array of parameters.
    """
    updated = {}
    i = 0
    for k, v in fitparams.items():
        n = len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1
        if n == 1:
            updated[k] = p[i]
        else:
            updated[k] = p[i:i+n].tolist()
        i += n
    return updated


def fit_and_calibrate(params, fit_guess, data = False, calibrate_resonances=True, plot=True):
    
    params = params.copy()

    # Unpack
    directory = params.get('directory', '')
    id = params.get('id', '')
    offset = params.get('offset', 0)
    side = params.get('side', '')
    npeaks = params.get('npeaks', 0)
    fitfunction = params.get('fitfunction', None)
    calibration_resonances = params.get('calibration_resonances', [])


    dat = read_ironanalytics_data(directory, id, offset = offset)
    velocities = dat.velocity_list
    counts = getattr(dat, f"data_{side}")
    description = dat.description   

    #load data
    if data:
        velocities = data[0]
        counts = data[1]
        description = "theory curve"

    #fit spectrum
    x = velocities
    y = counts
    p0 = np.concatenate([np.atleast_1d(v) for v in fit_guess.values()])
    p,dp = u.fit(fitfunction,x, y, p0,fullout=False)
    fit_result = update_fitparams_from_list(fit_guess, p)
    fit_result_errors = update_fitparams_from_list(fit_guess, dp) 

    baseline = poly5((fit_result['poly'] + [0]*6)[:6], x)
    
    #evaluate fit
    norm_res = (y - fitfunction(p, x))/np.sqrt(fit_result['poly'][0])
    reduced_chi2 = np.sum(norm_res**2)/(len(y)-len(p))


    #output parameters
    params['description'] = description
    params['fitfunction'] = fitfunction
    params['reduced_chi2'] = reduced_chi2
    params['total_rate'] = dat.data.sum()/dat.total_time
    params['total_counts'] = dat.data.sum()
    params['total_time'] = dat.total_time
    params['total_rate_fit'] = fit_result['poly'][0]*len(dat.data)/dat.total_time
    params['total_rate_fit_error'] = np.sqrt(fit_result['poly'][0]*len(dat.data))/dat.total_time
    params['resonances'] = fit_result['resonances']
    params['contrasts'] = fit_result['contrasts']
    params['fullwidths'] = fit_result['fullwidths']
    params['poly'] = fit_result['poly']
    params['y'] = counts
    params['x'] = velocities


    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle(f"{dat.name}        {dat.description}        {side}        {fitfunction.__name__}") 
        ax[0,0].plot(x, y, '.', markersize = 2, label='data')
        ax[0,0].plot(x, fitfunction(p, x), 'k' ,label='fit')
        ax[0,0].plot(x, baseline, 'k--', label='baseline')

        #ax[0,0].plot(x, fitfunction(p0, x), label='fit_guess')
        #ax[0,0].axhline(fit_result['poly'][0])
        text = (
            f"relative noise: root(counts)/conunts {1/np.sqrt(fit_result['poly'][0]): .3f}\n"
            f"contrasts: {[f'{c:.3f}' for c in np.atleast_1d(fit_result['contrasts'])]}\n"
            f"resonances: {[f'{r:.2f}' for r in np.atleast_1d(fit_result['resonances'])]}\n"
            f"fullwidths: {[f'{r:.2f}' for r in np.atleast_1d(fit_result['fullwidths'])]}\n"
            f"poly: {[f'{p:.2e}' for p in fit_result['poly']]}\n"
        )

        ax[0, 0].annotate(text,xy=(0.1, 0.1),xycoords='axes fraction', fontsize=9, color='k')
        ax[0,0].set_xlabel('velocity [mm/s]')
        ax[1,0].plot(x, norm_res, '.',markersize = 2)
        ax[1,0].plot(x, gaussian_filter(norm_res,10), 'r.',markersize = 2)
        ax[1,0].annotate(f"reduced chi2: {reduced_chi2:.2f}", xy=(0.1, 0.1), xycoords='axes fraction', fontsize=9, color='k')

    
   

        
    if calibrate_resonances:
        def poly1(p,x):
            return p[0] + p[1]*x
        
        
        calib_fitfunction = poly1
        p0 = [0,1]

        
        x = np.array(fit_result['resonances'])
        y = np.array(calibration_resonances)
        #print(x,y)
        dy = np.sqrt(((dat.velocity_max/512)/np.sqrt(12))**2 + np.array(fit_result_errors['resonances'])**2) # add uncertainty from velocity binning
        #dy =  fit_result_errors['resonances']
        #dy = ((dat.velocity_max/512)/np.sqrt(12))
        p,dp = u.fit(calib_fitfunction, x, y, p0=p0, fullout = False)
        norm_res = (y - calib_fitfunction(p,x))/dy
        reduced_chi2_calib = np.sum(norm_res**2)/(len(y) - len(p))
        p_calibration = p

        params['velocity_calibration'] = p_calibration # real_width = C*measured width 
        params['fullwidths_calibrated'] = list(p_calibration[1]*np.array(fit_result['fullwidths']))
        params['x_calibrated'] = p_calibration[0] + p_calibration[1]*velocities
        params['y_norm'] = counts/baseline # remove baseline and convert to counts/s


            
        if plot: 
            
            ax[0,1].plot(x, y, '.')
            ax[0,1].plot(np.linspace(-10,10,100), calib_fitfunction(p_calibration,np.linspace(-10,10,100)))
            ax[0,1].plot(np.linspace(-10,10,100), calib_fitfunction(p0,np.linspace(-10,10,100)))
            ax[0,1].annotate(f"calibration factor: real_v = {p_calibration[0]:.2f} + {p_calibration[1]:.2f}*measured_x", xy=(0.1, 0.1), xycoords='axes fraction', fontsize=9, color='k')
            ax[0,1].set_xlim(-7,7) 
            ax[1,1].plot(calibration_resonances, norm_res, '.')
            ax[1,1].annotate(f"reduced chi2: {reduced_chi2_calib:.2f}\nfit: {calib_fitfunction.__name__}\n dy =rot([(b-a)/sqrt(12)]^2+dy_fit^2])", xy=(0.1, 0.1), xycoords='axes fraction', fontsize=9, color='k')
            ax[1,1].set_xlim(-7,7)
            pass




            


    return params




def integrate_activity(start_timestamp,
                        duration_seconds, 
                        efficiency = 1,
                        halflife_days=271.8, 
                        activity_0 = 50e-3*3.7e10, 
                        reference_timestamp = datetime.strptime("20250108", "%Y%m%d").timestamp(), print_current_activity = False):

    delta_seconds =  start_timestamp - reference_timestamp
    
    halflife_seconds = halflife_days * 24 * 3600
    lambda_decay = np.log(2) / halflife_seconds
    
    # Convert from mCi to Bq (1 mCi = 3.7e10 Bq)
    activity_start = activity_0 * np.exp(-lambda_decay * delta_seconds)
    if print_current_activity: print(f"current activity {activity_start/3.7e7:.2e} mCi\noriginal activity {activity_0/3.7e7:.2e} mCi")
    
    # Integrate activity over measurement time
    # Integral of A(t) = A0 * exp(-λt) from t=0 to t=T is:
    # (A0/λ) * (1 - exp(-λT))
    integrated_activity = efficiency * (activity_start / lambda_decay) * (1 - np.exp(-lambda_decay * duration_seconds))
    
    return integrated_activity


    
if __name__ == "__main__":
    

    fit_guess = {
            'contrasts': [-0.05, -0.02, -0.01],
            'resonances':[-4.32,-2.55,-0.78,0.55,2.31,4.09],
            'fullwidth': 0.2,
            'poly': [4.5e5, 0, 0],  
            }


    params = {  'directory':  'C:/Users/magrini/Documents/programming/mossbauer_analysis/data/SAW_spectra/calbration_spectra',
                'id': 'A00043',
                'offset': -3,
                'side': 'left',
                'npeaks': 6,
                'fitfunction': six_peak_lorentzian_poly2,
                'calibration_resonances': [-5.48, -3.25, -1.01, 0.66, 2.90, 5.13],
            
            } 

    params = fit_and_calibrate(params = params , fit_guess = fit_guess, calibrate_resonances = True, plot=True)
    print(params)
    plt.show()
    

    fit_guess = {
            'contrasts': -0.2,
            'resonances': -0.078,
            'fullwidth': 0.2,
            'poly': [1e5, 1e-3, 1e-5],  
            }


    params = {  'directory':  'C:/Users/magrini/Documents/programming/mossbauer_analysis/data/SAW_spectra/calbration_spectra',
                'id': 'A00049',
                'offset': -3,
                'side': 'right',
                'npeaks': 6,
                'fitfunction': single_peak_lorentzian_poly2,
                'calibration_resonances': -0.78,
            
        } 


    #params_out = fit_and_calibrate(params = params , fit_guess = fit_guess, calibrate_resonances = False, plot = True)
    #print(params)
    #plt.show()