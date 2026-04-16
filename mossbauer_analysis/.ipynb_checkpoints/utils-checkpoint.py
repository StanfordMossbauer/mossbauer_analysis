import time
from numpy import *
import numpy as np
from scipy.interpolate import  splrep, splev
from scipy.optimize import leastsq, least_squares as least_sqares

from scipy.optimize import curve_fit
from scipy.optimize import minimize

#import string
def convert_image(image: np.ndarray) -> np.ndarray:
    row, col4 = image.shape
    col = col4 // 4
    im0 = image[:, 0*col : 1*col]
    im1 = image[:, 1*col : 2*col]
    im2 = image[::-1, 2*col : 3*col]
    im3 = image[::-1, 3*col : 4*col]
    return np.vstack([np.hstack([im2, im3])
                      ,np.hstack([im0, im1])])



def loop(fun, sleep=0):
	while True:
		try:
			fun()
			time.sleep(sleep)
		except KeyboardInterrupt:
			break

class interpolation:
	"""
	class interpolation(x,y,k=3)

	Description:
		returns a callable object giving a spline interpolation  of x,y  data
	"""
	def __init__(self, x,y,k=3):
		self.datax=array(x)
		self.datay=array(y)
		self.datax.sort()
		self.datay.sort()
		self.tck=splrep(x,y,k=k)

	def __call__(self, x):
		if len(shape(x))==0:
			return self._singleval(x)
	
		elif len(shape(x))==1:
			return array(map(self._singleval,x))

	def _singleval(self,x):
		return splev(x,self.tck)


def fit(fun,x,y,p0,xmin=None, xmax=None,fullout=True):

	if not xmin:
		xmin=min(x)
	if not xmax:
		xmax=max(x)
	cond=greater_equal(x,xmin)*less_equal(x,xmax)
	x=compress(cond,x)
	y=compress(cond,y)



	def cost(p,x,y):
		return y-fun(p,x)

	[p, cov_p, infodict,ier ,m]=leastsq(cost, p0, args=(x,y), full_output=1)
	res=y-fun(p,x)
	sig=res.std()
	#print(p, cov_p)
	dp=sqrt(diag(cov_p))*sig
	
	if fullout:
		npar=len(p0)
		print()
		print()
		print("Mean squared residual: %.3e" % (res**2).mean())
		print()
		print( "Covariance Matrix:")
		print()
		print( 5*" ",end='')
		for i in  range(npar):
			print(("p["+str(i)+"]").rjust(6),end='')
		print(	)

		for i in range(npar):
			print(("p["+str(i)+"]").rjust(5),end='')    
			for j in range(0,i+1): #per i=0 ritorna [0] invece di []=range(0)
				print(("%.2f" % (cov_p[i,j]/sqrt(cov_p[i,i]*cov_p[j,j]))).rjust(6),end='')
			print()
		print() 

		print("Final Parameters:")
		print() 
		for i in range(npar):
			print(" p["+str(i)+"]"+\
					" = %.5e +/- %.1e (%.2f%%)" %\
					(p[i],dp[i],dp[i]/abs(p[i])*100.))
		print()

	return [p,dp]

#same as fit but allows for fixed params

def fit_fix(fun, x, y, p0, p_fix ,xmin=None, xmax=None,fullout=True):


	if not xmin:
		xmin = min(x)
	if not xmax:
		xmax = max(x)
	cond = greater_equal(x, xmin) * less_equal(x, xmax)
	x = compress(cond, x)
	y = compress(cond, y)

	def cost(p, x, y):
		return y - fun(p, p_fix, x)


	[p, cov_p, infodict, ier, m] = leastsq(cost, p0, args=(x, y), full_output=1)
	res = y - fun(p, p_fix, x)
	sig = res.std()
	# print(p, cov_p)
	dp = sqrt(diag(cov_p)) * sig

	if fullout:
		npar = len(p0)
		print()
		print()
		print("Mean squared residual: %.3e" % (res ** 2).mean())
		print()
		print("Covariance Matrix:")
		print()
		print(5 * " ", end='')
		for i in range(npar):
			print(("p[" + str(i) + "]").rjust(6), end='')
		print()

		for i in range(npar):
			print(("p[" + str(i) + "]").rjust(5), end='')
			for j in range(0, i + 1):  # per i=0 ritorna [0] invece di []=range(0)
				print(("%.2f" % (cov_p[i, j] / sqrt(cov_p[i, i] * cov_p[j, j]))).rjust(6), end='')
			print()
		print()

		print("Final Parameters:")
		print()
		for i in range(npar):
			print(" p[" + str(i) + "]" + \
				  " = %.5e +/- %.1e (%.2f%%)" % \
				  (p[i], dp[i], dp[i] / abs(p[i]) * 100.))
		print()

	return [p, dp]


#same as above but also returns mean square resuiduals and doesnt only print it

def fit_residuals(fun, x, y, p0, xmin=None, xmax=None, fullout=True):
	if not xmin:
		xmin = min(x)
	if not xmax:
		xmax = max(x)
	cond = greater_equal(x, xmin) * less_equal(x, xmax)
	x = compress(cond, x)
	y = compress(cond, y)

	def cost(p, x, y):
		return y - fun(p, x)

	[p, cov_p, infodict, ier, m] = leastsq(cost, p0, args=(x, y), full_output=1)
	res = y - fun(p, x)
	sig = res.std()
	dp = sqrt(diag(cov_p)) * sig

	if fullout:
		npar=len(p0)
		print()
		print()
		print("Mean squared residual: %.3e" % (res**2).mean())
		print()
		print( "Covariance Matrix:")
		print()
		print( 5*" ",end='')
		for i in  range(npar):
			print(("p["+str(i)+"]").rjust(6),end='')
		print(	)

		for i in range(npar):
			print(("p["+str(i)+"]").rjust(5),end='')    
			for j in range(0,i+1): #per i=0 ritorna [0] invece di []=range(0)
				print(("%.2f" % (cov_p[i,j]/sqrt(cov_p[i,i]*cov_p[j,j]))).rjust(6),end='')
			print()
		print() 

		print("Final Parameters:")
		print() 
		for i in range(npar):
			print(" p["+str(i)+"]"+\
					" = %.5e +/- %.1e (%.2f%%)" %\
					(p[i],dp[i],dp[i]/abs(p[i])*100.))
		print()

	return [p, dp, (res ** 2).mean()]

#this also considers error bars on the y axis, problem here is the covariance matrix, its not really rescaled to that

def fit_error(fun, x, y, yerr, p0, xmin=None, xmax=None, fullout=True):
	if not xmin:
		xmin = min(x)
	if not xmax:
		xmax = max(x)
	cond = greater_equal(x, xmin) * less_equal(x, xmax)
	x = compress(cond, x)
	y = compress(cond, y)

	def cost(p, x, y,yerr):
		return (y - fun(p, x))/yerr*max(yerr)

	[p, cov_p, infodict, ier, m] = leastsq(cost, p0, args=(x, y,yerr), full_output=1)
	res = y - fun(p, x)
	sig = res.std()
	try:
		dp = sqrt(diag(cov_p)) * sig
	except:
		dp=0.
		print('couldnt find errors.. one parameters results in too flat curvature.. try redefining the function')
	if fullout:
		npar=len(p0)
		print()
		print()
		print("Mean squared residual: %.3e" % (res**2).mean())
		print()
		print( "Covariance Matrix:")
		print()
		print( 5*" ",end='')
		for i in  range(npar):
			print(("p["+str(i)+"]").rjust(6),end='')
		print(	)

		for i in range(npar):
			print(("p["+str(i)+"]").rjust(5),end='')    
			for j in range(0,i+1): #per i=0 ritorna [0] invece di []=range(0)
				print(("%.2f" % (cov_p[i,j]/sqrt(cov_p[i,i]*cov_p[j,j]))).rjust(6),end='')
			print()
		print() 

		print("Final Parameters:")
		print() 
		for i in range(npar):
			print(" p["+str(i)+"]"+\
					" = %.5e +/- %.1e (%.2f%%)" %\
					(p[i],dp[i],dp[i]/abs(p[i])*100.))
		print()
        
        
	return [p, dp]



def fit_new(fun, x, y, yerr, p0=None,bounds=None , fullout=True):

	[p, cov_p] = curve_fit(fun, x, y, p0=p0, sigma=yerr, bounds=bounds)
	dp = sqrt(diag(cov_p))

	return [p, dp]


def fitMLE(fun, x, y, params, fullout=False):
	# For MLE, minimize the negative log likelihood
	def neglnlike(params, x, y):
		# h/t James Kuszlewicz
		# Data generated as Chi^2_2 about a Lorentzian plus offset
		model = fun(params, x)
		output = sum(log(model) + y / model)
		# Check that this is valid, returning large number if not
		if not isfinite(output):
			return 1.0e30
		return output

	res = minimize(neglnlike, params, args=(x, y), method='Nelder-Mead')
	if fullout:
		print(res)

	return res.x


from scipy.optimize import least_squares
import numpy as np

def fit_bounds(fun, x, y, p0, xmin=None, xmax=None, bounds=(-inf, inf), fullout=True):

    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)

    cond = (x >= xmin) & (x <= xmax)
    x = x[cond]
    y = y[cond]

    def cost(p):
        return y - fun(p, x)

    result = least_squares(cost, p0, bounds=bounds)

    p = result.x
    res = y - fun(p, x)
    sig = std(res)

    # Estimate covariance matrix if possible
    try:
        J = result.jac
        JTJ = J.T @ J
        cov_p = linalg.inv(JTJ)
        dp = sqrt(diag(cov_p)) * sig
        corr = cov_p / outer(sqrt(diag(cov_p)), sqrt(diag(cov_p)))
    except:
        cov_p = None
        dp = full_like(p, nan)
        corr = None

    if fullout:
        print("\nMean squared residual: %.3e" % mean(res**2))
        
        if cov_p is not None:
            print("\nCovariance Matrix (correlations):\n")
            npar = len(p)
            print("     ", end='')
            for i in range(npar):
                print(f"p[{i}]".rjust(7), end='')
            print()
            for i in range(npar):
                print(f"p[{i}]".rjust(5), end='')
                for j in range(i+1):
                    print(f"{corr[i,j]:7.2f}", end='')
                print()
        else:
            print("\nCovariance matrix could not be estimated.\n")

        print("\nFinal Parameters:\n")
        for i in range(len(p)):
            err = dp[i]
            perr = 100 * err / abs(p[i]) if p[i] != 0 else nan
            print(f" p[{i}] = {p[i]:.5e} +/- {err:.1e} ({perr:.2f}%)")
        print()

    return [p, dp]



'''
#here with numpy as np 

def fitMLE(fun, x, y, params, fullout=False):
	# For MLE, minimize the negative log likelihood
	def neglnlike(params, x, y):
		# h/t James Kuszlewicz
		# Data generated as Chi^2_2 about a Lorentzian plus offset
		model = fun(params, x)
		output = np.sum(np.log(model) + y / model)
		# Check that this is valid, returning large number if not
		if not np.isfinite(output):
			return 1.0e30
		return output

	res = minimize(neglnlike, params, args=(x, y), method='Nelder-Mead')
	if fullout:
		print(res)

	return res

'''