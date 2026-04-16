import numpy as np

def linear(p, x):
    a = p
    return a * x


def poly1(p, x):
    a , b = p
    return a + b * x

def poly2(p, x):
    return p[0] * (1 + p[1]*x +  p[2]*x**2)

def poly5(p, x):
    return p[0] * (1 + p[1]*x +  p[2]*x**2 +  p[3]*x**3 + p[4]*x**4 +  p[5]*x**5)

def single_peak_lorentzian(p, x):
    c0, x0, fullwidth, offset = p 
    return offset * ( 1 + c0 *(fullwidth/2)**2 / ((x - x0)**2 + (fullwidth/2)**2))


def single_peak_lorentzian_poly1(p, x):
    c0, x0, fullwidth, offset, a1 = p 
    return offset * ( 1 + a1*x + c0*(fullwidth/2)**2 / ((x - x0)**2 + (fullwidth/2)**2))

def single_peak_lorentzian_poly2(p, x):
    c0, x0, fullwidth, offset, a1, a2 = p 
    poly = 1 + a1*x +a2*x**2
    return offset * ( poly + c0*(fullwidth/2)**2 / ((x - x0)**2 + (fullwidth/2)**2))

def single_peak_lorentzian_poly3(p, x):
    c0, x0, fullwidth, offset, a1, a2, a3 = p 
    poly = 1 + a1*x + a2*x**2 +a3*x**3
    return offset * ( poly + c0 *(fullwidth/2)**2/ ((x - x0)**2 + (fullwidth/2)**2))



def six_peak_lorentzian(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + peaks)


def six_peak_lorentzian_poly1(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset, a1 = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + a1*x + peaks)

def six_peak_lorentzian_poly2(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset, a1, a2 = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + a1*x +  a2*x**2 + peaks)

def six_peak_lorentzian3widths_poly2(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, g0, g1, g2, offset, a1, a2 = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    fullwidths = [g0, g1, g2, g2, g1, g0]
    peaks = np.zeros_like(x, dtype=float)


    for c_i, x_i, g_i in zip(constrasts, resonances, fullwidths):
        peaks += c_i*(g_i/2)**2 / ((x - x_i)**2 + (g_i/2)**2)
    
    return offset * (1 + a1*x +  a2*x**2 + peaks)


def six_peak_lorentzian_poly3(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset, a1, a2 ,a3 = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + a1*x +  a2*x**2 +  a3*x**3 + peaks)

def six_peak_lorentzian_poly4(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset, a1, a2 ,a3, a4 = p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + a1*x +  a2*x**2 +  a3*x**3 + a4*x**4 + peaks)

def six_peak_lorentzian_poly5(p, x):
    
    c0, c1, c2, x0, x1, x2, x3, x4, x5, fullwidth, offset, a1, a2 ,a3, a4 , a5= p
    constrasts = [c0, c1, c2, c2, c1, c0]
    
    resonances = [x0, x1, x2, x3, x4, x5]
    peaks = np.zeros_like(x, dtype=float)

    for c_i, x_i in zip(constrasts, resonances):
        peaks += c_i*(fullwidth/2)**2 / ((x - x_i)**2 + (fullwidth/2)**2)
    
    return offset * (1 + a1*x +  a2*x**2 +  a3*x**3 + a4*x**4 +  a5*x**5 + peaks)

