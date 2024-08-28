""" 
hybesscy.pyx is a module that computes Hyperspherical Bessel functions
(HBF) of complex order. The method utilized to perform the
computation is based on recursive relations and the modified
Lentz algorithm. It is a modification to Cython from the code
presented in the file hyperspherical.c in Class.
References: T. Tram Computation of hyperspherical Bessel functions 
(arXiv:1311.0839v2)

Functions:

HyperBesselComplex : Computes HBF

HyperBesselPrimeComplex : Computes derivative of HBF

HyperBesselPrime2Complex : Computes second derivative of HBF

BackwardsRecurrenceComplex : Computes HBF using backward recursion

ForwardsRecurrenceComplex : Computes HBF using forward recursion

ContinuedFractionComplex : Computes the continued fraction needed
    for backward recursion

ximoverkc : Numerical factor

EpsilonComplex : Computes the electric part of the radial function

BetaComplex : Computes the magnetic part of the radial function

epsilon : Calls EpsilonComplex

beta : Calls BetaComplex 

epsbeta_for_tensor : Computes the radial functions for the tensor part
    of the multipoles

epsbeta_for_vector : Computes the radial functions for the vector part
    of the multipoles
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sin, cos, tan, asin, sinh, cosh, asinh, sqrt, fabs
cdef extern from "math.h" nogil:
    long double sinhl (long double)
    long double coshl(long double)
cdef extern from "complex.h" nogil:
    long double complex csinl(long double complex)
    long double complex csqrtl(long double complex)
    long double complex ctanl(long double complex)
    long double cabsl(long double complex)
    long double creall(long double complex)
    long double cimagl(long double complex)
    double complex I

cdef struct Geometric_variables:
    # Struct that contains some variables used in all functions
    long double r # dimensionless radial coordinate 
    long double dr_over_r 


cdef long double complex[:] HyperBesselComplex(long double chi,
                                               long double complex nu,
                                               int ell_max,
                                               Geometric_variables gvar):
    """Calculates HBF of complex order.
    
    It is only implemented for calK == -1

    Returns: Typed Memoryview

    Parameters
    ----------
    chi : double
        Comoving distance
    nu : long complex
        Modes of the Bianchi-FLRW matching
    ell_max : inthighest multipole 
        Largest multipole
    gvar : struct
        Geometrical variables

    Returns
    -------
    typed memoryview
     """

    cdef:
        long double complex [:] phi
        double xfwd = asinh(sqrt(ell_max * (ell_max + 1.)) / cabsl(nu))  # Critical point that defines
                                                                         # whether forwards or backwards recurrence is used
        

    # Calculating phi at chi/ellc given the order nu for multipoles 0 to ell_max + 1
    if chi< xfwd:
        phi = BackwardsRecurrenceComplex(chi, nu, ell_max, gvar)
    else:
        phi = ForwardsRecurrenceComplex(chi, nu, ell_max, gvar)

    return phi


cdef long double complex[:] HyperBesselPrimeComplex(long double complex[:] phi,
                                                    long double complex nu,
                                                    int ell_max, 
                                                    Geometric_variables gvar):
    """Calculates the derivative of HBF of complex order

    Parameters
    ----------
    phi : typed memoryview
        Hyperspherical Bessel function
    nu : complex
        Modes of the Bianchi-FLRW matching
    ell_max : int
        Largest multipole
    gvar : struct
        Geometrical variables

    Returns
    -------
    typed memoryview     
     """

    cdef:
        long double complex[:] dphi = np.empty(ell_max + 1, dtype = np.clongdouble)
        long double cotK = gvar.dr_over_r 
        long double complex nu2 = nu * nu
        int l
        
    for l in range(ell_max+1):
        dphi[l] = l * cotK * phi[l] - csqrtl(nu2 + (l + 1.) * (l + 1.)) *phi[l+1]
    return dphi

cdef long double complex[:] HyperBesselPrime2Complex(long double complex nu, 
                                                     int ell,
                                                     long double complex[:] phi,
                                                     long double complex[:] dphi,
                                                     Geometric_variables gvar):
    """Second derivative of HBF for complex order.

    Paramters
    ---------
    nu : complex
        Modes of the Bianchi-FLRW matching
    ell : int
        Largest multipole
    phi : typed memoryview
        Hyperspherical Bessel function
    dphi : typed memoryview
        Hyperspherical Bessel function
    gvar : struct
        Geometrical variables

    Returns
    -------
    typed memoryview    
    """

    cdef:
        long double x = gvar.dr_over_r 
        long double y2 = gvar.r * gvar.r 
        long double complex nu2 = nu * nu
        int l
        long double complex[:] d2phi = np.empty(ell+1, dtype = np.clongdouble)

    for l in range(ell + 1):
        d2phi[l] = -2 * x * dphi[l] -(nu2 + 1.-l * (l+1.) /y2) * phi[l]
    return d2phi 
       
cdef long double complex[:] BackwardsRecurrenceComplex(double x,
                                                       long double complex nu,
                                                       int ell,
                                                       Geometric_variables gvar):
    """Calculates the HBF using backward recurrence method for complex order
    
    For details, see Numerical Recipes in C chapter 6,
    Tram (2017): Computation of hyperspherical Bessel functions,
    Class's hyperspherical.c file.

    Parameters
    ----------
    x : double
        Argument
    nu : complex
        Degree
    ell : int
        Order
    gvar : struct
        geometrical variables

    Returns
    -------
    typed memoryview
    """

    cdef:
        long double complex ratio
        long double sinK = gvar.r 
        long double cotK = gvar.dr_over_r
        long double complex nu2 = nu * nu
        int i 
        Py_ssize_t j, k
        int sign_r, sign_i
        int nonconvergence = 1

    # Finding the first element of the sequence up to a multiplicative factor
    ContinuedFractionComplex(nu, ell, cotK, &ratio, &sign_r,  &sign_i, &nonconvergence)

    cdef:
        long double complex[:] phi = np.empty(ell + 2, dtype = np.clongdouble)
        long double complex phi0 = csinl(nu * x) / (nu * sinK)  # Phi at ell = 0
        double auxsign_r =  <double> sign_r
        double auxsign_i =  <double> sign_i
        long double complex phi1 = auxsign_r + I * auxsign_i  # Phi at ell + 1 up to a mult. factor
        long double complex denom = csqrtl(nu2 + (ell + 1) * (ell + 1))
        long double complex phi_p1 = phi1 * (ell * cotK - ratio)
        long double complex val

    phi[ell + 1] = phi_p1 / denom  # Phi at ell + 1 (needed for dphi)
    phi[ell] = phi1
    phi[ell - 1] = ((2. * ell + 1.) * cotK * phi[ell] - phi_p1) /csqrtl(nu2  + ell*ell)

    # Using backwards recurrence to evalute phi up to a multiplicative constant
    for i in range(ell-2, -1, -1):
        denom = csqrtl(nu2 + (i + 1.) * (i + 1.))
        val = csqrtl(nu2 + (2. + i) * (2. + i))
        phi[i] = ((2. * (i + 2.) -1.) * cotK * phi[i+1] - val * phi[i+2]) / denom
        # Renormalize everything in case of overflow
        if(cabsl(phi[i])> 1e200):
            for j in prange(i, ell+1, nogil = True):
                phi[j] *= (1e-200)

    cdef long double complex factor = phi0 / phi[0]  # Multiplicative factor
    for i in prange(ell + 2, nogil = True):
        phi[i] *= factor
    return phi

cdef long double complex[:] ForwardsRecurrenceComplex(long double x,
                                                      long double complex nu,
                                                      int ell,
                                                      Geometric_variables gvar):
    """Calculates the HBF using forward recurrence method for complex order
    
    For details, see Numerical Recipes in C chapter 6,
    Tram (2017): Computation of hyperspherical Bessel functions,
    Class hyperspherical.c file.

    Parameters
    ----------
    x : double
        Argument
    nu : complex
        Degree
    ell : int
        Order
    gvar : struct
        geometrical variables

    Returns
    -------
    typed memoryview
    """

    cdef:
        long double complex[:] phi = np.empty(ell + 2, dtype = np.clongdouble)
        long double cotK = gvar.dr_over_r 
        long double sinK = gvar.r 
        long double complex nu2 = nu * nu
        int l

    phi[0] = csinl(nu * x) / (nu * sinK)
    phi[1] = phi[0] * (cotK - nu / ctanl(nu * x)) / csqrtl(nu2 + 1.)
    for l in range(2, ell + 2):
        phi[l] = ((2. * l - 1.) * cotK * phi[l-1] - phi[l-2] * csqrtl(nu2 + (l - 1.) * (l - 1.)))\
            /csqrtl(nu2 + l*l)
        
    return phi 

@cython.cfunc
cdef void ContinuedFractionComplex(long double complex nu, 
                                   int ell,
                                   long double cotK, 
                                   long double complex *ratio,
                                   int *sign_r,
                                   int *sign_i, 
                                   int *nonconvergence):
    """Lentz algorithm for the calculation of continued fraction
    
    See Numerical Recipes in C page 169

    Parameters
    ----------
    nu : complex
        Degree
    ell : int
        Order
    cotK : double
        Numerical factor
    ratio : pointer
        Pointer to the variable that stores 
        the estimative of the continued fraction
    sign_r : pointer
        Pointer to the variable that stores
        the sign of the real part phi[ell]
    sign_i : pointer
        Pointer to the variable that stores
        the sign of the imaginary part phi[ell]
    nonconvergence : int
        Whether the continued fraction has converged or not
    """
    
    nonconvergence[0] = 0
    sign_r[0] = 1
    sign_i[0] = 1
    cdef:
        long double tiny = 1e-100 
        long double complex f =  ell * cotK
        long double complex C = f
        long double complex D = 0. 
        double i = 1.
        long double complex nu2 = nu*nu
        long double complex denom
        long double complex a
        long double complex b
        long double complex Delta
        long double epsilon_mac = np.finfo(np.clongdouble).eps
        long double eps = 1.
        double ell_a = <double> ell

    while (eps > epsilon_mac):
        denom = csqrtl(nu2 + (ell_a + i + 1.) * (ell_a + i + 1.))
        b = (2. *(ell_a + i) + 1.) * cotK / denom
        a = -csqrtl(nu2 + (ell_a + i)*(ell_a + i)) /csqrtl(nu2 + (ell_a + i + 1.)*(ell_a + i + 1.))
        if(i == 1):
            a = csqrtl(nu2 + (ell_a + 1.)* (ell_a + 1.)) * a
        D = b + a * D
        C = b + a / C
        if(cabsl(D) == 0):
            D = tiny
        if (cabsl(C) == 0):
            C = tiny
        D = 1.0 / D
        Delta = C * D
        f = f * Delta
        if(creall(D) < 0):
            sign_r[0] *= -1
        if(cimagl(D) < 0):
            sign_i[0] *= -1
        eps = cabsl(Delta - 1)
        i+= 1.
        if(i == 1000000):
            print("Continued fraction did not converge")
            nonconvergence[0] = 1
            break
    ratio[0] = f

@cython.cfunc 
cdef long double complex ximoverkc(int m, long double complex nu):
    """Eq. 3.14 in arxiv.org/abs/1909.13687

    Parameters
    ----------
    m : int
        Perturbation type
    nu : complex
        Fourier mode
    
    Returns
    -------
    complex
    """
    cdef:
        int i 
        long double complex prod = 1
        long double complex nu2 = nu * nu
    for i in prange(m, nogil = True):
        prod *= 1 / csqrtl(nu2 + (i+1) * (i+1))
    return prod

cdef long double complex[:] EpsilonComplex(long double chi,
                                           int s, int j, int m, 
                                           int ell_max, 
                                           long double complex nu, 
                                           long double ellc,
                                           int calK, 
                                           str mode):
    """Even (electric) part of radial function 

    See eq. (2.40) in Phys. Rev. D 100, 123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi : double
        Comoving distance
    s : int
        Spin
    j : int
        Multipole
    m : int
        Magnetic number
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    calK : int
        Curvature
    mode : {'tensor', 'vector'}

    Returns
    -------
    numpy_array
    """
    
    cdef:
        long double r = ellc * sinhl(chi/ellc) 
        long double x = coshl(chi/ellc) 
        long double y = r / ellc 
        long double complex nured = nu * ellc
        int l
        long double complex xi_k = ximoverkc(j,nured)
        long double complex[:] epsilon = np.empty(ell_max+1,dtype = np.clongdouble)
        long double complex[:] dphi
        long double complex[:] d2phi
        Geometric_variables gvar
    
    gvar.r = y
    gvar.dr_over_r = x / y
    cdef long double complex[:] phi = HyperBesselComplex(chi/ellc, nured ,ell_max,  gvar)
    if(mode == 'tensor'):
        ell0 = 2
    elif(mode == 'vector'):
        ell0 = 1
    else:
        raise ValueError('Nonexisting or non implemented mode')
    if m == 0:
        if s == 0:
            if j == 0:
                return phi
                # for l in range(ell0, ell_max+1):
                #     epsilon[l] = phi[l]
            elif j == 1:
                dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
                return xi_k*np.asarray(dphi)
                # for l in range(ell0, ell_max+1):
                #     epsilon[l] = xi_k*dphi[l]
            elif j == 2:
                dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
                d2phi = HyperBesselPrime2Complex(nured,ell_max,phi, dphi, gvar)
                return xi_k/2. *(3.*np.asarray(d2phi) +(nured*nured + 1.)*np.asarray(phi[0:ell_max+1]))
                # for l in range(ell0, ell_max+1):
                #     epsilon[l] = xi_k/2. *(3.*HyperBesselPrime2Complex(nured,l,phi, dphi, gvar)\
                #           +(nured*nured + 1.)*phi[l])
        elif s == 1:
            if j == 1:
                for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt(l*(l+1.)/2.)\
                                 *phi[l]/y
            elif j == 2:
                dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
                for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt(3.*l*(l+1.)/2.)\
                                 *(dphi[l]/y-phi[l]*x/(y*y))
        elif s == 2 and j == 2:
            for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt((3./8.)*(l+2.)*(l+1.)*(l)*(l-1.))\
                                 *phi[l]/(y*y)

    elif m == 1:
        if s == 0:
            if j == 1:
                for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt(l*(l+1.)/2.)\
                                 *phi[l]/y
            elif j == 2:
                dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
                for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt(3.*l*(l+1.)/2.)\
                                 *(dphi[l]/y\
                                 -phi[l]*x/y**2) 
        elif s == 1:
            dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
            if j == 1:
                return (xi_k/2.) *(np.asarray(dphi) + np.asarray(phi[0:ell_max+1])*x/y)
                # for l in range(ell0, ell_max+1):
                #     epsilon[l] = (xi_k/2.)\
                #                  *(dphi[l]+phi[l]*x/y)
            if j == 2:
                d2phi = HyperBesselPrime2Complex(nured,ell_max,phi, dphi, gvar)
                return xi_k *(np.asarray(d2phi) +(x/y)*np.asarray(dphi) +(nured*nured/2.-1./(y*y))*np.asarray(phi[0:ell_max+1]))
                # for l in range(ell0, ell_max+1):
                #     epsilon[l] = xi_k\
                #                  *(HyperBesselPrime2Complex(nured,l,phi, dphi, gvar)
                #                  +(x/y)*dphi[l]
                #                  +(nured*nured/2.-1./y**2)*phi[l])
        elif s == 2 and j == 2:
            dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
            for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt((l+2)*(l-1))/2\
                                 *(dphi[l]/y +phi[l]*x/(y*y))

    elif m == 2 and j == 2:
        if s == 0:
            for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt((3./8.)*(l+2.)*(l+1.)*(l)*(l-1.))\
                                 *phi[l]/y**2
        elif s == 1:
            dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
            for l in range(ell0, ell_max+1):
                    epsilon[l] = xi_k\
                                 *sqrt((l+2.)*(l-1.))/2.\
                                 *(dphi[l]/y
                                 +phi[l]*x/y**2)
        elif s == 2:
            dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
            d2phi = HyperBesselPrime2Complex(nured,ell_max,phi, dphi, gvar)
            return (xi_k/4) *(np.asarray(d2phi) +4.*(x/y)*np.asarray(dphi) + (2.*((x/y)**2) +1. - nured*nured)\
                              *np.asarray(phi[0:ell_max+1]))
            # for l in range(ell0, ell_max+1):
            #         epsilon[l] = (xi_k/4)\
            #                      *(HyperBesselPrime2Complex(nured,l,phi, dphi, gvar)
            #                      +4.*(x/y)*dphi[l]
            #                      +(2.*((x/y)**2) - calK - nured**2)
            #                      *phi[l])
    return epsilon

cdef long double complex[:] BetaComplex(long double chi, 
                                        int s, int j, int m, 
                                        int ell_max, 
                                        long double complex nu, 
                                        long double ellc, 
                                        int calK, 
                                        str mode):
    """Odd (magnetic) part of radial function 

    See eq. (2.40) in Phys. Rev. D 100, 123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi : double
        Comoving distance
    s : int
        Spin
    j : int
        Multipole
    m : int
        Magnetic number
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    calK : int
        Curvature
    mode : {'tensor', 'vector'}

    Returns
    -------
    numpy_array
    """

    cdef:
        long double r = ellc * sinhl(chi/ellc) 
        long double x = coshl(chi/ellc) 
        long double y = r/ellc 
        int l
        long double complex nured = nu * ellc
        long double complex[:] dphi
        long double complex[:] beta = np.empty(ell_max+1,dtype = np.clongdouble)
        Geometric_variables gvar
        long double complex xi_k = ximoverkc(j,nured)
    
    gvar.r = y
    gvar.dr_over_r = x/y
    cdef long double complex [:] phi = HyperBesselComplex(chi/ellc, nured ,ell_max, gvar)
    if(mode == 'tensor'):
        ell0 = 2
    elif(mode == 'vector'):
        ell0 = 1
    else:
        raise ValueError('Nonexisting or non implemented mode')

    if m == 0 or s == 0:
        return np.zeros(ell_max+1)
    elif m == 1:
        if s == 1:
            if j == 1:
                return -nured/2*xi_k * np.asarray(phi[0:ell_max+1])
                # for l in range (ell0, ell_max+1):
                #     beta[l] = -nured/2*xi_k\
                #               *phi[l]
            elif j == 2:
                dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
                return -nured/2*xi_k\
                              *(np.asarray(dphi) - (x/y)*np.asarray(phi[0:ell_max+1]))
                # for l in range (ell0, ell_max+1):
                #     beta[l] = -nured/2*xi_k\
                #               *(dphi[l] - (x/y)*phi[l])
        elif s == 2 and j == 2:
            for l in range (ell0, ell_max+1):
                    beta[l] = -nured*xi_k\
                              *sqrt((l+2.)*(l-1.))/2.\
                              *phi[l]/y
    elif m == 2:
        if s == 1 and j == 2:
            for l in range (ell0, ell_max+1):
                    beta[l] = -nured*xi_k\
                              *sqrt((l+2.)*(l-1.))/2.\
                              *phi[l]/y
        elif s == 2 and j == 2:
            dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
            return -nured/2*xi_k\
                              *(np.asarray(dphi) +2*(x/y)*np.asarray(phi[0:ell_max+1]))
            # for l in range (ell0, ell_max+1):
            #         beta[l] = -nured/2*xi_k\
            #                   *(dphi[l] +2*(x/y)*phi[l])
                    
    return beta


def epsilon(double chi,
            int s, int j, int m,
            int ell_max,
            double complex nu,
            int calK, double ellc ,
            str mode):
    """Even (electric) part of radial function 

    See eq. (2.40) in Phys. Rev. D 100, 123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi : double
        Comoving distance
    s : int
        Spin
    j : int
        Multipole
    m : int
        Magnetic number
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    calK : int
        Curvature
    mode : {'tensor', 'vector'}

    Returns
    -------
    numpy_array
    """

    return EpsilonComplex(chi, s, j, m, ell_max, nu, ellc, calK, mode)

def beta(double chi,
         int s, int j, int m,
         int ell_max,
         double complex nu,
         int calK,
         double ellc,
         str mode):
    """Odd (magnetic) part of radial function 

    See eq. (2.40) in Phys. Rev. D 100, 123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi : double
        Comoving distance
    s : int
        Spin
    j : int
        Multipole
    m : int
        Magnetic number
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    calK : int
        Curvature
    mode : {'tensor', 'vector'}

    Returns
    -------
    numpy_array
    """

    return BetaComplex(chi, s, j, m, ell_max, nu, ellc, calK, mode)

def epsbeta_for_tensor(double [:] chi_grid,
                       int ell_max,
                       double complex nu,
                       double ellc,
                       int grid_length ):
    """Even and odd part of radial function 

    See eq. (2.40) in Phys. Rev. D 100, 123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi_grind : array
        Array of distances
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius
    grind_length : int
        lenght of chi_grind

    Returns
    -------
    list
        List containing three arrays, the radial functions
        for temperature and polarizations multipoles
    """
    cdef:
        long double complex[:,:] epsilonT = np.empty((grid_length, ell_max-1), dtype = np.clongdouble)
        long double complex[:,:] epsilonE = np.empty((grid_length, ell_max-1), dtype = np.clongdouble)
        long double complex[:,:] betaB = np.empty((grid_length, ell_max-1), dtype = np.clongdouble)
        long double complex[:] phi
        long double complex[:] dphi
        long double complex[:] d2phi
        double[:] coef = np.empty(ell_max - 1)
        Geometric_variables gvar
        long double complex nured = nu * ellc
        long double chi 
        double complex xi_k = ximoverkc(2,nured)
        double complex nured2 = nured * nured
        long double dr_over_r2 
        long double r2 
        Py_ssize_t i, j
        int l

    coef[0] = 3.
    for l in range(3, ell_max+1):
        coef[l-2] = sqrt((l+2.) / (l-2.)) * coef[l - 3]

    for i in range (grid_length):
        chi = chi_grid[i]
        gvar.r = sinhl(chi/ellc)
        gvar.dr_over_r = coshl(chi/ellc) / gvar.r
        dr_over_r2 = gvar.dr_over_r * gvar.dr_over_r
        r2 = gvar.r * gvar.r
        phi =  HyperBesselComplex(chi/ellc, nured ,ell_max,  gvar)
        dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
        d2phi = HyperBesselPrime2Complex(nured,ell_max,phi, dphi, gvar)
        for j in range(2, ell_max+1):
            epsilonT[i][j-2] = xi_k* coef[j-2] *phi[j]/r2
            epsilonE[i][j-2] = (xi_k/4.) *(d2phi[j] +4.* gvar.dr_over_r * dphi[j]\
                             + (2. * dr_over_r2 + 1. - nured2)\
                              *phi[j])
            betaB[i][j-2] = -nured/2. * xi_k *(dphi[j] + 2. * gvar.dr_over_r * phi[j])
    
    return [epsilonT, epsilonE, betaB]

def epsbeta_for_vector(double [:] chi_grid, int ell_max, double complex nu, double ellc, int grid_length):
    """
    Even and odd part of radial function 

    See eq. (2.40) in Phys. Rev. D 100,  123535 (2019)
    [arXiv:1909.13687 [gr-qc]]

    Parameters
    ----------
    chi_grind : array
        Array of distances
    ell_max : int
        Maximum multipole
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius
    grind_length : int
        lenght of chi_grind

    Returns
    -------
    list
        List containing four arrays, the radial functions
        for temperature (two arrays) and polarizations multipoles
    """
    cdef:
        long double complex[:,:] epsilonT = np.empty((grid_length, ell_max), dtype = np.clongdouble)
        long double complex[:,:] epsilonT2 = np.empty((grid_length, ell_max), dtype = np.clongdouble)
        long double complex[:,:] epsilonE = np.empty((grid_length, ell_max), dtype = np.clongdouble)
        long double complex[:,:] betaB = np.empty((grid_length, ell_max), dtype = np.clongdouble)
        long double complex[:] phi
        long double complex[:] dphi
        double[:] coef = np.empty(ell_max)
        double[:] coef2 = np.empty(ell_max)
        double[:] coef3 = np.empty(ell_max)
        Geometric_variables gvar
        long double complex nured = nu * ellc
        long double chi 
        double complex xi_k2 = ximoverkc(2,nured)
        double complex xi_k1 = ximoverkc(1,nured)
        double complex nured2 = nured * nured
        Py_ssize_t i, j
        int l

    coef[0] = 1.
    coef2[0] = sqrt(3.)
    coef3[0] = 0.
    for l in range (2, ell_max + 1):
        coef[l - 1] = sqrt( (l + 1.) / (l - 1.) ) * coef[l - 2]
        coef2[l - 1] = sqrt(3.) * coef[l - 1]
        coef3[l - 1] = sqrt( (l + 2.) * (l - 1.) )  
    
    for i in range (grid_length):
        chi = chi_grid[i]
        gvar.r = sinhl(chi/ellc)
        gvar.dr_over_r = coshl(chi/ellc) / gvar.r
        phi =  HyperBesselComplex(chi/ellc, nured ,ell_max,  gvar)
        dphi = HyperBesselPrimeComplex(phi, nured, ell_max, gvar)
        for j in range (1, ell_max + 1):
            epsilonT[i][j - 1] = xi_k1 * coef[j-1] * phi[j] / gvar.r
            epsilonT2[i][j - 1] = xi_k2 * coef2[j-1] * (dphi[j] / gvar.r - phi[j] * gvar.dr_over_r / gvar.r )
            epsilonE[i][j - 1] = (xi_k2 / 2.) * coef3[j -1] * (dphi[j] / gvar.r + phi[j] * gvar.dr_over_r / gvar.r )
            betaB[i][j - 1] = - nured * (xi_k2 / 2.) * coef3[j -1] * phi[j] / gvar.r
        
    return [epsilonT, epsilonT2, epsilonE, betaB]

def HyperBesselTestComplex(double[:] chi_grid, long double complex nu, int ell_max, int calK = -1, double ellc = 1.):
    cdef int lenght = len(np.array(chi_grid)) * (ell_max+2)
    cdef int m 
    cdef int i
    cdef int j
    cdef long double chi
    cdef long double complex[:] auxphi = np.empty(ell_max+2, dtype = np.clongdouble)
    #cdef double[:] auxdphi = np.empty(ell_max+1)
    cdef long double complex[:] phi = np.empty(lenght, dtype = np.clongdouble)
    #cdef double[:] dphi = np.empty(n)
    cdef Geometric_variables gvar
    cdef long double complex nured = nu *ellc

    for i, chi in enumerate(np.array(chi_grid)):
        r = ellc * sinhl(chi/ellc)
        x = coshl(chi/ellc)
        
        y = r/ellc # dimensionless radial coordinate
        gvar.r = y
        gvar.dr_over_r = x/y
        auxphi = HyperBesselComplex(chi/ellc, nured ,ell_max, gvar)
        HyperBesselPrimeComplex(auxphi, nured, ell_max, gvar)
        for j in range (ell_max+2):
            m = i * (ell_max+2) + j
            phi[m] = auxphi[j]
    phii = np.reshape(np.array(phi, dtype = np.clongdouble), (len(chi_grid), ell_max+2))
    return phii
