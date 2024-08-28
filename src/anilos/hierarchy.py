"""
Module that contain the methods related to the Boltzmann Hierarchy

kappa_array : Computes all the free streaming coefficients
    in a given multipole interval

zeta_factor_array : Computes the "non-plane wave coefficient"
    in a given interval. The definition of zeta_factor is given
    in eq. (5.10) in Phys. Rev. D 100,
    123534 (2019) [arXiv:1909.13688 [gr-qc]]

zetas_array : Compute zeta in a given multipole interval
    see the definition of zeta_factor_array

tensor_hierarchy : Implements the Boltzmann equation using numpy 
    1d arrays and numba

tensor_tight_coupling_hierarchy : Implements the Boltzmann equation
    in tight coupling regime using numpy  1d arrays and numba

vector_hierarchy : Implements the Boltzmann equation using numpy 
    1d arrays and numba

vector_tight_coupling_hierarchy : Implements the Boltzmann equation in tight
    coupling regime using numpy  1d arrays and numba
"""

import numpy as np
from numba import njit, c16, f8, i8
from numba.types import unicode_type

@njit( c16[:](i8, i8, i8, i8, c16, f8) )
def kappa_array(ell0, ell_max, s, m, nu, ellc):
    """Creates an 1d array contaning the values of kappa in [ell0, ell_max]
    
    The definition of kappa is given in eq. (3.19) in Phys. Rev. D 100,
    123535 (2019) [arXiv:1909.13687 [gr-qc]]

    Parameters
    -----------
    ell0 : int
        First multipole
    ell_max : int
        Last multipole
    s : int
        Spin eigenvalue
    m : int
        Perturbation type
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius

    Returns
    -------
    complex 1d array 
    """

    nu2 = nu * nu
    ellc2 = ellc * ellc
    s2 = s * s
    m2 = m * m
    kappa_array = np.empty(ell_max + 1 - ell0, dtype = 'complex')
    for l in range(ell0, ell_max + 1):
        kappa_array[l - ell0] =  np.sqrt( ( l*l -m2 ) *
                                 ( l*l-s2 ) / (l*l) )*np.sqrt(nu2 + (1./ellc2) *l * l)

    return kappa_array

@njit( c16[:]( i8, i8, f8, i8) )
def zeta_factor_array(ell0, ell_max, sqrth, m):
    """Creates an array containing the values of zeta_factor in[ell0, ell_max]

    The definition of zeta_factor is given in eq. (5.10) in Phys. Rev. D 100,
    123534 (2019) [arXiv:1909.13688 [gr-qc]]

    Parameters
    ----------
    ell0 : int
        First multipole
    ell_max : int
        Last multipole
    sqrth : float
        Spiraling length / curvature radius
    m : int
        Perturbation type

    Returns
    -------
    Complex 1d array
    """

    zeta_factor_array = np.empty(ell_max + 1 - ell0, dtype = 'complex')
    for l in range(ell0, ell_max + 1):
       zeta_factor_array[l- ell0] = (-1j)*np.sqrt((l - 1. + m* 1j / sqrth) / ( l + 1. - m * 1j / sqrth ))

    return zeta_factor_array

@njit( c16[:]( i8, i8, f8, i8))
def zetas_array(ell0, ell_max, sqrth, m):
    """Creates an array containing the values of zeta in [ell0, ell_max]
    
    The definition of zeta is given in eq. (5.10) in Phys. Rev. D 100,
    123534 (2019) [arXiv:1909.13688 [gr-qc]]

    Parameters
    ----------
    ell0 : int
        First multipole
    ell_max : int
        Last multipole
    sqrth : float
        Spiraling length / curvature radius
    m : int
        Perturbation type

    Returns
    -------
    Complex 1d array
    """

    zeta_factor = zeta_factor_array(m+1, ell_max + 1, sqrth, m)
    zetas_array = np.empty(ell_max + 1 - ell0, dtype= 'complex')
    factor = (-1) ** m
    zetas_array[0] = factor * np.prod(zeta_factor[: ell0 - m])
    for l in range (ell0 + 1, ell_max+1):
        zetas_array[l - ell0] = zeta_factor[l - m - 1] * zetas_array[l -ell0 - 1]

    return zetas_array

@njit(c16[:](f8, c16[:], i8, c16, f8, c16, f8, f8, f8, f8, f8, f8, i8[:], c16[:], c16[:], f8[:], c16[:]))
def tensor_hierarchy(eta,
                     y,
                     ell_max,
                     nu,
                     ellc,
                     calS2,
                     calK,
                     calH,
                     scalefac,
                     rho_g,
                     ratio_neutrino_photons,
                     tauprime,
                     ur_index,
                     kappa0_array,
                     kappa2_array,
                     denom_array,
                     zeta_array):
    
    """ Boltzmann Hierarchy for tensor modes

    System of 4 * ell_max - 2 coupled equations of the variables
    (beta,beta',ur,T,E,B) where
    beta, beta': spacetime shear and its 1st derivative
    ur: neutrinos
    T,E,B: Temperature, E and B modes of photons
    It employs numpy arrays and is optimized by numba @njit decorator.

    Parameters
    ----------
    eta : float
        Conformal time
    y : 1d complex array of  4 * ell_max - 2 elements
        (beta,beta',ur,T,E,B)
    ell_max : int
        Multipole where truncation happens
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius
    calS2 : complex
        Eq. D.2 in arxiv:1909.13688
    calK : float
        Dimensionless curvature constant 
    calH : float
        Hubble parameter times scale factor at eta
    scalefac : float
        Scale factor at eta
    rho_g : float
        Photon density at eta
    ratio_neutrino_photons : float
    tauprime : float
     Optical depth derivative at eta
    ur_index : 1d int array of ell_max - 1 elements
        Array containing the indices of the neutrino's multipoles
    kappa0_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    kappa2_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    denom_array : 1d float array of ell_max elements
        Array containing values needed to compute the
        free streaming coefficients
    zeta_array : 1d complex array of ell_max - 2 elements
        Array containing values needed to compute the
        free streaming coefficients

    Returns
    -------
    complex array
        Array of 4 * ell_max - 2 entries containing the
        derivatives of each variable
    """

    total = 4 * (ell_max -1) + 2  # Number of equations in the hierarchy
    #Setting indices for the hierarchy
    index_beta = 0
    index_betaprime = 1
    index_ur2 = 2
    index_T2 = 3
    index_E2 = 4
    ur_max = ur_index[-1]
    T_max = ur_max + 1
    E_max = ur_max + 2
    B_max = ur_max + 3

    hierarchy_EB = np.zeros(total, dtype = 'complex')
    hierarchy_compton_out = np.zeros(total, dtype = 'complex')
    hierarchy_total = np.zeros(total, dtype= 'complex')

    #####
    ## Full Boltzmann hierarchy
    ## y[0] = beta, y[1] = beta', y[2] = ur_2, y[3] = T_2, y[4] = E_2, y[5] = B_2, y[6] = ur_3, ...
    #####

    #####
    ## 1 - Background shear couplings (non-stochastic, infinite wavelength gw).
    ## This implements all couplings between beta, beta', and the quadrupoles (ur_2,T_2,E_2,B_2)
    #####
    #Couples beta' with beta'
    hierarchy_beta_beta = y[index_betaprime]
    #Couples beta'' with beta, beta', ur_2 and T_2
    hierarchy_beta_betaprime = calS2 * y[index_beta] \
                            - 2 * calH * y[index_betaprime] \
                            + (8./5.) * scalefac * scalefac * rho_g * ratio_neutrino_photons * y[index_ur2] \
                            + (8./5.) * scalefac * scalefac * rho_g * y[index_T2] 
    #Couples ur_2' with beta'
    hierarchy_beta_ur =  - y[index_betaprime]
    #Couples T_2' with beta'
    hierarchy_beta_T =  - y[index_betaprime]

    hierarchy_total[index_beta] = hierarchy_beta_beta
    hierarchy_total[index_betaprime] = hierarchy_beta_betaprime
    hierarchy_total[index_ur2] = hierarchy_beta_ur
    hierarchy_total[index_T2] =  hierarchy_beta_T 

    #####
    ## 2 - Elements giving compton scattering in.
    ## This only couples T_2 and E_2
    #####
    hierarchy_compton_in_T2 = tauprime * (1./10.) * y[index_T2] + tauprime * (-np.sqrt(6.)/10.) * y[index_E2]
    hierarchy_compton_in_E2 = tauprime * (-np.sqrt(6.)/10.) * y[index_T2] + tauprime * (3./5.) * y[index_E2]

    hierarchy_total[index_T2] += hierarchy_compton_in_T2
    hierarchy_total[index_E2] = hierarchy_compton_in_E2

    #Bianchi IX depends only on the dipole of T and on the coupling of the polarisation
    if calK > 0:
        for i, ur_i in enumerate (ur_index[:-1]):
            T_i = ur_i + 1
            E_i = ur_i + 2
            B_i = ur_i + 3
            #####
            ## Elements coupling E and B modes (E-B mixing)
            #####
            val = 4. * nu / (i+2.) / (i+3.)
            hierarchy_EB[B_i] = val * y[E_i]
            hierarchy_EB[E_i] = -val * y[B_i]

            #####
            ## Elements giving Compton scattering out.
            ## Couples T->T, E->E and B->B
            #####
            hierarchy_compton_out[T_i: T_i + 3] = -tauprime * y[T_i: T_i + 3]
    
        hierarchy_total += hierarchy_EB + hierarchy_compton_out
        return hierarchy_total

    hierarchy_up = np.zeros(total, dtype = 'complex')
    hierarchy_down = np.zeros(total, dtype = 'complex')
    for i, ur_i in enumerate (ur_index[:-1]):
        T_i = ur_i + 1
        E_i = ur_i + 2
        B_i = ur_i + 3
        ur_ell_plus_1 = ur_i + 4
        T_ell_plus_1 = T_i + 4
        E_ell_plus_1 = E_i + 4
        B_ell_plus_1 = B_i + 4

        #####
        ## 3 - Free streaming terms in Boltzmann hierarchy
        ## coupling l to l+1 (upper right triangle of the matrix)
        ## and l to l-1 (lower left triangle of the matrix)
        #####

        #This gives 6.5a in Phys. Rev. D 100, 123535 (2019) [arXiv:1909.13687 [gr-qc]]
        #note that zeta^m_l/zeta^m_{l+1} = 1/zeta_factor
        coeff0 = - kappa0_array[i] * denom_array[i + 1] / zeta_array[i]
        coeff2 = - kappa2_array[i] * denom_array[i + 1] / zeta_array[i]
        hierarchy_up[ur_i] = coeff0 * y[ur_ell_plus_1]
        hierarchy_up[T_i] = coeff0 * y[T_ell_plus_1]
        hierarchy_up[E_i] = coeff2 * y[E_ell_plus_1]
        hierarchy_up[B_i] = coeff2 * y[B_ell_plus_1]

        # This gives 6.5b in Phys. Rev. D 100, 123535 (2019) [arXiv:1909.13687 [gr-qc]]
        # note that zeta^m_l/zeta^m_{l-1} = zeta_factor
        coeffd0 = kappa0_array[i] * zeta_array[i] * denom_array[i]
        coeffd2 = kappa2_array[i] * zeta_array[i] * denom_array[i]
        hierarchy_down[ur_ell_plus_1] = coeffd0 * y[ur_i]
        hierarchy_down[T_ell_plus_1] = coeffd0 * y[T_i]
        hierarchy_down[E_ell_plus_1] = coeffd2 * y[E_i]
        hierarchy_down[B_ell_plus_1] = coeffd2 * y[B_i]

        #####
        ## 4 - Elements coupling E and B modes (E-B mixing)
        #####
        val = 4. * nu / (i+2.) / (i+3.)
        hierarchy_EB[B_i] = val * y[E_i]
        hierarchy_EB[E_i] = -val * y[B_i]

        #####
        ## 5 - Elements giving Compton scattering out.
        ## Couples T->T, E->E and B->B
        #####
        hierarchy_compton_out[T_i: T_i + 3] = -tauprime * y[T_i: T_i + 3]

    #The loop above fill these entries, but they must be zero
    hierarchy_down[ur_max] = 0
    hierarchy_down[T_max] = 0
    hierarchy_down[E_max] = 0
    hierarchy_down[B_max] = 0
    
    #####
    ## 6 - Closure relation for Boltzmann hierarchy.
    ## This implements Eqs. (9) and (10) from Pitrou et al 2020 [arXiv:2005.12119]
    ## including a copy of Eq.(9) for neutrinos. @Cyril: please confirm that you agree.
    ## See also section 5.4.5 in Riazuelo's thesis:
    ## https://tel.archives-ouvertes.fr/tel-00003366
    #####

    auxval = -(ell_max + 3.) / ellc * 1. / np.tanh(eta/ellc) 
    auxval2 = kappa0_array[-1] / (ell_max - 2.) * (2. * ell_max + 1.) / (2. * ell_max - 1.) * zeta_array[-1]
    auxval3 = kappa2_array[-1] / (ell_max - 2.) * (2. * ell_max + 1.) / (2. * ell_max - 1.) * zeta_array[-1]
    hierarchy_trunc_ur = auxval * y[ur_max] + auxval2 * y[ur_max-4] 
    hierarchy_trunc_T = auxval * y[T_max] + auxval2 * y[T_max-4] - tauprime * y[T_max]
    hierarchy_trunc_E = auxval * y[E_max] + auxval3 * y[E_max-4] + (2 * nu/ ell_max) * y[B_max] - tauprime * y[E_max]
    hierarchy_trunc_B = auxval * y[B_max] + auxval3 * y[B_max-4] - (2 * nu/ ell_max) * y[E_max] - tauprime * y[B_max]
   
    # full hierarchy
    hierarchy_total += hierarchy_up + hierarchy_down + hierarchy_EB + hierarchy_compton_out
    hierarchy_total[-4] += hierarchy_trunc_ur
    hierarchy_total[-3] += hierarchy_trunc_T
    hierarchy_total[-2] += hierarchy_trunc_E
    hierarchy_total[-1] += hierarchy_trunc_B

    return hierarchy_total

@njit(c16[:](f8, c16[:], i8, f8, c16, f8, f8, f8, f8, f8, i8[:], c16[:], f8[:], c16[:]))
def tensor_tight_coupling_hierarchy(eta,
                                    y,
                                    ell_max,
                                    ellc,
                                    calS2,
                                    calK,
                                    calH,
                                    scalefac,
                                    rho_g,
                                    ratio_neutrino_photons,
                                    ur_index,
                                    kappa0_array,
                                    denom_array,
                                    zeta_array):
    """ Boltzmann Hierarchy for tensor modes in tight coupling regime

    System of 4 * ell_max - 2 coupled equations of the variables
    (beta,beta',ur,T,E,B) where
    beta, beta': spacetime shear and its 1st derivative
    ur: neutrinos
    T,E,B: Temperature, E and B modes of photons

    Returns: 1d complex array of length 4 * ell_max - 2 containing the derivatives of each variable

    Parameters
    ----------
    eta : float
        Conformal time
    y : 1d complex array of  4 * ell_max - 2 elements
        (beta,beta',ur,T,E,B)
    ell_max : int
        Multipole where truncation happens
    ellc : float
        Curvature radius
    calS2 : complex
        Eq. D.2 in arxiv:1909.13688
    calK : float
        Dimensionless curvature constant 
    calH : float
        Hubble parameter times scale factor at eta
    scalefac : float
        Scale factor at eta
    rho_g : float
        Photon density at eta
    ratio_neutrino_photons : float
    ur_index : 1d int array of ell_max - 1 elements
        Array containing the indices of the neutrino's multipoles
    kappa0_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    denom_array : 1d float array of ell_max elements
        Array containing values needed to compute the
        free streaming coefficients
    zeta_array : 1d complex array of ell_max - 2 elements
        Array containing values needed to compute the
        free streaming coefficients

    Returns
    -------
    complex array
        Array of 4 * ell_max - 2 entries containing the
        derivatives of each variable
    """

    total = 4 * (ell_max -1) + 2  # Number of equations in the hierarchy
    index_beta = 0
    index_betaprime = 1
    index_ur2 = 2
    hierarchy_total = np.zeros(total, dtype= 'complex')
    ur_max = ur_index[-1]

    #####
    ## Full Boltzmann hierarchy
    ## y[0] = beta, y[1] = beta', y[2] = ur_2, y[3] = T_2, y[4] = E_2, y[5] = B_2, y[6] = ur_3, ...
    #####

    #####
    ## Background shear couplings (non-stochastic, infinite wavelength gw).
    ## This implements all couplings between beta, beta', and the quadrupole ur_2
    #####
    #There is no coupling between beta' and T_2

    hierarchy_beta_beta = y[index_betaprime]
    hierarchy_beta_betaprime = calS2 * y[index_beta] - 2 * calH * y[index_betaprime] \
                                + (8./5.) * scalefac * scalefac * rho_g * ratio_neutrino_photons * y[index_ur2] 
    hierarchy_beta_ur =  - y[index_betaprime]

    hierarchy_total[index_beta] = hierarchy_beta_beta
    hierarchy_total[index_betaprime] = hierarchy_beta_betaprime
    hierarchy_total[index_ur2] = hierarchy_beta_ur
    if calK > 0:
        return hierarchy_total

    #####
    ## Free streaming terms in Boltzmann hierarchy
    ## coupling l to l+1 (upper right triangle of the matrix)
    ## and l to l-1 (lower left triangle of the matrix)
    #####

    hierarchy_up = np.zeros(total, dtype = 'complex')
    hierarchy_down = np.zeros(total, dtype = 'complex')
    for i, ur_i in enumerate (ur_index[:-1]):
        ur_ell_plus_1 = ur_i + 4
        hierarchy_up[ur_i] = - kappa0_array[i] * denom_array[i + 1] / zeta_array[i] * y[ur_ell_plus_1]
        hierarchy_down[ur_ell_plus_1] = kappa0_array[i] * zeta_array[i] * denom_array[i] * y[ur_i]

    hierarchy_down[-4] = 0

    #####
    ## Closure relation for Boltzmann hierarchy.
    ## This implements Eqs. (9) and (10) from Pitrou et al 2020 [arXiv:2005.12119]
    ## including a copy of Eq.(9) for neutrinos. @Cyril: please confirm that you agree.
    ## See also section 5.4.5 in Riazuelo's thesis:
    ## https://tel.archives-ouvertes.fr/tel-00003366
    #####
    
    hierarchy_trunc_ur = -(ell_max + 3) / ellc * 1. / np.tanh(eta/ellc) * y[ur_max] + \
                             kappa0_array[-1] / (ell_max - 2) * (2 * ell_max + 1) \
                            / (2 * ell_max - 1) * zeta_array[-1] * y[ur_max-4] 

    hierarchy_total += hierarchy_up + hierarchy_down
    hierarchy_total[-4] += hierarchy_trunc_ur
    return hierarchy_total


@njit(c16[:](f8, c16[:], i8, c16, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8[:], c16[:], c16[:], f8[:], c16[:], unicode_type))
def vector_hierarchy(eta,
                    y,
                    ell_max,
                    nu,
                    ellc,
                    curvature,
                    calH,
                    scalefac,
                    rho_g,
                    tauprime,
                    Rbg,
                    Omega_nu,
                    Omega_g,
                    ur_index,
                    kappa0_array,
                    kappa2_array,
                    denom_array,
                    zeta_array,
                    gauge):
   
    """ Boltzmann Hierarchy for vector modes made to be solved by solve_ivp

    System of 4 * ell_max + 2 coupled equations of the variables
    (phi, vb, ur, T, E, B) where
    phi: metric perturbation 
    vb: baryon velocity (in synchronous gauge)
    ur: neutrinos
    T,E,B: Temperature, E and B modes of photons
    It employs numpy arrays and is optimized by numba @njit decorator.

    Parameters
    ----------
    eta : float
        Conformal time
    y : 1d complex array of  4 * ell_max - 2 elements
        (beta,beta',ur,T,E,B)
    ell_max : int
        Multipole where truncation happens
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius
    curvature : float
        Curvature constant
    calH : float
        Hubble parameter times scale factor at eta
    scalefac : float
        Scale factor at eta
    rho_g : float
        Photon density at eta
    tauprime : float
        Optical depth at eta
    Rbg : float
        3/4 of baryons/photons densities
    Omega_nu : float
        Neutrino density parameter
    Omega_g : float
        Photon density parameter
    ur_index : 1d int array of ell_max - 1 elements
        Array containing the indices of the neutrino's multipoles
    kappa0_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    kappa2_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    denom_array : 1d float array of ell_max elements
        Array containing values needed to compute the
        free streaming coefficients
    zeta_array : 1d complex array of ell_max - 2 elements
        Array containing values needed to compute the
        free streaming coefficients
    gauge : str
        gauge choice

    Returns
    -------
    complex array
        Array of 4 * ell_max - 2 entries containing the
        derivatives of each variable
    """

    num_var = 4 * ell_max + 2  # Total number of variables
    hierarchy_up = np.zeros(num_var, dtype = 'complex')
    hierarchy_down = np.zeros(num_var, dtype = 'complex')
    hierarchy_EB = np.zeros(num_var, dtype = 'complex')
    hierarchy_compton_out = np.zeros(num_var, dtype = 'complex')
    ur_max = ur_index[-1]
    T_max = ur_max + 1
    E_max = ur_max + 2
    B_max = ur_max + 3

    # Convenient shortcuts
    ratio_neutrino_photons = Omega_nu / Omega_g
    prefactor_Phi_source = (8./5.)*np.sqrt(3.)*scalefac**2*rho_g/np.sqrt(nu**2 - 4*curvature)

    #####
    ## Full Boltzmann hierarchy
    ## y[0] = phi, y[1] = vb, y[2] = ur_1, y[3] = T_1, y[4] = E_1, y[5] = B_1, y[6] = ur_2, ...
    #####

    ####
    ## 1 - coupling between phi, vb, ur_1, ur_2, T_1 and T_2
    ####
    
    # Note that what is being called vb is actually vb + Phi, which is gauge invariant
    hierarchy_beta_phi = (-2 * calH * y[0] 
                          + prefactor_Phi_source * ratio_neutrino_photons * y[6] 
                          + prefactor_Phi_source * y[7]
                          )
    if gauge == 'synchronous':
        hierarchy_beta_vb = (-calH -tauprime / Rbg) * y[1] + tauprime/Rbg * y[3]
        hierarchy_beta_vur = 0.
        hierarchy_beta_vg = tauprime * y[1]
        hierarchy_beta_ur_2 = - kappa0_array[0] * zeta_array[0] / 3. * y[0]
        hierarchy_beta_T_2 = - kappa0_array[0] * zeta_array[0] / 3. * y[0]
    if gauge == 'newtonian':
        hierarchy_beta_vb = (calH * y[0] 
                             + (-calH -tauprime / Rbg)* y[1] 
                             + tauprime / Rbg * y[3] 
                             - prefactor_Phi_source*ratio_neutrino_photons * y[6]
                             - prefactor_Phi_source * y[7]
                             )
        hierarchy_beta_vur = (2 * calH * y[0]
                              - prefactor_Phi_source * ratio_neutrino_photons * y[6]
                              -prefactor_Phi_source * y[7]
                              )
        hierarchy_beta_vg = (2 * calH * y[0] 
                             + tauprime * y[1] 
                             - prefactor_Phi_source*ratio_neutrino_photons * y[6] 
                             - prefactor_Phi_source * y[7]
                             )
        hierarchy_beta_ur_2 = 0.
        hierarchy_beta_T_2 = 0.
        
    ####
    ## 2 - Compton in: couples T2 and E2
    ####
    hierarchy_compton_in_T2 = tauprime * (1 / 10) * y[7] + tauprime * (-np.sqrt(6) / 10) * y[8]
    hierarchy_compton_in_E2 = tauprime * (-np.sqrt(6) / 10) * y[7] + tauprime * (3./5.) * y[8]

    for i, ur_i in enumerate (ur_index[:-1]):
        T_i = ur_i + 1
        E_i = ur_i + 2
        B_i = ur_i + 3
        ur_ell_plus_1 = ur_i + 4
        T_ell_plus_1 = T_i + 4
        E_ell_plus_1 = E_i + 4
        B_ell_plus_1 = B_i + 4

        #####
        ## 3 - Free streaming terms in Boltzmann hierarchy
        ## coupling l to l+1 (upper right triangle of the matrix)
        ## and l to l-1 (lower left triangle of the matrix)
        #####

        coeff0 = - kappa0_array[i] * denom_array[i + 1] / zeta_array[i]
        coeff2 = - kappa2_array[i] * denom_array[i + 1] / zeta_array[i]
        hierarchy_up[ur_i] = coeff0 * y[ur_ell_plus_1]
        hierarchy_up[T_i] = coeff0 * y[T_ell_plus_1]
        hierarchy_up[E_i] = coeff2 * y[E_ell_plus_1]
        hierarchy_up[B_i] = coeff2 * y[B_ell_plus_1]

        coeffd0 = kappa0_array[i] * zeta_array[i] * denom_array[i]
        coeffd2 = kappa2_array[i] * zeta_array[i] * denom_array[i]
        hierarchy_down[ur_ell_plus_1] = coeffd0 * y[ur_i]
        hierarchy_down[T_ell_plus_1] = coeffd0 * y[T_i]
        hierarchy_down[E_ell_plus_1] = coeffd2 * y[E_i]
        hierarchy_down[B_ell_plus_1] = coeffd2 * y[B_i]

        #####
        ## 4 - Elements coupling E and B modes (E-B mixing)
        #####

        val = 2. * nu / (i+1.) / (i+2.)
        hierarchy_EB[B_i] = val * y[E_i]
        hierarchy_EB[E_i] = -val * y[B_i]

        #####
        ## 5 - Elements giving Compton scattering out.
        ## Couples T->T, E->E and B->B
        #####
        hierarchy_compton_out[T_i: T_i + 3] = -tauprime * y[T_i: T_i + 3]

    #The loop above fill these entries, but they must be zero
    hierarchy_down[ur_max] = 0
    hierarchy_down[T_max] = 0
    hierarchy_down[E_max] = 0
    hierarchy_down[B_max] = 0

    #####
    ## 6- Closure relation for Boltzmann hierarchy.
    #####

    auxval = -(ell_max + 3.) / ellc * 1. / np.tanh(eta/ellc)
    auxval2 = kappa0_array[-1] / (ell_max - 2.) * (2. * ell_max + 1.) / (2. * ell_max - 1.) * zeta_array[-1]
    auxval3 = kappa2_array[-1] / (ell_max - 2.) * (2. * ell_max + 1.) / (2. * ell_max - 1.) * zeta_array[-1]
    hierarchy_trunc_ur = auxval * y[ur_max] + auxval2 * y[ur_max-4] 
    hierarchy_trunc_T = auxval * y[T_max] + auxval2 * y[T_max-4] - tauprime * y[T_max]
    hierarchy_trunc_E = auxval * y[E_max] + auxval3 * y[E_max-4] + (nu/ ell_max) * y[B_max] - tauprime * y[E_max]
    hierarchy_trunc_B = auxval * y[B_max] + auxval3 * y[B_max-4] - (nu/ ell_max) * y[E_max] - tauprime * y[B_max]
    
   # Full hierarchy
    hierarchy_total = hierarchy_up + hierarchy_down + hierarchy_EB + hierarchy_compton_out

    hierarchy_total[0] = hierarchy_beta_phi
    hierarchy_total[1] = hierarchy_beta_vb
    hierarchy_total[2] += hierarchy_beta_vur
    hierarchy_total[3] += hierarchy_beta_vg
    hierarchy_total[6] += hierarchy_beta_ur_2
    hierarchy_total[7] =  hierarchy_total[7] + hierarchy_beta_T_2 + hierarchy_compton_in_T2
    hierarchy_total[8] += hierarchy_compton_in_E2
    hierarchy_total[-4] += hierarchy_trunc_ur
    hierarchy_total[-3] += hierarchy_trunc_T
    hierarchy_total[-2] += hierarchy_trunc_E
    hierarchy_total[-1] += hierarchy_trunc_B

    return hierarchy_total

@njit(c16[:](f8, c16[:], i8, c16, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8[:], c16[:], f8[:], c16[:], unicode_type))
def vector_tight_coupling_hierarchy(eta,
                                    y,
                                    ell_max,
                                    nu,
                                    ellc,
                                    curvature,
                                    calH,
                                    scalefac,
                                    rho_g,
                                    tauprime,
                                    Rbg,
                                    Omega_nu,
                                    Omega_g,
                                    ur_index,
                                    kappa0_array,
                                    denom_array,
                                    zeta_array,
                                    gauge):
    """ Boltzmann Hierarchy for vector modes in tight coupling regime

    System of 4 * ell_max + 2 coupled equations of the variables
    (beta,beta',ur,T,E,B) where
    phi: metric perturbation 
    vb: baryon velocity (in synchronous gauge)
    ur: neutrinos
    T,E,B: Temperature, E and B modes of photons

    Parameters
    ----------
    eta : float
        Conformal time
    y : 1d complex array of  4 * ell_max - 2 elements
        (beta,beta',ur,T,E,B)
    ell_max : int
        Multipole where truncation happens
    nu : complex
        Fourier mode
    ellc : float
        Curvature radius
    curvature : float
        Curvature constant
    calH : float
        Hubble parameter times scale factor at eta
    scalefac : float
        Scale factor at eta
    rho_g : float
        Photon density at 
    tauprime : float
        Optical depth at eta
    Rbg : float
        3/4 of baryons/photons densities
    Omega_nu : float
        Neutrino density parameter
    Omega_g : float
        Photon density parameter
    ur_index : 1d int array of ell_max - 1 elements
        Array containing the indices of the neutrino's multipoles
    kappa0_array : 1d complex array of ell_max - 2 elements
        Array containing the kappas needed to compute the 
        free streaming coefficients
    denom_array : 1d float array of ell_max elements
        Array containing values needed to compute the
        free streaming coefficients
    zeta_array : 1d complex array of ell_max - 2 elements
        Array containing values needed to compute the
        freestreaming coefficients
    gauge : str
        Gauge choice

    Returns
    -------
    complex array
        Array of 4 * ell_max - 2 entries containing the
        derivatives of each variable
    """

    num_var = 4 * ell_max + 2  # Total number of entries
    ratio_neutrino_photons = Omega_nu/Omega_g 
    prefactor_Phi_source_constant = (8. / 5.) * np.sqrt(3.) / np.sqrt(nu**2 -  4 * curvature)
    hierarchy_up = np.zeros(num_var, dtype = 'complex')
    hierarchy_down = np.zeros(num_var, dtype = 'complex')
    ur_max = ur_index[-1]
    prefactor_Phi_source = prefactor_Phi_source_constant * scalefac**2 * rho_g
    alpha = kappa0_array[0] / 5.
    sourceTCfluid = alpha**2 / (1. + Rbg) * 20. / 9. / tauprime

    #####
    ## Full Boltzmann hierarchy
    ## y[0] = phi, y[1] = vb, y[2] = ur_1, y[3] = T_1, y[4] = E_1, y[5] = B_1, y[6] = ur_2, ...
    #####

    ####
    ## Coupling between phi, vb, ur_1, ur_2
    ####
        
    hierarchy_beta_phi = -2 * calH * y[0] + prefactor_Phi_source * ratio_neutrino_photons * y[6]
    if gauge == 'synchronous':
        hierarchy_beta_vb = (-calH * Rbg / (1. + Rbg) - sourceTCfluid) * y[1] + sourceTCfluid * y[0] 
        hierarchy_beta_vur = 0.
        hierarchy_beta_ur_2 = - kappa0_array[0] * zeta_array[0] / 3. * y[0]
    
    if gauge == 'newtonian':
        hierarchy_beta_vb = ( (-calH * Rbg / (1. + Rbg) + 2 * calH) * y[0] 
                             + (-calH * Rbg / (1. + Rbg) - sourceTCfluid) * y[1]
                            -prefactor_Phi_source * ratio_neutrino_photons * y[6]
                            )
        hierarchy_beta_vur = 2 * calH * y[0] - prefactor_Phi_source * ratio_neutrino_photons * y[6]
        hierarchy_beta_ur_2 = 0.
    
    #####
    ## Free streaming terms in Boltzmann hierarchy
    ## coupling l to l+1 (upper right triangle of the matrix)
    ## and l to l-1 (lower left triangle of the matrix)
    #####
    
    for i, ur_i in enumerate (ur_index[:-1]):
        ur_ell_plus_1 = ur_i + 4
        coeff0 = - kappa0_array[i] * denom_array[i + 1] / zeta_array[i]
        hierarchy_up[ur_i] = coeff0 * y[ur_ell_plus_1]

        coeffd0 = kappa0_array[i] * zeta_array[i] * denom_array[i]
        hierarchy_down[ur_ell_plus_1] = coeffd0 * y[ur_i]

    # The loop above fill these entries, but they must be zero
    hierarchy_down[-4] = 0

    #####
    ## Closure relation for Boltzmann hierarchy.
    #####

    hierarchy_trunc_ur = (-(ell_max + 3) / ellc * 1. / np.tanh(eta/ellc) * y[ur_max] 
                          + kappa0_array[-1] / (ell_max - 2) * (2 * ell_max + 1) / (2 * ell_max - 1) 
                          * zeta_array[-1] * y[ur_max-4] )

    hierarchy_total = hierarchy_up + hierarchy_down
    hierarchy_total[0] = hierarchy_beta_phi
    hierarchy_total[1] = hierarchy_beta_vb
    hierarchy_total[2] += hierarchy_beta_vur
    hierarchy_total[6] += hierarchy_beta_ur_2
    hierarchy_total[-4] += hierarchy_trunc_ur
    return hierarchy_total
