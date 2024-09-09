__authors__ = "Cyril Pitrou, João Vicente, Thiago Pereira"
__version__ = "0.0.1"
__maintainer__ = "João Vicente"
__email__ = "jgvicente2000@gmail.com"

from classy import Class
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.integrate import solve_ivp, trapezoid
import warnings
from .hybess import epsbeta_for_tensor, epsbeta_for_vector
from .hierarchy import *

class Anilos:  
    """AniLoS: Anisotropic Line-of-Sight.

    Line of sight CMB integration for nearly-isotropic Bianchi /
    Homogeneously-perturbed FLRW models.
    Written by Thiago Pereira, Cyril Pitrou, and João Vicente
    Based on: Phys. Rev. D 100, 123534 (2019)
              Phys. Rev. D 100, 123535 (2019)
    
    Methods
    -------
    tensor_solver
        Solves the tensor perturbation equations.
    vector_solver
        Solves the vector perturbation equations.
    alm_tensor
        Computes the tensor mode alm coefficients.
    alm_vector
        Computes the vector mode alm coefficients.
    
    Variables
    ---------
    ztable : array
        Table of redshift values.
    etatable : array
        Table of conformal time values.
    atable : array
        Table of scale factor values.
    tensor_tc : float
        Conformal time when tensor tight coupling ends.
    vector_tc : float
        Conformal time when vector tight coupling ends.
    nteb_tensor : array
        Array of the number of tensor modes for each multipole.
    nteb_vector : array
        Array of the number of vector modes for each multipole.
    almT_tensor : array
        Array of the tensor mode alm coefficients for temperature.
    almE_tensor : array
        Array of the tensor mode alm coefficients for E-mode polarization.
    almB_tensor : array
        Array of the tensor mode alm coefficients for B-mode polarization.
    almT_vector : array
        Array of the vector mode alm coefficients for temperature.
    almE_vector : array
        Array of the vector mode alm coefficients for E-mode polarization.
    almB_vector : array
        Array of the vector mode alm coefficients for B-mode polarization.
    """

    def __init__(self, params=None): 

        """
        Parameters
        ----------
        params : dict, optional
            Dictionary containing the input variables.
        h_hubble : float, default: 0.67810
            Dimensionless Hubble constant.
        Omega_K : float, default: 1e-6
            Curvature density today.
        Omega_b : float
            Baryon density parameter today.
        Omega_cdm : float
            CDM density parameter today.
        Omega_m : float
            Matter density parameter today.
        Omega_Lambda : float
            Cosmological constant density parameter today.
        z_reio : float, default: 7.6711
            Reionization redshift.
        cutoff_multipole : int, default: 30
            Cutoff multipole for Boltzmann hierarchy.
        sqrth : float, default: 0.1
            Spiraling length over curvature radius.
        epsilon_TC : float, default: 0.03
            Condition to define the tight coupling regime, i.e,
            for H(eta)/tau(eta) < epsilon_TC, tight coupling is assumed.
        IC : {'isocurvature', 'octupole'}
            Initial conditions for vector modes
        gauge : {'synchronous', 'newtonian'}
            Defines the gauge of the variables for vector modes.
        verbose : bool
            Whether the user wants a more detailed output.

        Notes
        -----
        Only Bianchi VII_h and Bianchi IX are implemented.
        Bianchi V and Bianchi VII_0 are limiting cases of Bianchi VII_h:
        Bianchi VII_0: non-zero sqrth but small Omega_K (1e-6) 
        Bianchi V: non-zero Omega_K and large  sqrth (around 1e5)

        Class has a particular method to assign Omega_b, Omega_cdm,
        and Omega_Lambda given Omega_K. The user may inform only Omega_K
        and the other parameters will be inferred from Class.

        Some useful variables are:
        ztable : array
            Table of redshift values.
        etatable : array
            Table of conformal time values.
        atable : array
            Table of scale factor values.
        Hcaltable : array
            Table of Hubble parameter values.
        tauprimetable : array
            Table of derivative of optical depth values.
        """

        # Default values
        self.c_light = 299792458  # Meters/second
        self.h_hubble = 0.67810  # H0 = 100*h_hubble*km/s/Mpc
        self.Omega_K = 0.000001  # Curvature density today
        self.z_reio = 7.6711  # Reionization redshift
        self.cutoff_multipole = 30  # Cutoff multipole for Boltzmann hierarchy
        self.sqrth = 0.1  # (spiraling length) = sqrth*(curvature radius)
        self.epsilon_TC = 0.003  # Condition to find the Tight Coupling regime
        self.IC = 'isocurvature'  # Can be 'isocurvature' or 'octupole'.
        self.verbose = False
        self.gauge = 'synchronous'  # Can be synchronous or newtonian. Changes only the velocities.
        # These variables are not independent from Omega_K.
        # Class has a particular method to assign them. They are either assingned as 
        # input or from Class routines
        self.Omega_b = None
        self.Omega_Lambda = None
        self.Omega_cdm = None
        self.Omega_m = None
                
        for key, value in params.items():  # load external parameters if provided,
            if hasattr(self, key):         # otherwise uses default.
                setattr(self, key, value)
        
        # Exceptions
        if not isinstance(self.h_hubble, (float, int)):
            raise TypeError(f"float or int expected, not {type(self.h_hubble).__name__} ")
        if self.h_hubble <= 0:
            raise ValueError("Hubble constant must be a non zero positive number")
        if not isinstance(self.Omega_K, (float, int)):
            raise TypeError(f"float or int expected, not {type(self.Omega_K).__name__} ") 
        if np.abs(self.Omega_K) > 1:
            raise ValueError("the absolute value of the curvature density cannot be larger than 1")
        if self.Omega_K == 0:
            self.Omega_K = 1e-6
            warnings.warn("Omega_K was set to 1e-6")
        if not isinstance(self.z_reio, (float, int)):
            raise TypeError(f"float or int expected, not {type(self.z_reio).__name__} ")
        if not isinstance(self.cutoff_multipole, int):
            raise TypeError(f"int expected, not {type(self.cutoff_multipole).__name__} ")
        if self.cutoff_multipole > 100:
            raise ValueError("maximum value for the cutoff multipole is 100")
        if not isinstance(self.sqrth, (float, int)):
            raise TypeError(f"float or int expected, not {type(self.sqrth).__name__} ")
        # if self.sqrth < 0.01:
        #     warnings.warn("small values of sqrth may take more time to compute")
        if not isinstance(self.epsilon_TC, (float, int)):
            raise TypeError(f"float or int expected, not {type(self.epsilon_TC).__name__} ")
        if self.IC != 'isocurvature' and self.IC != 'octupole':
            raise ValueError("initial conditions for vector modes must be either 'isocurvature' or 'octupole' ")
        if self.gauge != 'synchronous' and self.gauge != 'newtonian':
            raise ValueError("gauge must be either 'synchronous' or 'newtonian' ")
        if self.Omega_b is not None:
            if not isinstance(self.Omega_b, (float)):
                raise TypeError(f"float or int expected, not {type(self.Omega_b).__name__} ")
        if self.Omega_cdm is not None:
            if not isinstance(self.Omega_cdm, (float)):
                raise TypeError(f"float or int expected, not {type(self.Omega_cdm).__name__} ")
        if self.Omega_Lambda is not None:
            if not isinstance(self.Omega_Lambda, (float)):
                raise TypeError(f"float or int expected, not {type(self.Omega_Lambda).__name__} ")
        if self.Omega_m is not None:
            if not isinstance(self.Omega_m, (float)):
                raise TypeError(f"float or int expected, not {type(self.Omega_m).__name__} ")
        if self.Omega_b is not None and self.Omega_cdm is not None and self.Omega_m is not None:
            if (self.Omega_b + self.Omega_cdm != self.Omega_m):
                raise ValueError(f"Inconsistence in the matter component \
                                 {self.Omega_b} + {self.Omega_cdm} != {self.Omega_m}.\
                                 Please provide either Omega_b and Omega_cdm or Omega_m.")
        if self.Omega_b is not None and self.Omega_cdm is not None and self.Omega_Lambda is not None:
            if np.abs(self.Omega_b + self.Omega_cdm + self.Omega_Lambda + self.Omega_K - 1.) > 1e-4:
                raise ValueError(f"Incosistence in the energy component.\
                                 Condition |Omega_m + Omega_Lambda + Omega_K - 1| < 1e-4  \
                                 not satisfied.")  
        if self.Omega_m is not None and self.Omega_Lambda is not None:
            if np.abs(self.Omega_m + self.Omega_Lambda + self.Omega_K - 1.) > 1e-4:
                raise ValueError(f"Incosistence in the energy component.\
                                 Condition |Omega_m + Omega_Lambda + Omega_K - 1| < 1e-4  \
                                 not satisfied.")  
        
        self.DH = self.c_light / 100000 / self.h_hubble  # Hubble horizon in Mpc
        self.curvature_radius = self.DH / np.sqrt(np.abs(self.Omega_K))  # ell_c in Mpc
            # to work with dimensionless variables, set curvature_radius = 1
        self.calK = -1 * np.sign(self.Omega_K)  # Dimensionless curvature constant (i.e., -1, 0, +1).
            # This is called \cal{K} in arXiv:1909.13688.
        self.curvature_constant = -1 * self.Omega_K / self.DH**2  # Dimensional curvature constant (i.e., <, =, > 0)
                    
        # Calling Class
        params = {'Omega_k':self.Omega_K,'z_reio':self.z_reio, 'h': self.h_hubble}
        if self.Omega_b is not None:
            params.update({'Omega_b' : self.Omega_b})
        if self.Omega_cdm is not None:
            params.update({'Omega_cdm' : self.Omega_cdm})
        if self.Omega_m is not None:
            params.update({'Omega_m' : self.Omega_m})
        if self.Omega_Lambda is not None:
            params.update({'Omega_Lambda' : self.Omega_Lambda})
        backcosmo = Class() 
        backcosmo.set(params)
        backcosmo.compute()  # Running CLASS
        background_class = backcosmo.get_background()
        thermodynamics_class = backcosmo.get_thermodynamics()

        # Extracting background info from this CLASS run
        self.ztable = background_class.get('z')  # Table of z values (redshift)
        self.etatable = background_class.get('conf. time [Mpc]')  # Table of \eta values (conformal time)
        self.Hcaltable = background_class.get('H [1/Mpc]')/(1+self.ztable)  # Table of \cal{H} values (Hubble param. in conformal time)
        self.rho_baryons = background_class.get('(.)rho_b')
        self.rho_photons = background_class.get('(.)rho_g')
        self.rho_neutrinos = background_class.get('(.)rho_ur')
        self.Omega_r = backcosmo.Omega_r()  # All relativistic content
        self.Omega_g = backcosmo.Omega_g()  # Photons
        self.Omega_nu = backcosmo.Omega_r() - backcosmo.Omega_g() 
        self.Omega_m = backcosmo.Omega_m()  # All matter (cdm and baryons)
        self.Omega_b = backcosmo.Omega_b()  # Baryons only
        self.Omega_Lambda = 1. - self.Omega_K - self.Omega_m
        self.Omega_cdm = self.Omega_m - self.Omega_b
        self.ratio_neutrino_photons = self.Omega_nu / self.Omega_g

        # The range of values for eta or a is reduced for thermodynamics (does not start as early as other things)
        # Also CLASS provides these thermodynamic quantities in reversed chronological order. so [::-1] reverse it.
        self.tauprimetable = thermodynamics_class.get("kappa' [Mpc^-1]")[::-1]
        self.expmtautable = thermodynamics_class.get("exp(-kappa)")[::-1]
        self.etatable2 = thermodynamics_class.get('conf. time [Mpc]')[::-1]
        self.eta0 = self.etatable2[-1]
        self.etamin = self.etatable2[0]
        #self.chilist = np.linspace(0.001,self.eta0 - self.etamin,len(self.etatable2))

        # Storing interpolations
        self.calH_interp = CubicSpline(self.etatable, self.Hcaltable)
        self.scalefac_interp = CubicSpline(self.etatable, 1 / (1 + self.ztable))
        self.rho_g_interp = CubicSpline(self.etatable, self.rho_photons)
        self.rho_b_interp = CubicSpline(self.etatable, self.rho_baryons)
        self.tauprime_interp = CubicSpline(self.etatable2, self.tauprimetable)
        self.Rbg = CubicSpline(self.etatable, 3. / 4. * self.rho_baryons/self.rho_photons)
        
        self.Hcaltable2 = self.calH_interp(self.etatable2)
        self.atable = thermodynamics_class.get('scale factor a')[::-1]
        
        # bianchi parameters
        self.spiraling_length = self.sqrth * self.curvature_radius
        if self.spiraling_length < 100:
            warnings.warn(f"small value of spiraling length. It may take longer to compute. \
                          spiraling_length = {self.spiraling_length} Mpc. \
                          Recall that spiraling_length = sqrth * curvature_radius")

        if self.verbose:
            print("parameters are:")
            print(f"h = {self.h_hubble}")
            print(f"Omega_b = {self.Omega_b}")
            print(f"Omega_cdm = {self.Omega_m - self.Omega_b}")
            print(f"Omega_photons = {self.Omega_g}")
            print(f"Omega_nu = {self.Omega_nu}")
            print(f"curvature radius = {self.curvature_radius} Mpc")
            print(f"spiraling_length = {self.spiraling_length} Mpc")

    def __bianchi_parameters(self, m):
        """ Initialize variables that are Bianchi model dependent
        
        Parameters:
        -----------
        m : int
            perturbation type: m = 1 (vector) or m = 2 (tensor)
        """

        # Generalized Fourier mode
        if self.calK > 0:
            self.nu = 3 / self.curvature_radius  # Bianchi IX
        else:
            self.nu = m / self.spiraling_length + 1j / self.curvature_radius
        if self.verbose:
            if m == 1:
                svt_type = 'vector'
            if m == 2:
                svt_type = 'tensor'
            print(f"generalized fourier mode for {svt_type} modes is {self.nu}")
        self.nured = self.nu * self.curvature_radius  # Reduced (i.e., dimensionless) Fourier mode
        # eq. D.2 in arxiv:1909.13688
        self.calS2 = -self.nu**2 + self.calK / self.curvature_radius**2 
        self.fourier_k = np.abs(self.nu)
        self.index_TC_k_condition = np.where(self.fourier_k/self.tauprimetable < self.epsilon_TC )[0][-1]  # Index where TC ends
        self.index_TC_H_condition = np.where(self.Hcaltable2/self.tauprimetable < self.epsilon_TC )[0][-1]
        self.end_tightc_k = self.etatable2[self.index_TC_k_condition]
        self.end_tightc_H = self.etatable2[self.index_TC_H_condition]
        if self.verbose:
            print('TC conditions on k and H are at eta = ',self.end_tightc_k,' and ',self.end_tightc_H)
        self.index_end_tightc = min(self.index_TC_k_condition, self.index_TC_H_condition)
        self.end_tightc = self.etatable2[self.index_end_tightc]  # Instant when fourier_k/calH*tau' = 1%

    def __alm_healpy(self, alm_array_list, ell_max, m):
        """Sets the alms in the format expected by healpy

        Only the m = 0 alms are non-zero. These values
        are provided and are set in the correct format
        expected by healpy

        Parameters
        ----------
        alm_array_list : list
            List that contains all the
            non-zero multipoles for 
            temperature and/or polarization
        ell_max : int
            Maximum multipole
        m : int
            Perturbation type
        
        Returns
        -------
        array or list of arrays
            Alm coefficients in the format expected by healpy
        """
        if m == 2:
            inter = [2 * ell_max + 1,3 * ell_max]
        else:
            inter = [ell_max + 1,2 * ell_max + 1]
        len_alm = sum([l+1 for l in range(ell_max+1)])  # a_{l,-m} = (-1)^m a*_{lm}
        row = [0 for _ in range(ell_max - m + 1)]
        col = [i for i in range(inter[0],inter[1])]
        output_list = []
        for i, alm_array in enumerate (alm_array_list):
            alm = coo_matrix((alm_array,
                                (row,col)
                                ),
                                shape=(1,len_alm),
                                dtype='complex'
                                )
            alm = np.array(alm.todense())[0]
            output_list.append(alm)   
        return output_list

    ### Tensor modes ###

    def __tensor_call(self, eta, y, ell_max):
        """Computes the values needed for tensor_hierarchy.

        Parameters
        ----------
        eta : float
            Conformal time.
        y : 1d complex array of  4 * ell_max - 2 elements
            (beta, beta', ur, T, E, B).
        ell_max : int
            Multipole where truncation happens.

        Returns
        -------
        complex array
            Array of 4 * ell_max - 2 entries containing the
            derivatives of each variable.
        """

        calH = float(self.calH_interp(eta))
        scalefac = float(self.scalefac_interp(eta))
        rho_g = float(self.rho_g_interp(eta))
        tauprime = float(self.tauprime_interp(eta))
        return tensor_hierarchy(eta,
                            y,
                            ell_max,
                            self.nu,
                            self.curvature_radius,
                            self.calS2,
                            self.calK,
                            calH,
                            scalefac,
                            rho_g,
                            self.ratio_neutrino_photons,
                            tauprime,
                            self.ur_index,
                            self.kappa0_array_tensor,
                            self.kappa2_array_tensor,
                            self.denom_array_tensor,
                            self.zeta_array_tensor)

    def __tensor_tight_call(self, eta, y, ell_max):
        """Computes the values needed for tensor_tight_coupling_hierarchy.

        Parameters
        ----------
        eta : float
            Conformal time.
        y : 1d complex array of 4 * ell_max - 2 elements
            (beta, beta', ur, T, E, B).
        ell_max : int
            Multipole where truncation happens.

        Returns
        -------
        complex array
            Array of 4 * ell_max - 2 entries containing the
            derivatives of each variable.
        """

        calH = float(self.calH_interp(eta))
        scalefac = float(self.scalefac_interp(eta))
        rho_g = float(self.rho_g_interp(eta))
        return tensor_tight_coupling_hierarchy(eta,
                            y,
                            ell_max,
                            self.curvature_radius,
                            self.calS2,
                            self.calK,
                            calH,
                            scalefac,
                            rho_g,
                            self.ratio_neutrino_photons,
                            self.ur_index,
                            self.kappa0_array_tensor,
                            self.denom_array_tensor,
                            self.zeta_array_tensor)

    def tensor_solver(self, calS2=None, method='RK45', rtol=1e-7, atol=1e-10):
        """Solver for (beta, beta') and multipoles (N, T, E, B) as a function of time.

        The shear beta and beta' and the multipoles N and T during
        the tight coupling phase are stored in the array Anilos.tensor_tc (for later use
        in the source terms of the integral solution) with
        the following structure:
        Anilos.tensor_tc: [beta, beta', N_2, T_2, N_3, T_3, ...]

        The full hierarchy outside the tight coupling regime is stored in Anilos.nteb_tensor
        with the following structure:
        Anilos.nteb_tensor: [beta, beta', N_2, T_2, E_2, B_2, N_3, T_3, E_3, B_3, ...]

        Parameters
        ----------
        calS2 : complex, default: Eq. D.2 in arxiv:1909.13688
            Eq. D.2 in arxiv:1909.13688
        method : str, default: RK45
            Method used to solve the system
        rtol, atol : float, default: 1e-7 and 1e-10
            Precision (see solve_ivp documentation for details)
        """

        ell_max = self.cutoff_multipole
        self.__bianchi_parameters(2)  # Set Bianchi variables
        if not calS2:
            calS2 = self.calS2
        if self.calK > 0:
            if ell_max != 2:
                warnings.warn("ell_max must be 2 in Bianchi IX. Changing the value.")   
                ell_max = 3 
            elif ell_max == 2:
                ell_max = 3
        
        ####################################
        # Tight coupling phase
        ####################################
        time_init_tc = self.etatable2[0]  # Start of tight coupling approximation
        time_end_tc = self.end_tightc  # End of tight coupling (defined when fourier_k/calH*tau' = 1%)
        time_span_tc = [time_init_tc, time_end_tc]

        # Initializes beta, beta' and ur during tight coupling phase
        betaur0 = np.zeros(4*(ell_max-1)+2,dtype=complex)
        # Set initial conditions on beta and beta'
        betaur0[0] = 1
        betaur0[1] = -self.fourier_k**2*self.etatable2[0]/3

        # Setting hierarchy variables needed for the coefficients
        self.ur_index = np.array([2 + 4 *(i -2) for i in range(2,ell_max+1)])
        self.kappa0_array_tensor = kappa_array(3, ell_max, 0, 2, self.nu, self.curvature_radius)
        self.kappa2_array_tensor = kappa_array(3, ell_max, 2, 2, self.nu, self.curvature_radius)
        self.denom_array_tensor = np.array([1./(2.*i + 5.) for i in range(0, ell_max - 1)])
        self.zeta_array_tensor = zeta_factor_array(3, ell_max, self.sqrth, 2)

        betaur_table = solve_ivp(self.__tensor_tight_call,
                                time_span_tc,
                                betaur0,
                                method=method,
                                t_eval=self.etatable2[0:self.index_end_tightc],
                                rtol=rtol, # scipy's default = 1e-3
                                atol=atol,  # scipy's default = 1e-6
                                args = (ell_max,)).y
        self.tensor_tc = betaur_table[0:3, :]

        ####################################
        # after tight coupling
        ####################################
        time_init = self.end_tightc  # Start at end_tightc (when fourier_k/calH*tau' = 1%)
        time_end = self.etatable2[-1]
        time_span = [time_init, time_end]

        # Note on early convention: nteb was named initally for Neutrinos, Temp, E-mode and B-mode.
        # then we included the shear, but didn't bother changing this name.
        nteb0 = np.zeros(4*(ell_max-1)+2,dtype=complex)  # Initializes beta, beta', ur, T, E and B

        # Set initial conditions from values at the end of tight coupling approximation
        nteb0[0] = betaur_table[0][-1]  # beta at end of tight coupling
        nteb0[1] = betaur_table[1][-1]  # beta' at end of tight coupling
        nteb0[3] = -4/3*betaur_table[1][-1]/self.tauprime_interp(self.end_tightc)  # T2(t_init)
        nteb0[4] = np.sqrt(2/3)*betaur_table[1][-1]/self.tauprime_interp(self.end_tightc)  # E2(t_init)
        # Initializes all neutrino multipoles
        for j in range(2,ell_max):
            nteb0[4*(j-2)+2] = betaur_table[4*(j-2)+2][-1]  # Neutrino quadripole at end of tight coupling

        nteb_table_full = solve_ivp(self.__tensor_call,
                                    time_span,
                                    nteb0,
                                    method=method,
                                    t_eval=self.etatable2[self.index_end_tightc:],
                                    rtol=rtol,
                                    atol=atol,
                                    args = (ell_max,)
                                    )
        nteb_table = nteb_table_full.y
        self.nteb_tensor = nteb_table

    def __sources_tensor(self):
        """Sources of temperature and polarization needed for los integration.

        Computes the sources at each value in etatable2 and creates 
        interpolated functions.
        """

        # Array position of end of tight coupling
        itc = self.index_end_tightc

        # beta', T_2 and E_2 during and after tight coupling
        betap_before = self.tensor_tc[1]
        temp2_before = -4. / 3. * betap_before / self.tauprimetable[0:itc]
        elet2_before = np.sqrt(2. / 3.) * betap_before / self.tauprimetable[0:itc] 
        betap_after = self.nteb_tensor[1]
        temp2_after = self.nteb_tensor[3]
        elet2_after = self.nteb_tensor[4]
        st2_full = np.empty(self.expmtautable.shape[0], dtype = 'complex')
        se2_full = np.empty(self.expmtautable.shape[0], dtype = 'complex')

        # Sources during tight coupling
        st2_full[0:itc] = self.expmtautable[0:itc] * (self.tauprimetable[0:itc] / 10
                                                    * (temp2_before - np.sqrt(6) * elet2_before)
                                                    - betap_before
                                                    )
        se2_full[0:itc] = self.expmtautable[0:itc] * (-np.sqrt(6) * self.tauprimetable[0:itc] /10
                                                    *(temp2_before - np.sqrt(6) * elet2_before)
                                                    )

        # Sources after tight coupling
        st2_full[itc:] = self.expmtautable[itc:] * (self.tauprimetable[itc:] / 10
                                                * (temp2_after - np.sqrt(6) * elet2_after)
                                                - betap_after # @Cyril: -betap or +betap?
                                                )              # Kernel of integral 7.33 in Harmonics paper

        se2_full[itc:] = self.expmtautable[itc:] * (-np.sqrt(6) * self.tauprimetable[itc:] / 10
                                                * (temp2_after - np.sqrt(6) * elet2_after)
                                                    ) # Kernel of integral 7.33 in Harmonics paper

        #store the interpolated quantities
        self.st2_tensor_interp = CubicSpline(self.etatable2,st2_full)
        self.se2_tensor_interp = CubicSpline(self.etatable2,se2_full)

    def alm_tensor(self, ell_max, healpy = False):
        """Computes alm_T, alm_E, and alm_B simultaneously
        using line-of-sight integration.

        Integration is performed for multipoles 2 to ell_max.
        The only non-zero a_{lm} are a_{l0}. They are stored in
        Anilos.almT_tensor, Anilos.almE_tensor, and Anilos.alm_B_tensor
        either as 1d arrays with length ell_max - 2 (if healpy is set to false),
        in which only the non-zero a_{lm} are stored, or as sparse matrices,
        which can be passed into healpy.

        Parameters
        ----------
        ell_max : int
            Maximum multipole. It does not need to be
            equal to the cutoff multipole.
        healpy : bool, default = False
            Whether to set the alms into a format ready to be used in healpy.
        """

        if self.calK < 0:
            ell_max = ell_max + 5 # Avoid errors in the last element
        if self.calK > 0:
            if ell_max != 2:
                warnings.warn("There are no non zero multipoles \
                              higher than the quadrupole in Bianchi IX")   
                ell_max = 3 
            elif ell_max == 2:
                ell_max = 3

        self.__bianchi_parameters(2)
        ellc = self.curvature_radius
        sqrth = self.sqrth
        # Set the sources
        self.__sources_tensor()
        chimax = self.eta0 - self.etamin
        chimin = 0.001
        grid_chi = np.arange(chimin,chimax,20.)  # Grid of points where the integrated function is calculated
        len_grid = len(grid_chi)
        theta_ell_grid = np.zeros((len_grid, ell_max-1), dtype = 'complex')
        Emode_ell_grid = np.zeros((len_grid, ell_max-1), dtype = 'complex')
        Bmode_ell_grid = np.zeros((len_grid, ell_max-1), dtype = 'complex')
        source_gridT = self.st2_tensor_interp(self.eta0-grid_chi)
        source_gridEB = self.se2_tensor_interp(self.eta0-grid_chi)
        nu = self.nu

        # Filling the arrays to integrate
        if self.calK < 0:
            zeta_array = zetas_array(2, ell_max, sqrth, 2)
            zeta_term = zeta_array[0]
            normalization = np.array([(1j) ** i * np.sqrt(4 * np.pi * (2. * i + 1.)) for i in range(2, ell_max +1)])
            zeta_array = np.tile(zeta_array, (len_grid, 1))
            normalization = np.tile(normalization, (len_grid, 1))

            source_termt = np.tile(np.array([source_gridT]).T, (1, ell_max -1))
            source_terme = np.tile(np.array([source_gridEB]).T, (1, ell_max -1))
            epsilonT, epsilonE, betaB = epsbeta_for_tensor( grid_chi, ell_max, nu, ellc, len_grid)
            
            theta_ell_grid = normalization * epsilonT *zeta_array/zeta_term*source_termt
            Emode_ell_grid  = normalization * (epsilonE *zeta_array/zeta_term*source_terme)
            Bmode_ell_grid  = -normalization*(betaB *zeta_array/zeta_term *source_terme)
        else:
            theta_ell_grid[:,0] = source_gridT
            Emode_ell_grid[:,0] = np.cos(2 * grid_chi/ellc) * source_gridEB
            Bmode_ell_grid[:,0] = np.sin(2 * grid_chi/ellc) * source_gridEB

        theta_ell_grid = theta_ell_grid.T
        Emode_ell_grid = Emode_ell_grid.T
        Bmode_ell_grid = Bmode_ell_grid.T
        self.almT_tensor = trapezoid(theta_ell_grid,grid_chi)
        self.almE_tensor = trapezoid(Emode_ell_grid,grid_chi)
        self.almB_tensor = trapezoid(Bmode_ell_grid,grid_chi)
        if self.calK < 0:
            self.almT_tensor = self.almT_tensor[:-5]
            self.almE_tensor = self.almE_tensor[:-5]
            self.almB_tensor = self.almB_tensor[:-5]
            ell_max = ell_max - 5
        if healpy == True:
            self.almT_tensor, self.almE_tensor, self.almB_tensor = \
                self.__alm_healpy([self.almT_tensor, self.almE_tensor, self.almB_tensor], ell_max, 2)

    ### Vector modes ###

    def __pos_v(self, key, ell=1):
        """Indices of rows and collumns of free streamimg matrix elements
        for vector modes (m=1)

        [0,1,2,3,4,5] corresponding to [Phi,vb,ur,T,E,B]
        note that Phi and vb modes do not repeat in the hierarchy.
        """

        pos = {'Phi': 0,
            'vb': 1,
            'ur': 2 + 4*(ell-1),
            'T': 3 + 4*(ell-1),
            'E': 4 + 4*(ell-1),
            'B': 5 + 4*(ell-1)
            }
        if ((key == 'Phi' or key == 'vb') and ell != 1):
            raise Exception('Phi and baryon velocity defined at ell=1 only')
        return pos[key]

    def __vector_call(self, eta, y, ell_max):
        """Computes the values needed by vector_hierarchy

        Parameters
        ----------
        eta : float
            Conformal time
        y : 1d complex array of  4 * ell_max - 2 elements
            (beta,beta',ur,T,E,B)
        ell_max : int
            Multipole where truncation happens

        Returns
        -------
        complex array
            Array of 4 * ell_max - 2 entries containing the
            derivatives of each variable
        """

        tauprime = float(self.tauprime_interp(eta))
        calH = float(self.calH_interp(eta))
        scalefac = float(self.scalefac_interp(eta))
        rho_g = float(self.rho_g_interp(eta))
        Rbg = float(self.Rbg(eta))
        return vector_hierarchy(eta,
                                y,
                                ell_max,
                                self.nu,
                                self.curvature_radius,
                                self.curvature_constant,
                                calH,
                                scalefac,
                                rho_g,
                                tauprime,
                                Rbg,
                                self.Omega_nu,
                                self.Omega_g,
                                self.ur_index,
                                self.kappa0_array_vector,
                                self.kappa2_array_vector,
                                self.denom_array_vector,
                                self.zeta_array_vector,
                                self.gauge)
                
    def __vector_tight_call(self, eta, y, ell_max):
        """Computes the values needed by vector_tight_coupling_hierarchy

        Parameters
        ----------
        eta : float
            Conformal time
        y : 1d complex array of  4 * ell_max - 2 elements
        (beta,beta',ur,T,E,B)
        ell_max : int
            Multipole where truncation happens

        Returns
        -------
        complex array
            Array of 4 * ell_max - 2 entries containing the
            derivatives of each variable
        """

        tauprime = float(self.tauprime_interp(eta))
        calH = float(self.calH_interp(eta))
        scalefac = float(self.scalefac_interp(eta))
        rho_g = float(self.rho_g_interp(eta))
        Rbg = float(self.Rbg(eta))
        return vector_tight_coupling_hierarchy(eta,
                                            y,
                                            ell_max,
                                            self.nu,
                                            self.curvature_radius,
                                            self.curvature_constant,
                                            calH,
                                            scalefac,
                                            rho_g,
                                            tauprime,
                                            Rbg,
                                            self.Omega_nu,
                                            self.Omega_g,
                                            self.ur_index,
                                            self.kappa0_array_vector,
                                            self.denom_array_vector,
                                            self.zeta_array_vector,
                                            self.gauge)
                
    def __setIC(self):
        """Set initial conditions.

        Initial conditions for non-decaying vector modes.
        Two ICs are implemented: 'isocurvature' and 'octupole'.
        Isocurvature: photons and neutrinos have opposite directions
        velocities [1].
        Octupole: an initial non-zero neutrino octupole [2].

        References:
        [1] Lewis, A. (2004). Observable primordial vector modes.
        Physical Review D, 70(4), 043518.
        [2] Rebhan, A. K., & Schwarz, D. J. (1994). 
        Kinetic versus thermal-field-theory approach to 
        cosmological perturbations. Physical Review D, 50(4), 2541.
        """
        
        #Initial values which are used for all types of initial conditions
        time_init = self.etatable2[0]
        self.time_init = time_init
        time_end = self.etatable2[-1]

        calH_init = self.calH_interp(time_init)
        self.calH_init = calH_init
        self.calH_0 = self.calH_interp(time_end)
        tauprime = self.tauprime_interp(time_init)
        Rbg = self.Rbg(time_init)
        curvature = self.curvature_constant
        nu = self.nu
        sqrth = self.sqrth
        alpha = np.sqrt(3.) / 5. * np.sqrt(nu**2 - 4 * curvature) 
        beta = 5. / 3. * alpha
        #today's abundances.
        Or = self.Omega_r
        Om = self.Omega_m
        Og = self.Omega_g
        Onu = self.Omega_nu
        #Ob = self.Omega_b   
            
        if self.IC == 'isocurvature':
            # See [1]
            self.Phi_i_0 = 1. # Need corrections of order eta
            self.Phi_i_1 = (-15. / (30. * Or + 8. * Onu * self.zeta_array_vector[0]) 
                            * Om * np.sqrt(Or) * self.calH_0 * self.Phi_i_0 * time_init
                            )
            self.Phi_i = self.Phi_i_0 + self.Phi_i_1

            # Other IC
            self.vf_i = (-5. / 4. * Or / (Og * self.zeta_array_vector[0]) - Onu / Og) / (1. + Rbg) * self.Phi_i_0
            self.T1_i = self.vf_i 

            # Neutrinos
            self.N1_i_0 = (5. * Or / self.zeta_array_vector[0] + 4 * Onu) / (4 * Onu) * self.Phi_i_0
            self.N1_i_1 = 0.
            self.N1_i = self.N1_i_0 + self.N1_i_1
            self.N2_i_1 = 5. / 12. * Or / Onu * self.kappa0_array_vector[0] * time_init * self.Phi_i_0
            self.N2_i_2 = (15. /6. / (30.*Or + 8. * Onu * self.zeta_array_vector[0]) 
                        * Om * np.sqrt(Or) * self.calH_0 * self.kappa0_array_vector[0] 
                        * self.zeta_array_vector[0] * self.Phi_i_0 * time_init**2
                        )
            self.N2_i = self.N2_i_1 + self.N2_i_2
            self.N3_i = (1. / 24. * Or / Onu * self.kappa0_array_vector[0] 
                        * self.kappa0_array_vector[1] * self.zeta_array_vector[1] 
                        * time_init**2 * self.Phi_i_0
                        )
            
            # Tight coupling corrections
            self.T2_i = 4. / 3. * beta * self.zeta_array_vector[0] * (self.T1_i- self.Phi_i) / tauprime 
            self.Vslip_i =  (-Rbg / tauprime / (1. + Rbg) 
                            * (calH_init * self.vf_i - alpha * self.T2_i / self.zeta_array_vector[0])
                            )
            # revoir et write with the kappaslm for better readbility

        if self.IC == 'octupole':
            # Alluded in Eqs 6.7 of Rebhan et al. (9403032) paper
            self.Phi_i_0 = 1.
            self.Phi_i_1 = (-15. / (30. * Or + 8. * Onu * self.zeta_array_vector[0]) 
                            * Om * np.sqrt(Or) * self.calH_0 *self.Phi_i_0 * time_init
                            )
            self.Phi_i = self.Phi_i_0 + self.Phi_i_1
            
            # Other IC
            self.vf_i = 0.
            self.T1_i = self.vf_i

            # Neutrinos
            self.N1_i_0 = 0.
            self.N1_i_1 = 0.
            self.N1_i_2 = -self.Phi_i_0 * Or / Onu /24 * self.kappa0_array_vector[0]**2 * time_init**2 #SHould put correct expression 
            self.N1_i = self.N1_i_0 + self.N1_i_1 + self.N1_i_2
            self.N2_i_1 = 5. / 12. * Or / Onu * self.kappa0_array_vector[0] * self.Phi_i_0 * time_init
            self.N2_i_2 = (15. / 6. / (30.* Or + 8. * Onu * self.zeta_array_vector[0]) 
                        * Om * np.sqrt(Or) * self.calH_0 * self.kappa0_array_vector[0] 
                        * self.zeta_array_vector[0] * self.Phi_i_0 * time_init**2
                        )
            self.N2_i = self.N2_i_1 + self.N2_i_2
            self.N3_i = (-7. / 12 * (5. * Or + 4. * Onu * self.zeta_array_vector[0]) / Onu 
                        * self.kappa0_array_vector[0] * self.zeta_array_vector[1] 
                        / self.kappa0_array_vector[1]  * self.Phi_i_0
                        )
            
            # Tight coupling corrections
            self.T2_i = 4. / 3. * beta * self.zeta_array_vector[0] * (self.T1_i - self.Phi_i) / tauprime
            self.Vslip_i =  (-Rbg / tauprime / (1.+Rbg) 
                            * (calH_init * self.vf_i - alpha * self.T2_i / self.zeta_array_vector[0] )
                            )
            
        self.vb_i = self.vf_i + self.Vslip_i / (1. + Rbg)
        self.vg_i = self.vf_i - Rbg* self.Vslip_i / (1. + Rbg)

        # We improve tight coupling on T2 and set the one on E2.
        self.T2_i = 4. / 3. * beta * self.zeta_array_vector[0] * (self.T1_i - self.Phi_i) / tauprime    
        self.E2_i = -np.sqrt(2. / 3.) * beta * self.zeta_array_vector[0] * (self.T1_i - self.Phi_i) / tauprime    

        #For Newtonian gauge we substract Phi from all velocities and dipoles.
        if self.gauge == 'newtonian':
            self.N1_i -= self.Phi_i
            self.T1_i -= self.Phi_i
            self.vg_i -= self.Phi_i
            self.vb_i -= self.Phi_i
            self.vf_i -= self.Phi_i

    def vector_solver(self, method='RK45', rtol=1e-7, atol=1e-10):
        """Solver of (Phi, vb) and multipoles (ur,T,E,B) 

        The vector potential, baryon velocity and the NTEB hierarchy up until
        the quadrupole during tight coupling regime are stored in
        Anilos.vector_tc with the following structure:
        Anilos.vector_tc: [Phi, vb, ur_1, T_1, E_1, B_1, ur_2, T_2, ...]

        The full hierarchy after tight coupling is stored in Anilos.nteb_vector
        with the following structure:
        Anilos.nteb_vector: [Phi, vb, ur_1, T_1, E_1, B_1, ur_2, T_2, E_2, B_2, ...]

        Parameters
        ----------
        ell_max : int, default: cutoff_multipole or 2 (Bianchi IX)
            Maxium value of ell to compute the hierarchy (max is 300)
        method : str, default: RK45
            Method to solve the system
        rtol, atol : float, default: 1e-7 and 1e-10
            Precision (see solve_ivp documentation for details)
        """

        self.__bianchi_parameters(1)  # Set Bianchi variables
        ell_max = self.cutoff_multipole

        num_var = 4 * ell_max + 2  # Total number of elements
        alpha = np.sqrt(3.) / 5. * np.sqrt(self.nu**2 - 4 * self.curvature_constant) 
        self.ur_index = np.array([2 + 4 * (i -1) for i in range(1,ell_max+1)])
        self.kappa0_array_vector = kappa_array(2, ell_max, 0, 1, self.nu, self.curvature_radius)
        self.kappa2_array_vector = kappa_array(2, ell_max, 2, 1, self.nu, self.curvature_radius)
        self.denom_array_vector = np.array([1. / (2. * i + 3.) for i in range(0, ell_max)])
        self.zeta_array_vector = zeta_factor_array(2, ell_max, self.sqrth, 1)
        
        ####################################
        # tight-coupling phase
        ####################################
        time_init_TC = self.etatable2[0]  # Start of tight-coupling approximation
        time_end_TC = self.end_tightc  # End of tight coupling (defined when fourier_k/calH*tau' = 1%)
        time_span_TC = [time_init_TC, time_end_TC]

        # Find out initial conditions. comment here that baryons here stand for the tight coupled fluid.
        self.__setIC()
        # Initializes beta, beta', ur, T, E and B type = complex or float
        nteb0_TC = np.zeros(num_var,dtype=complex)
            
        #Set initial conditions
        nteb0_TC[self.__pos_v('Phi')] = self.Phi_i  # Phi vector mode
        nteb0_TC[self.__pos_v('vb')] = self.vf_i  # Velocity of baryons which is in TC the tc fluid.
        nteb0_TC[self.__pos_v('ur', 1)] = self.N1_i  # Dipole of neutrinos
        nteb0_TC[self.__pos_v('ur', 2)] = self.N2_i
        nteb0_TC[self.__pos_v('ur', 3)] = self.N3_i

        nteb_table_TC = solve_ivp(self.__vector_tight_call,
                                time_span_TC,
                                nteb0_TC,
                                method= method,
                                t_eval=self.etatable2[0:self.index_end_tightc],
                                rtol=rtol,
                                atol=atol,
                                args = (ell_max, )
                                ).y

        # We shall now rebuild the results for baryons and photons from the Tight coupled fluid.
        vf_array = nteb_table_TC[self.__pos_v('vb')] 
        Phi_array = nteb_table_TC[self.__pos_v('Phi')]
        tauprime_array = self.tauprime_interp(self.etatable2[0:self.index_end_tightc])
        Rbg_array = self.Rbg(self.etatable2[0:self.index_end_tightc])
        calH_array = self.calH_interp(self.etatable2[0:self.index_end_tightc])

        #Need to revisit all this TC expansion. I am a bit inconsistent.
        if self.gauge == 'synchronous':
            T1_array_TC0 = vf_array - Phi_array
            vfsynchronous_array = vf_array
        if self.gauge == 'newtonian':
            T1_array_TC0 = vf_array
            vfsynchronous_array = vf_array + Phi_array
        T2_array_TC0 = 20. / 9. * alpha /tauprime_array  * T1_array_TC0 * self.zeta_array_vector[0]
        Vslip_array = (- Rbg_array / tauprime_array / (1. + Rbg_array) 
                    * (calH_array * vfsynchronous_array - alpha * T2_array_TC0 / self.zeta_array_vector[0])
                    )
        vb_array = vf_array + Vslip_array / (1. + Rbg_array)
        vg_array = vf_array - Rbg_array*Vslip_array / (1. + Rbg_array)
        T1_array =  vg_array
        nteb_table_TC[self.__pos_v('T', 1)] = T1_array
        nteb_table_TC[self.__pos_v('vb')]  = vb_array # Now vb is really vb a posteriori and not vf.

        # The photons quadrupole in temperature and E-polarization is also inferred
        T2_array = 4. / 3. * 5. / 3. * alpha / tauprime_array  * T1_array_TC0 * self.zeta_array_vector[0]
        E2_array = -np.sqrt(2. / 3.) * 5. / 3. * alpha / tauprime_array  * T1_array_TC0 * self.zeta_array_vector[0]
        nteb_table_TC[self.__pos_v('T', 2)] = T2_array
        nteb_table_TC[self.__pos_v('E', 2)] = E2_array
        self.vector_tc = nteb_table_TC[0:10]
        
        ####################################
        # After tight-coupling
        ####################################
        time_init = self.end_tightc  # Start at end_tightc (when fourier_k/calH*tau' = 1%)
        time_end = self.etatable2[-1]
        time_span = [time_init, time_end]
        nteb0 = np.zeros(num_var,dtype=complex) 
        # We now copy the final results from tight coupling.
        for j in range(num_var):
            nteb0[j] = nteb_table_TC[j][-1]

        nteb_table = solve_ivp( self.__vector_call,
                                time_span,
                                nteb0,
                                method=method,
                                t_eval=self.etatable2[self.index_end_tightc:],
                                rtol=rtol,
                                atol=atol,
                                args = (ell_max, )
                                ).y
        self.nteb_vector = nteb_table

    def __sources_vector(self):
        """Sources of temperature and polarization needed for los integration.

        Computes the sources at each value in etatable2 and creates 
        interpolated functions.
        """

        itc = self.index_end_tightc  # Array position of end of tight coupling
        curvature = self.curvature_constant
        nu = self.nu
        prefactor_Phi_source = ((8. / 5.) * np.sqrt(3.) * self.scalefac_interp(self.etatable2)**2
                            * self.rho_g_interp(self.etatable2) / np.sqrt(nu**2 - 4 * curvature)
                            )
        ratio_neutrino_photons = self.Omega_nu / self.Omega_g
        sqrth = self.sqrth
        st2_full1 = np.empty(len(self.tauprimetable), dtype= 'complex')
        st2_full2 = np.empty(len(self.tauprimetable), dtype= 'complex')
        se2_full = np.empty(len(self.tauprimetable), dtype= 'complex')
    
        #setting variables before TC
        phi_before = self.vector_tc[0]
        vb_before = self.vector_tc[1]
        temp1_before = self.vector_tc[3]
        ur2_before = self.vector_tc[6]
        if self.gauge == 'synchronous':
            temp2_before = ((4. / 9.) * self.kappa0_array_vector[0] * self.zeta_array_vector[0] 
                        * (temp1_before - phi_before) / self.tauprimetable[0:itc] 
                        )
            elet2_before = (- np.sqrt(2. / 3.) * (self.kappa0_array_vector[0] * self.zeta_array_vector[0] / 3) 
                        * (temp1_before - phi_before) / self.tauprimetable[0:itc] 
                        )
        else:
            temp2_before = ((4. / 9.) * self.kappa0_array_vector[0] * self.zeta_array_vector[0] 
                        * temp1_before / self.tauprimetable[0:itc]
                        )
            elet2_before = (- np.sqrt(2./3.) * (self.kappa0_array_vector[0] * self.zeta_array_vector[0] / 3) 
                        * temp1_before / self.tauprimetable[0:itc]
                        )

        #setting variables after TC
        phi_after = self.nteb_vector[0]
        vb_after = self.nteb_vector[1]
        ur2_after = self.nteb_vector[6]
        temp2_after = self.nteb_vector[7]
        elet2_after = self.nteb_vector[8]
        
        if self.gauge == 'synchronous':
            st2_full1[0:itc] = self.expmtautable[0:itc] * self.tauprimetable[0:itc] * vb_before
            st2_full2[0:itc] =  (self.expmtautable[0:itc]
                                * ( self.tauprimetable[0:itc]
                                *((1/10) * temp2_before - (np.sqrt(6)/10) * elet2_before) 
                                - (self.kappa0_array_vector[0] * self.zeta_array_vector[0] / 3) 
                                * phi_before)
                                )
            st2_full1[itc:] = self.expmtautable[itc:] * self.tauprimetable[itc:] * vb_after
            st2_full2[itc:] =  (self.expmtautable[itc:]
                            * ( self.tauprimetable[itc:]
                            * ((1/10) * temp2_after - (np.sqrt(6)/10) * elet2_after) 
                            - (self.kappa0_array_vector[0] * self.zeta_array_vector[0] / 3)
                            * phi_after)
                            )
        else:
            st2_full1[0:itc]  = (self.expmtautable[0:itc]
                                * ( self.tauprimetable[0:itc] * vb_before
                                + 2 * self.Hcaltable2[0:itc] * phi_before 
                                - prefactor_Phi_source[0:itc] * ratio_neutrino_photons * ur2_before 
                                - prefactor_Phi_source[0:itc] * temp2_before)
                                )
            st2_full2[0:itc] = (self.expmtautable[0:itc] 
                            * (self.tauprimetable[0:itc] / 10) 
                            * (temp2_before - np.sqrt(6) * elet2_before)
                            )
            st2_full1[itc:] = (self.expmtautable[itc:]
                            * (self.tauprimetable[itc:] * vb_after
                            + 2*self.Hcaltable2[itc:] * phi_after
                            - prefactor_Phi_source[itc:] * ratio_neutrino_photons * ur2_after 
                            - prefactor_Phi_source[itc:] * temp2_after)
                            )
            st2_full2[itc:]  = (self.expmtautable[itc:] 
                            * (self.tauprimetable[itc:] / 10 )
                            *(temp2_after - np.sqrt(6)* elet2_after)
                            )
            
        se2_full[0:itc] = (self.expmtautable[0:itc]
                        * (self.tauprimetable[0:itc] / 10)
                        * (-np.sqrt(6) *temp2_before + 6 * elet2_before)
                        )
        se2_full[itc:] = (self.expmtautable[itc:]
                        * (self.tauprimetable[itc:] / 10) 
                        * (-np.sqrt(6) * temp2_after + 6 * elet2_after)
                        )

        #interpolated functions
        self.st2_interp1 = CubicSpline(self.etatable2,st2_full1)
        self.st2_interp2 = CubicSpline(self.etatable2,st2_full2)
        self.se2_interp = CubicSpline(self.etatable2,se2_full)

    def alm_vector(self, ell_max, healpy = False):
        """Computes alm_T, alm_E and alm_B simultaneously
        using line-of-sight integration

        Integration is performed for multipoles 1 to ell_max.
        The only non-zero a_{lm} are a_{l0}. They are stored in
        Anilos.almT_tensor, Anilos.almE_tensor and Anilos.alm_B_tensor
        either as 1d arrays with length ell_max - 1 (if healpy is set to false),
        in which are stored only the non-zero a_{lm}, or as sparse matrices,
        which can be passed into healpy.

        Parameters
        ----------
        ell_max : int
            Maximum multipole. It does not need to be
            equal to the cutoff multipole
        healpy : bool, default = False
            Whether to set the alms into a format ready to be used in healpy
        """

        if self.calK > 0:
            raise ValueError("Bianchi IX does not generate vector perturbation")
        ell_max = ell_max + 5 # Avoid errors in the last element
        self.__bianchi_parameters(1)
        ellc = self.curvature_radius
        sqrth = self.sqrth
        nu = self.nu
        chimax = self.eta0 - self.etamin
        chimin = 0.001
        grid_chi = np.arange(chimin,chimax,20.)  # Grid of points where the integrated function is calculated
        len_grid = len(grid_chi)
        Bmode_ell_grid =  np.empty((ell_max, len_grid), dtype = 'complex')

        #set the sources
        self.__sources_vector()
        source_grid1T = np.array([self.st2_interp1(self.eta0-grid_chi)]).T
        source_grid2T = np.array([self.st2_interp2(self.eta0-grid_chi)]).T
        source_gridEB = np.array([self.se2_interp(self.eta0-grid_chi)]).T

        zeta_array = zetas_array(1, ell_max, sqrth, 1)
        zeta_term1 = zeta_array[0]
        zeta_term2 = zeta_array[1]
        normalization = np.array([(1j) ** i * np.sqrt(4 * np.pi * (2. * i + 1.)) for i in range(1, ell_max +1)])
        zeta_array = np.tile(zeta_array, (len_grid, 1))
        normalization = np.tile(normalization, (len_grid, 1))
        source_termt1 = np.tile(source_grid1T, (1, ell_max))
        source_termt2 = np.tile(source_grid2T, (1, ell_max))
        source_terme = np.tile(source_gridEB, (1, ell_max))
        epsilon1T, epsilon2T, epsilonE, betaB = epsbeta_for_vector( grid_chi, ell_max, nu, ellc, len_grid)

        # Filling the arrays to integrate
        theta_ell_grid = (normalization * zeta_array 
                        * (epsilon1T * source_termt1 / zeta_term1 + epsilon2T * source_termt2 / zeta_term2 )
                        )
        Emode_ell_grid = normalization * epsilonE * zeta_array * source_terme / zeta_term2
        Bmode_ell_grid = - normalization * betaB * zeta_array * source_terme / zeta_term2
                
        theta_ell_grid = theta_ell_grid.T
        Emode_ell_grid = Emode_ell_grid.T
        Bmode_ell_grid = Bmode_ell_grid.T
        self.almT_vector = trapezoid(theta_ell_grid,grid_chi)[:-5]
        self.almE_vector = trapezoid(Emode_ell_grid,grid_chi)[:-5]
        self.almB_vector = trapezoid(Bmode_ell_grid,grid_chi)[:-5]
        if healpy == True:
            self.almT_vector, self.almE_vector, self.almB_vector = \
                self.__alm_healpy([self.almT_vector, self.almE_vector, self.almB_vector], ell_max - 5, 1)

