"""
airProps.py

import with: 
from airProps import airProps

Function:
airProps(Tin, T_TubeS)
Returns:
(Density_air, Cp_air, K_air, KVisc_air, Pr_air, Dvisc_air, Pr_tube, Dvisc_tube)
    
Interpolating based on properties from 
Heat and Mass Transfer 
Fundamental & Applications 
Yunus A. Cengel 
Afshin J. Ghajar 
5th Ed. 
ISBN 978-0-07-339818-1


Parameters
----------
Tin: Inlet temperature [C]
T_TubeS: Melting temperature (Tube surface temp) [C]

Returns
-------
Density_air: Density [kg/m^3].
Cp_air: Specific heat [J/(kg·K)]
K_air: Thermal conductivity [W/mK].
KVisc_air: Kinematic viscosity [m^2/s].
Pr_air: Prandtl number at bulk temperature (dimensionless).
Dvisc_air: Dynamic viscosity evaluated at bulk air temperature [kg/(m·s)].
Pr_tube: Prandtl number at tube surface temperature (dimensionless).
Dvisc_tube: Dynamic viscosity evaluated at tube surface [kg/(m·s)].

"""

from typing import Union
import numpy as np

ArrayLike = Union[float, int, np.ndarray, list, tuple]

def airProps(
    Tin: ArrayLike,
    T_TubeS: ArrayLike):    

    #Reference data (C)
    Temp   = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float) # C
    Density= np.array([1.269, 1.246, 1.225, 1.204, 1.184, 1.164, 1.145, 1.127], dtype=float)   # kg/m^3
    Cp     = np.array([1006, 1006, 1007, 1007, 1007, 1007, 1007, 1007], dtype=float)          # J/kg-K
    K      = np.array([0.02401, 0.02439, 0.02476, 0.02514, 0.02551, 0.02588, 0.02625, 0.02662], dtype=float)  # W/m-K
    KVisc  = np.array([1.382e-5, 1.426e-5, 1.470e-5, 1.516e-5, 1.562e-5, 1.608e-5, 1.655e-5, 1.702e-5], dtype=float) # m^2/s
    Dvisc  = np.array([1.754e-5, 1.778e-5, 1.802e-5, 1.825e-5, 1.849e-5, 1.872e-5, 1.895e-5, 1.918e-5], dtype=float) # kg/m-s

    #Track scalar-ness of inputs
    Tin_is_scalar = np.ndim(Tin) == 0
    Ts_is_scalar = np.ndim(T_TubeS) == 0

    #Work with 1-D arrays internally
    Tin_1d = np.atleast_1d(np.asarray(Tin, dtype=float))
    Ts_1d = np.atleast_1d(np.asarray(T_TubeS, dtype=float))

    #Inlet Properties
    Density_air_1d = np.interp(Tin_1d, Temp, Density)
    Cp_air_1d      = np.interp(Tin_1d, Temp, Cp)
    K_air_1d       = np.interp(Tin_1d, Temp, K)
    KVisc_air_1d   = np.interp(Tin_1d, Temp, KVisc)
    Dvisc_air_1d   = np.interp(Tin_1d, Temp, Dvisc)
    Pr_air_1d      = (KVisc_air_1d * Density_air_1d * Cp_air_1d) / K_air_1d

    #Tube surface 
    #Compute Pr using properties evaluated at surface temperature
    Density_tube_1d = np.interp(Ts_1d, Temp, Density)
    Cp_tube_1d      = np.interp(Ts_1d, Temp, Cp)
    K_tube_1d       = np.interp(Ts_1d, Temp, K)
    KVisc_tube_1d   = np.interp(Ts_1d, Temp, KVisc)
    Dvisc_tube_1d   = np.interp(Ts_1d, Temp, Dvisc)
    Pr_tube_1d      = (KVisc_tube_1d * Density_tube_1d * Cp_tube_1d) / K_tube_1d

    #Return scalars if inputs were scalar; else 1-D arrays
    def scalar_check(arr, is_scalar):
        return float(arr[0]) if is_scalar else arr

    Density_air = scalar_check(Density_air_1d, Tin_is_scalar)
    Cp_air      = scalar_check(Cp_air_1d, Tin_is_scalar)
    K_air       = scalar_check(K_air_1d, Tin_is_scalar)
    KVisc_air   = scalar_check(KVisc_air_1d, Tin_is_scalar)
    Pr_air      = scalar_check(Pr_air_1d, Tin_is_scalar)
    Dvisc_air   = scalar_check(Dvisc_air_1d, Tin_is_scalar)

    Pr_tube     = scalar_check(Pr_tube_1d, Ts_is_scalar)
    Dvisc_tube  = scalar_check(Dvisc_tube_1d, Ts_is_scalar)

    return Density_air, Cp_air, K_air, KVisc_air, Pr_air, Dvisc_air, Pr_tube, Dvisc_tube
