"""
htc_dp.py

import with:
from htc_dp import htc_dp

Function:
htc_dp(MS_V_ms, Sn, Sp, OD_Tube_SI, KVisc_air, K_air, Density_air, Pr_air, Pr_tube, Tube_rows, Dvisc_tube, Dvisc_air)
Returns:
(h_air, Delta_P)

Calculate heat transfer coefficient (h_air) and pressure drop (Delta_P).

Equations from:
Tube bank Nusselt number correlation - 
Heat and Mass Transfer 
Fundamental & Applications 
Yunus A. Cengel 
Afshin J. Ghajar 
5th Ed. 
ISBN 978-0-07-339818-1

Tube bank pressure drop friction factor - 
Heat Transfer 
J. P. Holman 
10th Ed. 
ISBN 978-0-07-352936-2

   
Parameters
----------
MS_V_ms: Face velocity [m/s].
Sn: Row spacing (frontal pitch) [m].
Sp: Transverse spacing [m].
OD_Tube_SI: Tube outer diameter [m].
KVisc_air: Kinematic viscosity [m^2/s].
K_air: Thermal conductivity [W/mK].
Density_air: Density [kg/m^3].
Pr_air: Prandtl number at bulk temperature (dimensionless).
Pr_tube: Prandtl number at tube surface temperature (dimensionless).
Tube_rows: Number of transverse rows (integer-like).
Dvisc_tube: Dynamic viscosity evaluated at tube surface [kg/(m·s)].
Dvisc_air: Dynamic viscosity evaluated at bulk air temperature [kg/(m·s)].

Returns
-------
h_air: Heat transfer coefficient [W/m^2/K].
Delta_P: Pressure drop [in.H2O].
"""

from typing import Tuple, Union
import numpy as np

ArrayLike = Union[float, int, np.ndarray, list, tuple]

def htc_dp(MS_V_ms: ArrayLike,
           Sn: ArrayLike,
           Sp: ArrayLike,
           OD_Tube_SI: ArrayLike,
           KVisc_air: ArrayLike,
           K_air: ArrayLike,
           Density_air: ArrayLike,
           Pr_air: ArrayLike,
           Pr_tube: ArrayLike,
           Tube_rows: ArrayLike,
           Dvisc_tube: ArrayLike,
           Dvisc_air: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:

    # Convert inputs to numpy arrays for broadcasting
    MS_V_ms = np.asarray(MS_V_ms, dtype=float)
    Sn = np.asarray(Sn, dtype=float)
    Sp = np.asarray(Sp, dtype=float)
    OD_Tube_SI = np.asarray(OD_Tube_SI, dtype=float)
    KVisc_air = np.asarray(KVisc_air, dtype=float)
    K_air = np.asarray(K_air, dtype=float)
    Density_air = np.asarray(Density_air, dtype=float)
    Pr_air = np.asarray(Pr_air, dtype=float)
    Pr_tube = np.asarray(Pr_tube, dtype=float)
    Tube_rows = np.asarray(Tube_rows, dtype=float)
    Dvisc_tube = np.asarray(Dvisc_tube, dtype=float)
    Dvisc_air = np.asarray(Dvisc_air, dtype=float)

    # 1) Vmax based on minimum frontal area
    Vmax = MS_V_ms * (Sn / (Sn - OD_Tube_SI))

    # 2) Reynolds number based on outer diameter and min frontal area
    Re = (Vmax * OD_Tube_SI) / KVisc_air
    # Initialize/preallocate Nu with same shape as Re
    Nu = np.zeros_like(Re, dtype=float)

    # 3) Piecewise Nusselt correlations (aligned tubes)
    # mask used to compute potential array and filter into appropraite Nusselt 
    # equation based on Reynolds number range. 
    
    # Re <= 100
    mask = (Re <= 100)
    Nu[mask] = 0.9 * (Re[mask] ** 0.4) * (Pr_air[mask] ** 0.36) * ((Pr_air[mask] / Pr_tube[mask]) ** 0.25)

    # 100 < Re <= 1000
    mask = (Re > 100) & (Re <= 1000)
    Nu[mask] = 0.52 * (Re[mask] ** 0.5) * (Pr_air[mask] ** 0.36) * ((Pr_air[mask] / Pr_tube[mask]) ** 0.25)

    # 1000 < Re <= 2e5
    mask = (Re > 1000) & (Re <= 2e5)
    Nu[mask] = 0.27 * (Re[mask] ** 0.63) * (Pr_air[mask] ** 0.36) * ((Pr_air[mask] / Pr_tube[mask]) ** 0.25)

    # 2e5 < Re <= 2e6
    mask = (Re > 2e5) & (Re <= 2e6)
    Nu[mask] = 0.033 * (Re[mask] ** 0.8) * (Pr_air[mask] ** 0.4) * ((Pr_air[mask] / Pr_tube[mask]) ** 0.25)

    # 4) Heat transfer coefficient from nussel tnumber
    h_air = Nu * K_air / OD_Tube_SI

    # 5) Mass velocity at minimum flow area [kg/m^2/s] - for pressure drop 
    G_max = Density_air * Vmax

    # 6) Friction factor for aligned tubes
    # Broken up into terms to simplify readability
    term1 = 0.044
    term2_num = 0.08 * Sp / OD_Tube_SI
    term2_den = (( (Sn - OD_Tube_SI) / OD_Tube_SI ) ** 0.43) + ((1.13 * OD_Tube_SI) / Sp)
    FF = (term1 + term2_num / term2_den) * (Re ** -0.15)

    # 7) Use friction factor to find Pressure drop [Pa]
    Delta_P_pa = ((2.0 * FF * (G_max ** 2) * Tube_rows) / Density_air) * ((Dvisc_tube / Dvisc_air) ** 0.14)

    # Ensure outputs are numpy arrays
    h_air = np.asarray(h_air)
    Delta_P = np.asarray(Delta_P_pa)

    return h_air, Delta_P