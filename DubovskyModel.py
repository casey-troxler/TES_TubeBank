"""
DubovskyModel.py 

import with: 
from DubovskyModel import dubovsky_bank_model

Function: 
dubovsky_bank_model(V_dot_m3_s,Density_air,Cp_air,Tube_cols,Tube_rows,OD_Tube_SI,ID_Tube_SI,L_tube_SI,ht_air,Epsilon,K_PCM,Q_total_J,T_melt_C,MS_Temp_C,dt,h_air_outer,k_tm)
Returns
class AnalyticalResult:
t_s: np.ndarray
t_h: np.ndarray
Q_t_J: np.ndarray
q_t_W: np.ndarray
M_dot_total_kg_s: float


Finds transient performance of tube bank based on latent heat storage. 
Based on analytical model described in: 
Dubovsky, V., Ziskind, G., Letan, R., 2011. 
Analytical model of a PCM-air heat exchanger.
Applied Thermal Engineering 31, 3453–3462. 
doi:10.1016/j.applthermaleng.2011.06.
031.

Parameters
----------
V_dot_m3_s: Total volumetric flow [m^3/s]
Density_air: Air density [kg/m^3]
Cp_air: Air specific heat [J/kg-K]
Tube_cols: Columns across the bank
Tube_rows: Rows in flow direction (N)
OD_Tube_SI: Tube outer diameter [m]
ID_Tube_SI: Tube inner diameter [m]
L_tube_SI: Tube length [m]
ht_air: air-side h  [W/m^2-K]
Epsilon: air-temperature correction factor
K_PCM: conductivity for ID/(4*K) term [W/m-K]
Q_total_J: Total energy storage (m_pcm * hsl) [J]
T_melt_C: float, Melting temperature [C]
MS_Temp_C: Inlet Temperature [C] 
dt: dimensionless step
h_air_outer: outer-surface h_air [W/m^2-K]
k_tm: tube-wall thermal conductivity [W/m-K]

Returns
-------
Returned as custom class with following; 
class AnalyticalResult:
t_s: time vector [s]
t_h: time vector [h]
Q_t_J: energy stored [J]
q_t_W: heat transfer rate [W]

"""

from dataclasses import dataclass
import numpy as np
import math
from scipy.optimize import brentq

@dataclass
class AnalyticalResult:
    t_s: np.ndarray
    t_h: np.ndarray
    Q_t_J: np.ndarray
    q_t_W: np.ndarray

def dubovsky_bank_model(
    V_dot_m3_s: float,    # total volumetric flow [m^3/s]
    Density_air: float,   # [kg/m^3]
    Cp_air: float,        # [J/kg-K]
    Tube_cols: int,       # columns across the bank
    Tube_rows: int,       # rows in flow direction (N)
    OD_Tube_SI: float,    # [m]
    ID_Tube_SI: float,    # [m]
    L_tube_SI: float,     # [m]
    ht_air: float,        # air-side h (used if no override) [W/m^2-K]
    Epsilon: float,       # air-temperature correction factor
    K_PCM: float,   # conductivity for ID/(4*K) term [W/m-K]
    Q_total_J: float,     # energy scale (e.g., m_pcm * hsl) [J]
    T_melt_C: float,      # C
    MS_Temp_C: float,     # C
    h_air_outer: float,  # outer-surface h_air [W/m^2-K]
    k_tm: float,       # tube-wall thermal conductivity [W/m-K]
    dt: float = 0.005    # dimensionless step
) -> AnalyticalResult:

    # 1) Mass flow (total, then per column)
    M_dot_total = V_dot_m3_s * Density_air               # [kg/s]
    Tube_cols = int(Tube_cols)
    M_dot_col  = M_dot_total / Tube_cols                 # [kg/s per column]

    # 2) Single-tube external area: A1 = 2pi*r(L + r)
    r_o = 0.5 * OD_Tube_SI
    A1  = 2.0 * math.pi * r_o * (L_tube_SI + r_o)        # [m^2]
    # 3) hf
    hf  = (M_dot_col * Cp_air) / A1                      # [W/m^2-K]
    # tube wall resistance & effective inner-surface h ----
    rt = (OD_Tube_SI / (2.0 * k_tm)) * math.log(OD_Tube_SI / ID_Tube_SI)  # [m^2·K/W]
    ht_for_ho = 1.0 / ( (ID_Tube_SI / (OD_Tube_SI * h_air_outer)) + rt )     # [W/m^2·K]

    # 4) Overall ho
    ho  = 1.0 / ( (1.0/ht_for_ho) + (Epsilon/hf) + (ID_Tube_SI/(4.0*K_PCM)) )
    
    # 5) P and solve b from (1 - exp(-b))/b = 1 - P
    P   = ho * (ID_Tube_SI/(4.0*K_PCM))
    target = 1.0 - P
    f = lambda b: (1.0 - np.exp(-b)) / b - target
    b = float(brentq(f, 1e-9, 1e3, xtol=1e-12, maxiter=200))

    # 6) N and tau_knot
    N = Tube_rows
    tau_knot = 1.0 + (ho/hf) * (N - 1.0)

    # 7) Xi and tau grid
    Xi  = ho / (hf * (1.0 - math.exp(-b)))
    tau = np.arange(0.0, tau_knot + 1e-12, dt)

    # 8) Qt(τ) (dimensionless), piecewise
    Qt_dim = np.zeros_like(tau)
    exp_neg_b = math.exp(-b)
    one_minus_exp_neg_b = 1.0 - exp_neg_b
    C1 = 1.0 / one_minus_exp_neg_b
    C2 = (1.0 / (b * N)) * (hf / ho)

    for i, t in enumerate(tau):
        if t <= 1.0:
            Qt_dim[i] = C1 - C2 * math.log(1.0 + math.exp(-b*t) * (math.exp(b*Xi*N) - 1.0))
        else:
            theta = 1.0 + exp_neg_b * (math.exp(b*Xi*(N - (t - 1.0)*(hf/ho))) - 1.0)
            Qt_dim[i] = C1 - (exp_neg_b / one_minus_exp_neg_b) * (hf/ho) * ((t - 1.0)/N) - C2 * math.log(theta)

    # 9) Scale energy and time
    Q_t = Qt_dim * Q_total_J
    
    dT_scale = (T_melt_C - MS_Temp_C)
    t1_s = (Q_total_J/(Tube_cols*Tube_rows)) / (A1 * dT_scale * ho)
    t_s  = tau * t1_s
   
    # 10) Convert change in energy storage to power q(t) = dQ/dt
    q_t = np.gradient(Q_t, t_s, edge_order=2)
    t_h  = t_s / 3600.0
    
    return AnalyticalResult(
        t_s=t_s,
        t_h=t_h,
        Q_t_J=Q_t,
        q_t_W=q_t,
    )
