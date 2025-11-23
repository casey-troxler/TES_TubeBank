# -*- coding: utf-8 -*-
"""
Tube-bank PCM TES Design Process - Main Function

import with: 
from tube_bank_model import run_tube_bank_model

Function: 
def run_tube_bank_model(PCM_Info, TubeODs, TubeMat, PCM_Sel, Tube_Sel, Tube_OD_index, targetStore_kJ, TubeThickness_mm, ductWidth_m, ductLength_m, flowRate_L_s, Tin_C, Snr, Spr, epsilon, dt_s
)
Returns: Returns the following organized into 4 groups, one of which is the class for the Dubovsky model 
tubeOD_m: tubeOD,
tubeID_m: tubeID,
tubeCols: tubeCols,
tubeRows: tubeRows,
NTubes: NTubes,
Sn_m: Sn,
Sp_m: Sp,
m_PCM_kg: m_PCM,
V_PCM_m3: V_PCM,
C_PCM_$: C_PCM,
Tmelt_C: TmPCM,
tubeMass_kg: tubeM,
tubeCost_$: tubeC,
deviceMass_kg: deviceMass,
deviceCost_$: deviceCost,
MS_V_ms: MS_V_ms,
h_air_W_m2K: float(h_air),
Delta_P_Pa: float(Delta_P_Pa),
dubovsky: - see DubvskyModel function for more information 

Solves tube design process based on a set of minimum inputs predicting key performance variables. 
Including pressure drop, heat transfer rate, weight, volume, and estimated cost. 
This relies on basic first principles, correlations, and an analytical model. 
Only accounts for latent heating, thermal resistance from air convection, tube wall, and forming PCM phase. 
Weight is based on raw tube & PCM materials 
Volume is based on height from tube layout and inlet area 
Cost is based on per weight estimates for material costs. 

Parameters
----------
PCM_Info: pd.DataFrame - Table from PCM_Properties.csv
TubeODs: pd.DataFrame - Table from TubeODs.csv
TubeMat: pd.DataFrame- Table from TubeMat.csv
PCM_Sel: PCM name key (e.g., 'C16H34')
Tube_Sel: (str) - Tube material key (e.g., 'PETG', 'AL', 'CU')
Tube_OD_index: Row index into TubeODs table
targetStore_kJ: Target latent storage [kJ]
TubeThickness_mm: Tube wall thickness [mm]
ductWidth_m: Duct width [m]
ductLength_m: Tube/duct length [m]
flowRate_L_s): Air flow rate [L/s]
Tin_C: Air inlet temp [C]
Snr, Spr: Relative tube spacings (Sn/OD, Sp/OD)
epsilon: Dubovsky epsilon
dt_s: Time step for Dubovsky model

Returns
-------
geometry: 
"tubeOD_m": tubeOD [m]
"tubeID_m": tubeID [m]
"tubeCols": tubeCols 
"tubeRows": tubeRows 
"NTubes": NTubes 
"Sn_m": Sn [m]
"Sp_m": Sp [m]

"pcm": 
"m_PCM_kg": m_PCM [kg]
"V_PCM_m3": V_PC [m3]
"C_PCM_$": C_PCM [C]
"Tmelt_C": TmPCM [C]

"tubes": 
"tubeMass_kg": tubeM [Kg]
"tubeCost_$": tubeC [$]
"deviceMass_kg": deviceMass [kg]
"deviceCost_$": deviceCost [$]

"air": 
"MS_V_ms": MS_V_m [m/s]
"h_air_W_m2K": float(h_air) [W/m2-K]
"Delta_P_Pa": float(Delta_P_Pa) [Pa]

"dubovsky": res,
"""

import math
import pandas as pd
from airProps import airProps
from htc_dp import htc_dp
from DubovskyModel import dubovsky_bank_model


def run_tube_bank_model(
    PCM_Info: pd.DataFrame,
    TubeODs: pd.DataFrame,
    TubeMat: pd.DataFrame,
    PCM_Sel: str = "C16H34",
    Tube_Sel: str = "PETG",
    Tube_OD_index: int = 0,
    targetStore_kJ: float = 24000,
    TubeThickness_mm: float = 0.3,
    ductWidth_m: float = 0.3048,
    ductLength_m: float = 0.6096,
    flowRate_L_s: float = 188.8,
    Tin_C: float = 12.2,
    Snr: float = 3.0,
    Spr: float = 1.05,
    epsilon: float = 0.5,
    dt_s: float = 0.5,
):

    # 1) PCM info, extracted from input spreadsheet
    pcm_row = PCM_Info.loc[PCM_Info["PCM_Name"] == PCM_Sel].iloc[0]

    hsl = float(pcm_row["hsl"])          # kJ/kg
    rhoPCM = float(pcm_row["rho"])      # kg/m³
    costPCM = float(pcm_row["cost"])    # $/kg
    TmPCM = float(pcm_row["Tmelt"])     # °C
    
    # 2) Thermal conductivity selection based on inlet and melting temperature
    if TmPCM > Tin_C:
        K_PCM =  float(pcm_row["k_s"])
    else: 
        K_PCM =  float(pcm_row["k_l"])
    
    # 3) Determine required PCM
    m_PCM = targetStore_kJ / hsl
    V_PCM = m_PCM / rhoPCM
    C_PCM = m_PCM * costPCM

    # 4) Tube material info, extracted from input spreadsheet
    tube_mat_row = TubeMat.loc[TubeMat["Material"] == Tube_Sel].iloc[0]

    rhoTube = float(tube_mat_row["rho"])
    kTube = float(tube_mat_row["k_tm"])
    costTube = float(tube_mat_row["cost"])

    # 4) Tube geometry 
    tubeOD = float(TubeODs.loc[Tube_OD_index, "OD_Tube "]) / 1000.0
    tubeID = tubeOD - 2.0 * (TubeThickness_mm / 1000.0)

    tubeIV = math.pi * (tubeID / 2) ** 2 * ductLength_m
    tubeOV = math.pi * (tubeOD / 2) ** 2 * ductLength_m

    NtubesI = math.ceil(V_PCM / tubeIV)

    Sn = Snr * tubeOD
    Sp = Spr * tubeOD

    tubeCols = math.floor(ductWidth_m / Sn) - 1
    if tubeCols == 0 :
        tubeCols = 1
    tubeRows = math.ceil(NtubesI / tubeCols)

    NTubes = tubeCols * tubeRows

    tubeV_single = tubeOV - tubeIV
    tubeM = tubeV_single * rhoTube * NTubes
    tubeC = tubeM * costTube

    deviceMass = tubeM + m_PCM
    deviceCost = tubeC + C_PCM
    deviceVolume = (Sp*tubeRows + tubeOD)*ductWidth_m*ductLength_m
    
    # 5) Air properties using subfunction 
    Density_air, Cp_air, K_air, KVisc_air, Pr_air, Dvisc_air, Pr_tube, Dvisc_tube = airProps(
        Tin_C, TmPCM
    )

    A_face = ductWidth_m * ductLength_m
    Q_m3s = flowRate_L_s / 1000.0
    MS_V_ms = Q_m3s / A_face

    # 6) Heat transfer coefficient and pressure drop using subfunction 
    h_air, Delta_P_Pa = htc_dp(
        MS_V_ms=MS_V_ms,
        Sn=Sn,
        Sp=Sp,
        OD_Tube_SI=tubeOD,
        KVisc_air=KVisc_air,
        K_air=K_air,
        Density_air=Density_air,
        Pr_air=Pr_air,
        Pr_tube=Pr_tube,
        Tube_rows=tubeRows,
        Dvisc_tube=Dvisc_tube,
        Dvisc_air=Dvisc_air,
    )

    # 7) Solve analytical model using subfunction 
    Q_total_J = NTubes * tubeIV * rhoPCM * hsl * 1000.0

    res = dubovsky_bank_model(
        V_dot_m3_s=Q_m3s,
        Density_air=float(Density_air),
        Cp_air=float(Cp_air),
        Tube_cols=tubeCols,
        Tube_rows=tubeRows,
        OD_Tube_SI=tubeOD,
        ID_Tube_SI=tubeID,
        L_tube_SI=ductLength_m,
        ht_air=float(h_air),
        Epsilon=epsilon,
        K_PCM=K_PCM,
        Q_total_J=Q_total_J,
        T_melt_C=TmPCM,
        MS_Temp_C=Tin_C,
        dt=dt_s,
        h_air_outer=float(h_air),
        k_tm=float(kTube),
    )

    # 8) Return results 
    return {
        "geometry": {
            "tubeOD_m": tubeOD,
            "tubeID_m": tubeID,
            "tubeCols": tubeCols,
            "tubeRows": tubeRows,
            "NTubes": NTubes,
            "Sn_m": Sn,
            "Sp_m": Sp,
        },
        "pcm": {
            "m_PCM_kg": m_PCM,
            "V_PCM_m3": V_PCM,
            "C_PCM_$": C_PCM,
            "Tmelt_C": TmPCM,
        },
        "tubes": {
            "tubeMass_kg": tubeM,
            "tubeCost_$": tubeC,
            "deviceMass_kg": deviceMass,
            "deviceCost_$": deviceCost,
            "deviceVolume_m3": deviceVolume,
        },
        "air": {
            "MS_V_ms": MS_V_ms,
            "h_air_W_m2K": float(h_air),
            "Delta_P_Pa": float(Delta_P_Pa),
        },
        "analytical": res,
    }
