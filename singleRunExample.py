# import required libraries for single run
import pandas as pd

# import tubeBankModel Function
from tubeBankModel import run_tube_bank_model


# Import required reference files
# Edit these files to add additional PCMs, Tube Diameters, or Tube Materials
PCM_Info = pd.read_csv("ImportFiles/PCM_Properties.csv")
TubeODs  = pd.read_csv("ImportFiles/TubeODs.csv")
TubeMat  = pd.read_csv("ImportFiles/TubeMat.csv")

#Populate require inputs; 3 spreadsheets and then single values for each parameter 
#See tubeBankModel for more details on each variable. 
out = run_tube_bank_model(
    PCM_Info=PCM_Info,
    TubeODs=TubeODs,
    TubeMat=TubeMat,
    PCM_Sel="KF4H2O",
    Tube_Sel="PETG",
    Tube_OD_index=4,
    targetStore_kJ=10000,
    TubeThickness_mm=0.3,
    ductWidth_m=2,
    ductLength_m=1,
    flowRate_L_s=188,
    Tin_C=12.2,
    Snr= 3,
    Spr=1.05,
    epsilon=0.5,
    dt_s=0.5
)
#Examine out data type to see outputs for particular run