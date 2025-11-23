#Import required library plus additional libraries for loop/plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import tubeBankModel Function
from tubeBankModel import run_tube_bank_model


#Import required reference files
#Edit these files to add additional PCMs, Tube Diameters, or Tube Materials
PCM_Info = pd.read_csv("ImportFiles/PCM_Properties.csv")
TubeODs  = pd.read_csv("ImportFiles/TubeODs.csv")
TubeMat  = pd.read_csv("ImportFiles/TubeMat.csv")


#For this set of results, we are interested in the impact of aspect ratio and inlet area 
#So we will sweep through 3 aspect ratios and through a range of inlet areas 
#The length and width of the inlet will be calculated from the target area and 
#selected aspect ratio. 

#Select a tube outer diameter, in this case 25.4 mm 
Tube_OD_index = 4

#Aspect ratios to test: AR = L / W
aspect_ratios = [0.5, 1.5, 2.5]

#Select flow rate, 755 L/s based on 4-ton unit flow rate
flowRate = 755 #L/s

#Inlet areas to test [m^2]
#linear range from 0.05 to 2.55, stepped through at 0.05 m2 increments
inlet_areas_m2 = np.arange(0.05, 2.5 + 0.05, 0.05)

#Predefine list for storing results
results = []

#Two part to interate through inlet area and aspect ratio
for A in inlet_areas_m2:
    for AR in aspect_ratios:
        #Given area A and aspect ratio AR = L/W:
        W = np.sqrt(A / AR)
        L = np.sqrt(A * AR)
        
        #Run tube bank model based on updated inputs
        out = run_tube_bank_model(
            PCM_Info=PCM_Info,
            TubeODs=TubeODs,
            TubeMat=TubeMat,
            PCM_Sel="KF4H2O",
            Tube_Sel="PETG",
            Tube_OD_index=Tube_OD_index,
            targetStore_kJ=303856.056,
            TubeThickness_mm=0.3,
            ductWidth_m=W,
            ductLength_m=L,
            flowRate_L_s=flowRate,
            Tin_C=12.2,
            Snr= 3,
            Spr=1.05,
            epsilon=0.5,
            dt_s=0.5
        )

        # Extract outputs of interest
        q_avg = out["analytical"].q_t_W.mean()
        device_cost = out["tubes"]["deviceCost_$"]
        dP = out["air"]["Delta_P_Pa"]
        volume = out["tubes"]["deviceVolume_m3"]
        COP = q_avg/(2*(flowRate/1000)*dP)
        
        #Store in results 
        results.append({
            "Tube_OD_index": Tube_OD_index,
            "Tube_OD_mm": TubeODs.loc[Tube_OD_index, "OD_Tube "],
            "Area_m2": A,
            "Aspect_Ratio_L_over_W": AR,
            "L_m": L,
            "W_m": W,
            "q_avg_W": q_avg,
            "Delta_P_Pa": dP,
            "Volume_m3": volume,
            "Cost_$": device_cost,
            "COP": COP
        })


#Convert to DataFrame for analysis & plotting
df_results = pd.DataFrame(results)

#Set-up 2x2 subplots to examine several variables of interest
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))
plt.subplots_adjust(wspace = 0.4)

#Preallocate colors and markers to look-up from for loop
markers = {0.5: "o", 1.5: "^", 2.5: "D"}     # circle, triangle, diamond
colors  = {0.5: "#E69F00", 1.5: "#56B4E9", 2.5: "#009E73"} #three color-blind friendly colors

#Iterate for loop through aspect ratios tested - each aspect ratio appears as different symbol/color
for AR in aspect_ratios:
    sub = df_results[df_results["Aspect_Ratio_L_over_W"] == AR] #select subpart of resutls for aspect ratio
    #Plot various items labeled by aspect ratio, with respect to area on x axis. 
    ax1.scatter(sub["Area_m2"], sub["q_avg_W"], marker=markers[AR], color=colors[AR],label=f"AR={AR}")
    ax2.scatter(sub["Area_m2"], sub["Delta_P_Pa"],marker=markers[AR],color=colors[AR],  label=f"AR={AR}")
    ax3.scatter(sub["Area_m2"], sub["Volume_m3"], marker=markers[AR],color=colors[AR], label=f"AR={AR}")
    ax4.scatter(sub["Area_m2"], sub["COP"], marker=markers[AR],color=colors[AR], label=f"AR={AR}")

#Set axis labels and tick parameters
ax1.set_xlabel("A (m$^2$)",fontname='Arial',fontsize=14)
ax1.set_ylabel("$q_{avg}$ (W)",fontname='Arial',fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.set_xlabel("A (m$^2$)",fontname='Arial',fontsize=14)
ax2.set_ylabel("$\Delta$P (Pa)",fontname='Arial',fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_yscale("log", nonpositive='mask')

ax3.set_xlabel("A (m$^2$)",fontname='Arial',fontsize=14)
ax3.set_ylabel("V (m$^3$)",fontname='Arial',fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

ax4.set_xlabel("A (m$^2$)",fontname='Arial',fontsize=14)
ax4.set_ylabel("COP",fontname='Arial',fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_yscale("log", nonpositive='mask')

#Locations for subplot labels
yloc = 1.02
xloc = -0.31

#Placement of subplot labels
ax1.text(xloc, yloc, '(a)', transform=ax1.transAxes, fontsize=16, va='top')
ax2.text(xloc, yloc, '(b)', transform=ax2.transAxes, fontsize=16, va='top')
ax3.text(xloc, yloc, '(c)', transform=ax3.transAxes, fontsize=16, va='top')
ax4.text(xloc, yloc, '(d)', transform=ax4.transAxes, fontsize=16, va='top')

#Limits of each subplot, tune per results 
ax1.axis([0,2.55,3900,5400])
ax2.axis([0,2.55,0.1,5*10**5])
ax3.axis([0,2.55,3.8,11])
ax4.axis([0,2.55,0.01,10**4])

#Include a legend on one plot 
ax1.legend(frameon=False,loc='upper right',ncol=1,fontsize=12)

#Final adjustment and save as pdf
plt.tight_layout()
plt.savefig('aspectRatio.pdf',bbox_inches='tight', dpi=150)

#show plot in previewer
plt.show()

