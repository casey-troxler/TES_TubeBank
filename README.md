# Design Procedure for Tube Banks backfilled with Phase Change Material 

Library for performance modeling of thermal energy storage (TES) using tube bank heat exchangers backfilled with phase change materials (PCM). 

Please see each function for description of functionality, but here is a short description:

**singleRunExample.py** - Barebones code running key function a single time and documenting results in dictionary file. 
**ExamplePlottingCode_aspectRatioStudy.py** - Extended code that runs tubeBankModel through an array of potential aspect ratios and areas for the inlet duct, then plots the results. 
**tubeBankModel.py** - Main function that calls subfunctions and makes small ancillary calculations to complete design process for PCM tube bank. 
**DubovskyModel.py** - Subfunction that calculates heat transfer rate performance of tube bank based on latent heat using analytical model. 
**htc_dp.py** - Subfunction that calculates air-side heat transfer coefficient and pressure drop for aligned tube bank. 
**airProps.py** - Subfunction that calculates air properties based on bulk and tube surface conditions required for calculations. 
