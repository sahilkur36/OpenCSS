## ############################################################### ##
## Developed by:                                                   ## 
##       Cesar A. Pájaro (cesar.pajaromiranda@canterbury.ac.nz)    ##
##       Jesus D. Caballero (caballerojd@uninorte.edu.co)          ##
##       Carlos A. Arteta (carteta@uninorte.edu.co)                ##
## ############################################################### ##

import numpy as np
import pandas as pd

def Jayaram_Baker(Periods_GMPE, Tint):
    Rho_values = []
    n_Per = len(Periods_GMPE)

    # Jayaram-Baker (2008)
    for i in range(n_Per):
        Tmax = np.max([Tint, Periods_GMPE[i]])
        Tmin = np.min([Tint, Periods_GMPE[i]])
        C1 = 1 - np.cos(0.5 * np.pi - 0.366 * np.log(Tmax / np.max([Tmin, 0.109])))
        
        if Tmax < 0.2:
            C2 = 1 - 0.105 * (1 - 1 / (1 + np.exp(100 * Tmax - 5))) * ((Tmax - Tmin) / (Tmax - 0.0099))
        else:
            C2 = 0
        
        if Tmax < 0.109:
            C3 = C2
        else:
            C3 = C1
        
        C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * Tmin / 0.109))
        
        if Tmax < 0.109:
            Rho_value = C2
        elif Tmin > 0.109:
            Rho_value = C1
        elif Tmax < 0.2:
            Rho_value = np.min([C2, C4])
        else:
            Rho_value = C4
        
        Rho_values.append(Rho_value)
    
    return Rho_values

def Macedo_Liu20(Periods_GMPE, Tint):
    Rho_values = []
    n_Per = len(Periods_GMPE)

    for i in range(n_Per):
        Tmax = np.max([Tint, Periods_GMPE[i]])
        Tmin = np.min([Tint, Periods_GMPE[i]])
        C1 = 1 - np.cos(0.5 * np.pi - 0.277 * np.log(Tmax / np.max([Tmin, 0.104])))
        
        if Tmax < 0.2:
            C2 = 1 - 0.662 * (1 - 1 / (1 + np.exp(100 * Tmax - 5))) * ((Tmax - Tmin) / (Tmax - 0.994))
        else:
            C2 = 0
        
        if Tmax < 0.1:
            C3 = C2
        else:
            C3 = C1
        
        C4 = C1 + 0.387 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * Tmin / 0.109))
        
        if Tmax < 0.109:
            Rho_value = C2
        elif Tmin > 0.109:
            Rho_value = C1
        elif Tmax < 0.2:
            Rho_value = np.min([C2, C4])
        else:
            Rho_value = C4
        
        Rho_values.append(Rho_value)
    
    return Rho_values
