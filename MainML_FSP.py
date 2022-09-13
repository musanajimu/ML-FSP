# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:27:15 2020

@author: musan
"""

import os
import glob
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


import itertools
from itertools import product
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm

plt.close()


data = pd.read_csv("ddSynConditionCalc.csv", index_col=0)


label=['oxy_flow', 'nitro_flow','fuel_flow', 'solv_frac1EHA', 'gas_pressure',
       'Liq_gas_volratio', 
       'percentage_n_fuel_1_in_excess', 'percentage_n_fuel_2_in_excess', 
      'percentage_n_fuel_total_in_excess', 'percentage_n_O2_in_excess', 'percentage_n_CO2_produced', 
      'percentage_n_H2O_produced', 'percentage_n_N2_actual', 'total_mol_after_reaction',
      'Equiv_ratio', 'Tadiabatic', 'DheltaH_rxn', 'Liq_oxid_ratio'
      ]



no_of_clusters=5
clustersizeforssetrial=10
print(len(data))

from Class_ML import Kmeansclustering
KM = Kmeansclustering(data)
KM.kmeansclusteringtrain(no_of_clusters)
KM.kmeansclusteringgenerate()
KM.KM_gen_clusteredcentroid()




'''
plotting
'''

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 2

plotvari1raw = ['oxy_flow', 'nitro_flow','fuel_flow', 'solv_frac1EHA', 'gas_pressure']

plotvarlist1 = list(itertools.combinations(plotvari1raw,2))

ii=1
for plotval in plotvarlist1:
    plotvariab = list(plotval)
    KM.plotclusters(plotvariab[0], plotvariab[1])
    # plt.savefig("Image "+ plotvariab[0] +" vs "+ plotvariab[1] + ".png",format="PNG")
    plt.savefig("plot rawsyn "+str(ii)+".png",format="PNG")
    ii+=1


plotvari2chemE = [ 
            # 'percentage_n_fuel_1_in_excess', 'percentage_n_fuel_2_in_excess', 
      'percentage_n_fuel_total_in_excess', 'percentage_n_O2_in_excess', 'percentage_n_CO2_produced', 
      'percentage_n_H2O_produced', 'percentage_n_N2_actual']
plotvarlist2 = list(itertools.combinations(plotvari2chemE,2))

ii=1
for plotval in plotvarlist2:
    plotvariab = list(plotval)
    KM.plotclusters(plotvariab[0], plotvariab[1])
    # plt.savefig("Image "+ plotvariab[0] +" vs "+ plotvariab[1] + ".png",format="PNG")
    plt.savefig("plot chemE "+str(ii)+".png",format="PNG")
    ii+=1
    
plotvari3tempE = ['nitro_flow', 'solv_frac1EHA', 'Equiv_ratio', 'Tadiabatic', 'DheltaH_rxn', 'Liq_oxid_ratio', 'total_mol_after_reaction']
plotvarlist3 = list(itertools.combinations(plotvari3tempE,2))

ii=1
for plotval in plotvarlist3:
    plotvariab = list(plotval)
    KM.plotclusters(plotvariab[0], plotvariab[1])
    # plt.savefig("Image "+ plotvariab[0] +" vs "+ plotvariab[1] + ".png",format="PNG")
    plt.savefig("plot tempE "+str(ii)+".png",format="PNG")
    ii+=1
    

    
# KM.plotclusters('Tadiabatic[0]', 'n_N2_actual')
# 
# KM.generate_sse_plot(clustersizeforssetrial)
plt.close('all')

KM.generate_sse_plot(10)
plt.savefig("K optimization plot.png")
# =============================================================================
# Training
# =============================================================================



