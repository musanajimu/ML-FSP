"""
Created on Thu Dec  3 16:04:26 2020
@author: musan
This code generates data
"""
import random
import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import glob
import scipy as sc
from scipy.signal import find_peaks
import sympy as sp
from itertools import product

liq = [2,4]
gas=[3,5]
# =============================================================================
# #  Synthesis conditions data generation
# =============================================================================

# (random.normalvariate(6,2))
##Generate synthetic synthesis conditions
# liquidflow_mlpmin = [random.uniform(min(liq),max(liq)) for i in range(20)]
# gasflow_lpm = [random.uniform(3,5) for i in range(20)]
# pressure_bar = [random.uniform(1.5,3) for i in range(5)]
# solv_volfracEHA = [random.uniform(0,1) for i in range(4)]

liquidflow_mlpmin = [2+i/50 for i in range(0,101,5)]
gasflow_lpm = [gas[0]+i/(100/(gas[1]-gas[0])) for i in range(0,101,5)]
nitrogen_content_percentage = [i for i in range(0,99,5)]
pressure_bar = [i+0.5 for i in range(1,3,1)]
solv_volfracEHA = [i/10 for i in range(0,11,1)]

# liquidflow_mlpmin = [2,3]
# gasflow_lpm = [3.5,4.5]
# nitrogen_content_percentage = [0]
# pressure_bar = [1.5]
# solv_volfracEHA = [0.5]

syncon = list(product(liquidflow_mlpmin, solv_volfracEHA, gasflow_lpm, nitrogen_content_percentage, pressure_bar))
features_label = ['liquidflow_mlpmin', 'solv_volfracEHA', 'gasflow_lpm','nitrogen_content_percentage','pressure_bar']
features_df = pd.DataFrame(syncon, columns=features_label)
features_df['oxyflow'] = features_df['gasflow_lpm']*(1-features_df['nitrogen_content_percentage']/100)
features_df['nitrogenflow'] = features_df['gasflow_lpm']*(features_df['nitrogen_content_percentage']/100)
features_df.to_csv('PSynCondition.csv')

# print(len(features_df.T))
# plot features
# plt.plot(features_df, 'o--')
# plt.legend()
# plt.show()
# =============================================================================
# # Synthesis cocnditions and features generation
# =============================================================================
##Data generation
from Tadd_solvents_Class import Feature
# Feature(volflow_O2, Volflow_N2, volflow_fuel_total, vol_frac1, pressure)
compd_list_attribute=[]
for i in range(len(features_df)):     #range(len(features_df))
    compdf = Feature(features_df.oxyflow[i], 
                     features_df.nitrogenflow[i],
                     # features_df.nitrogen_content_percentage[i]*features_df.gasflow_lpm[i], 
                     features_df.liquidflow_mlpmin[i], 
                     features_df.solv_volfracEHA[i], 
                     features_df.pressure_bar[i]
                     )

    rawsynconditionf  = compdf.rawsyncon()     #[oxy_flow,nitro_flow, fuel_flow, solv_frac1EHA, gas_pressure]
    sprayatomizationf  = compdf.sprayatom()    #[self.Liq_gas_volratio]
    combustion_chemEnvf = compdf.combustion_chemicalEnv_param()  
    combustion_tempEnvf = compdf.combustion_tempEnv_param() 
    combinedfeature = rawsynconditionf + sprayatomizationf + combustion_chemEnvf + combustion_tempEnvf
    compd_list_attribute.append(combinedfeature)

rawsyn_label = compdf.rawsynconlabel()
sprayatom_label = compdf.sprayatomlabel()
combustion_chemicalEnv_param_label = compdf.combustion_chemicalEnv_paramlabel()
combustion_tempEnv_param_label = compdf.combustion_tempEnv_paramlabel()

featurelistlabel = rawsyn_label + sprayatom_label + combustion_chemicalEnv_param_label + combustion_tempEnv_param_label
print(featurelistlabel)
# label = featurelistlabel
compd_list_attribute = np.array(compd_list_attribute)
label=['oxy_flow', 'nitro_flow','fuel_flow', 'solv_frac1EHA', 'gas_pressure',
       'Liq_gas_volratio', 
       'percentage_n_fuel_1_in_excess', 'percentage_n_fuel_2_in_excess', 
      'percentage_n_fuel_total_in_excess', 'percentage_n_O2_in_excess', 'percentage_n_CO2_produced', 
      'percentage_n_H2O_produced', 'percentage_n_N2_actual', 'total_mol_after_reaction',
      'Equiv_ratio', 'Tadiabatic', 'DheltaH_rxn', 'Liq_oxid_ratio'
      ]
compd_list_attribute_df = pd.DataFrame(compd_list_attribute, columns=label)
compd_list_attribute_df.to_csv('ddSynConditionCalcequic.csv')









