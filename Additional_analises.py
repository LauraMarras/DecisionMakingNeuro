import numpy as np
from scipy import ndimage, signal
from scipy.stats import norm, kstest, entropy
import pandas as pd
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm
from statsmodels.formula.api import ols


def get_peaks(df, condition):
   n_chans = df.shape[0]
   AUCs = np.vstack(df['AUC_t_{}'.format(condition)].values)
   sign_array = np.vstack(df['significant_t_{}'.format(condition)].values)
   sign_percent = np.sum(sign_array, axis=0)/n_chans*100
   
   peaks = signal.find_peaks(sign_percent, height=35)

   return peaks

def regression(AUC_inf, AUC_arb):
   # Create a DataFrame with your accuracy data
   inf_res = AUC_inf.reshape(AUC_inf.shape[0]*AUC_inf.shape[1])
   arb_res = AUC_arb.reshape(AUC_arb.shape[0]*AUC_arb.shape[1])
   aucs = np.hstack((arb_res, inf_res))
   time = [*range(39)]*AUC_arb.shape[0] + [*range(39)]*AUC_inf.shape[0]
   conds = ['arb' for x in [*range(39)]*AUC_arb.shape[0]] + ['inf' for x in [*range(39)]*AUC_inf.shape[0]]

   # Create a DataFrame with your accuracy data
   data = pd.DataFrame({
      'Time': time,
      'Condition': conds,
      'AUCs': aucs
      })

   # Create the formula for the linear regression model
   formula = 'AUCs ~ Time*Condition'

   # Fit the linear regression model
   model = ols(formula, data=data).fit()

   # Print the summary statistics
   print(model.summary())

   return data, model


if __name__=="__main__":
   out_path = 'AdditionalAnalyses/'
   
   df_path = 'channel_info/'
   df_all = pd.read_parquet(df_path + 'Channel_info_all')
   df_sorted = df_all.sort_values(['area_order', 'loc_short_nohemisphere', 'pNr', 'label', 'hemisphere'], ascending=False, ignore_index=True)

   informed = df_sorted.loc[df_sorted.significant_t_consecutive_informed == True]
   arbitrary = df_sorted.loc[df_sorted.significant_t_consecutive_arbitrary == True]

   infAUCs = np.vstack(informed['AUC_t_informed'].values)
   arbAUCs = np.vstack(arbitrary['AUC_t_arbitrary'].values)

   data, model = regression(infAUCs, arbAUCs)

   peak_inf = get_peaks(informed, 'informed')
   peak_arb = get_peaks(arbitrary, 'arbitrary')