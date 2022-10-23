import pyxdf
import numpy as np
import os
import pandas as pd

# Define function to extract labels
def extract_labels(pNr, markers):
# Extract info for each trial and store in list
    trials=[(((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[0:5])+[pNr] for x in markers if 'Sum' in x[0]]
    header_labels = ['trial_nr', 'stimulus_nr', 'repetition', 'decision', 'accuracy', 'pNr']

# Create dataframe containing all info about each trial: all possible labels
    labelsDF = pd.DataFrame(trials, columns=header_labels)
    labelsDF = labelsDF.replace(['Correct', 'Incorrect', 'No', 'w', 'l', 'None'], [1, 0, np.nan, 1, 0, np.nan])

# Add label about stimulus category
    labelsDF['stimulus_category'] = labelsDF.decision[labelsDF.accuracy == 1]
    labelsDF.stimulus_category[labelsDF.accuracy == 0] = 1-labelsDF.decision

    labelsDF['trial_ind'] = (labelsDF['trial_nr'].astype(int))-1
    labelsDF.set_index('trial_ind', inplace=True)
    labelsDF = labelsDF.astype(np.float).astype('Int64')

# Save dataframe
    labelsDF.to_pickle(out_path + '{}_labels.pkl'.format(pp))
    
if __name__=="__main__":
    PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
    data_path = '/RawData/'
    out_path = '/PreprocessedData/Labels/'
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    
    for pNr, pp in enumerate(PPs):
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(pp))
        if pp == 'p05':
            markers = data[1]['time_series']
        else:
            markers = data[0]['time_series']
       
        extract_labels(pNr, markers)