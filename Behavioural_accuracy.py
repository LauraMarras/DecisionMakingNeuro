from cmath import nan
import pyxdf
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd)

def behavioural_accuracy(PPs = ['p01', 'p02', 'p03', 'p04', 'p05']):
    data_path = 'RawData/'

    ppDF = pd.DataFrame(columns = ['pNr', 'trial nr.', 'stimulus nr.', 'repetition', 'decision', 'accuracy'])
    for pNr, pp in enumerate(PPs):
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(pp))
        if pp == 'p05':
            markers = data[1]['time_series']
        else:
            markers = data[0]['time_series']

    # Create dataframe with all info for each trial
        trials=[(((x[0].replace(',', '')).replace('Sum Trail: ', '')).split(' ')[0:5])+[pNr] for x in markers if 'Sum' in x[0]]
        header_labels = ['trial nr.', 'stimulus nr.', 'repetition', 'decision', 'accuracy', 'pNr']
        trialsDF = pd.DataFrame(trials, columns=header_labels)
        trialsDF = trialsDF.replace(['Correct', 'Incorrect', 'No', 'w', 'l', 'None'], [1, 0, nan, 1, 0, nan])

        trialsDF['stimulus category'] = trialsDF['decision'][trialsDF['accuracy']==1]
        trialsDF['stimulus category'][trialsDF['accuracy']==0] = 1-trialsDF['decision']

    # Add each PP dataframe to general DF
        ppDF = ppDF.append(trialsDF)
    
    return ppDF

if __name__=='__main__':

    PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
    ppDF = behavioural_accuracy(PPs).dropna().astype(int) 

    # Group level statistics: AnovaRM
    ppMeans = ppDF[['pNr', 'repetition', 'accuracy']].groupby(['pNr', 'repetition']).mean().reset_index()

    anovaRes = AnovaRM(data=ppMeans, depvar='accuracy', subject='pNr', within=['repetition']).fit()
    anovaResDF = anovaRes.anova_table
    print(anovaRes)

    # Multiple comparisons TukeyHSD test
    res = pairwise_tukeyhsd(ppDF['accuracy'], ppDF['repetition'])
    np.set_printoptions(suppress=True)
    print(res)