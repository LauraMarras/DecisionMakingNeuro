import pyxdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def RTs(PPs = ['p01', 'p02', 'p03', 'p04', 'p05'], rep=0):
    data_path = 'RawData/'
    
    DF = pd.DataFrame()
    for pNr, pp in enumerate(PPs):
    # Load data
        data, _ = pyxdf.load_xdf(data_path + '{}_test.xdf'.format(pp))
        if pp == 'p05':
            markers = data[1]['time_series']
            time = data[1]['time_stamps']
        else:
            markers = data[0]['time_series']
            time = data[0]['time_stamps']

    # Create dataframe with events and timestamps
        time -= time[0]
        markers_l = [x[0] for x in markers]
        events = pd.DataFrame({'markers':markers_l, 'timestamp':time})
    
    # Get trial number for each event
        indices = events[events.markers.str.contains('Start Trial')].index
        c=0
        tr_n = []
        for x in range(events.shape[0]):
            if x not in indices:
                tr_n.append(c)
            else:
                c+=1
                tr_n.append(c)
        events['trial_n'] = tr_n

    # add relative timing for each trial
        rel_t = np.array([])
        for t in range(len(events.trial_n.unique())):
            if t!=0:
                tr_df = events[events.trial_n == t]
                st_t = tr_df[tr_df.markers.str.contains('Start Stim')].timestamp.values[0]
                rel_t = np.concatenate((rel_t, tr_df.timestamp.to_numpy() - st_t))
            else:
                rel_t = np.concatenate((rel_t, np.zeros(1)))
        
        events['rel_t'] = rel_t

        labelsDF = pd.read_pickle('Labels/{}_labels.pkl'.format(pp))
        events.drop(index=0, inplace=True)
        events['condition'] = events.apply(lambda x: 'arb' if labelsDF[labelsDF['trial_nr'] == x.trial_n].repetition[(x.trial_n)-1] == 1 else 'inf', axis=1)
        events['repetition'] = events.apply(lambda x: labelsDF[labelsDF['trial_nr'] == x.trial_n].repetition[(x.trial_n)-1], axis=1)
        
    # RTs
        incorrect_press = events[((events.markers.str.contains('Press - wrong')) & (events.rel_t >=0) & (events.rel_t <=1.02))]
        RTs = []
        if rep != 0:
            events = events[events.repetition==rep]
        for t in events.trial_n.unique():
            df_t = events[events.trial_n == t]
            if t!=0 and t not in incorrect_press.trial_n.unique():
                rt = df_t[(df_t.markers.str.contains('Press') & (df_t.markers.str.contains('wrong') == False))].rel_t.tolist()[0]
                RTs.append(rt)
                
            elif t in incorrect_press.trial_n.unique():
                rt = df_t[df_t.markers.str.contains('Press - wrong')].rel_t.tolist()[0]
                
                RTs.append(rt)
            RTs = np.array(RTs)

        meanrt = np.mean(RTs) -1
        mean_anticipo = RTs[RTs<1].mean()-1

        incorrect_presses = incorrect_press.rel_t.to_numpy()

    # Create dataframe with all info for each trial
        ppDF = pd.DataFrame({'pNr':pNr, 'pp':pp, 'RTs':[RTs], 'meanRT':meanrt, 'meanAnticipation':mean_anticipo, 'incorrect_presses':[incorrect_presses]})
    # Add each PP dataframe to general DF
        DF = DF.append(ppDF)
    return DF

if __name__=='__main__':

    PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
    DF = RTs(PPs, 0)
    DF1 = RTs(PPs, 1)
    DF2 = RTs(PPs, 2)
    DF3 = RTs(PPs, 3)
    
# Group Analysis
    meanRT_group = DF.meanRT.mean()
    meanant_group = DF.meanAnticipation.mean()