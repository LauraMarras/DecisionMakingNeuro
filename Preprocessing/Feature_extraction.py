import numpy as np
import os



def windowing(envelope_epoched, window_len, step, sr):
          
    epoch_length = envelope_epoched.shape[1] #n samples = 2048
    win_len = int(window_len*sr) # 0.200 * 1024 = 204
    overlap = int(step*sr)  # 0.050 * 1024 = 51
    n_win = int(epoch_length/overlap) # (2048/51)-1 = 40

    start = np.arange(0, envelope_epoched.shape[1], overlap)
    stop = [x for x in start+win_len if x <= epoch_length]
    n_win = len(stop)
    start=start[:n_win]

    envelope_windowed = np.zeros((envelope_epoched.shape[0], n_win, win_len, envelope_epoched.shape[2], envelope_epoched.shape[3]))

    for w in range(n_win):
        envelope_windowed[:, w, :, :, :] = envelope_epoched[:, start[w]:stop[w], :, :]

    return envelope_windowed

if __name__=='__main__':
    PPs = ['kh21', 'kh22', 'kh23', 'kh24','kh25'] # 
    reref = 'ESR'
    sr=1024
    window_len = 0.5
    step = 0.05
    short_epochs = []#['feedback', 'stimulus', 'response', 'baseline']
    long_epochs = ['long_stim'] #'long_FB', 

    for pp in PPs:
        data_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Epoching/{}/'.format(pp)
        out_path = 'C:/Users/laura/Documents/Data_Analysis/Data/PreprocessedData/Features/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for epoch in short_epochs:
        # Load epoched envelopes array
            envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_{}.npy'.format(pp, reref, epoch)) # np.array samples*channels*bands
        # estimate mean along samples axis
            features = envelope_epoched.mean(axis=1)
        # Save data
            np.save(out_path + '{}_{}_features_{}'.format(pp,reref,epoch), features)

        for epoch in long_epochs:
        # Load epoched envelopes array
            envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_{}.npy'.format(pp, reref, epoch)) # np.array samples*channels*bands
        # Windowing
            envelope_windowed = windowing(envelope_epoched, window_len, step, sr)
        # estimate mean along samples axis
            features = envelope_windowed.mean(axis=2)
        # Save data
            np.save(out_path + '{}_{}_features_{}_w{}_st{}'.format(pp,reref,epoch,window_len,step), features) 