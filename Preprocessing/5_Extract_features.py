import numpy as np
import os

def windowing(envelope_epoched, window_len, step, sr):
          
    epoch_length = envelope_epoched.shape[1]
    win_len = int(window_len*sr)
    overlap = int(step*sr)
    n_win = int(epoch_length/overlap)

    start = np.arange(0, envelope_epoched.shape[1], overlap)
    stop = [x for x in start+win_len if x <= epoch_length]
    n_win = len(stop)
    start=start[:n_win]

    envelope_windowed = np.zeros((envelope_epoched.shape[0], n_win, win_len, envelope_epoched.shape[2], envelope_epoched.shape[3]), dtype=np.float32)

    for w in range(n_win):
        envelope_windowed[:, w, :, :, :] = envelope_epoched[:, start[w]:stop[w], :, :]

    return envelope_windowed

if __name__=='__main__':
    PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
    reref = 'ESR'
    sr=1024
    window_len = 0.1
    step = 0.05
    short_epoch = 'stimulus'
    long_epoch = 'long_stim'

    for pp in PPs:
        data_path = 'PreprocessedData/Epoching/{}/'.format(pp)
        out_path = 'PreprocessedData/Features/{}/'.format(pp)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        
        # Load epoched envelopes array
        envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_stimulus.npy'.format(pp, reref, short_epoch)) # np.array samples*channels*bands
        # estimate mean along samples axis
        features = envelope_epoched.mean(axis=1)
        # Save data
        np.save(out_path + '{}_{}_features_{}'.format(pp, reref, short_epoch), features)

        
        # Load epoched envelopes array
        envelope_epoched = np.load(data_path + '{}_{}_envelope_epoched_{}.npy'.format(pp, reref, long_epoch)) # np.array samples*channels*bands
        # Windowing
        envelope_windowed = windowing(envelope_epoched, window_len, step, sr)
        # estimate mean along samples axis
        features = envelope_windowed.mean(axis=2)
        # Save data
        np.save(out_path + '{}_{}_features_{}_w{}_st{}'.format(pp, reref, long_epoch, window_len,step), features) 