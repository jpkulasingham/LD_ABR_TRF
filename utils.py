import eelbrain as eel
import mne
import numpy as np

def preprocess_eeg(eegfile, refc=['EXG1', 'EXG2'], ch=['Cz'], thr=0.1, verbose=True):
    '''
    load EDF, rereference and find Erg start time
    eegfile: path to .bdf file
    refc: list of reference channels
    '''

    if verbose: print('LOADING', eegfile.stem)
    raw = mne.io.read_raw_bdf(eegfile)
    fs_eeg = raw.info['sfreq']

    erg = raw.get_data(picks=['Erg1'])
    erg_nd = eel.NDVar(erg[0,:], eel.UTS(0, 1/fs_eeg, erg.shape[1]))
    
    Cz = raw.get_data(picks=ch)
    Cz_nd = eel.NDVar(Cz[0,:], eel.UTS(0, 1/fs_eeg, Cz.shape[1]))
    if verbose: print(refc)
    refs = raw.get_data(picks=refc)
    ref_nd = eel.NDVar(refs, (eel.Case, eel.UTS(0, 1/fs_eeg, Cz.shape[1]))).mean('case')

    Cz_reref = Cz_nd - ref_nd

    if verbose: print(erg_nd.max(), erg_nd.min())
    if isinstance(thr, str):
        thr = int(thr)*np.std(erg_nd.x)
    
    erg_start = np.where(erg_nd.abs().x > thr)[0][0] # first peak in erg channel

    if verbose: print(thr)
    if verbose: print('Erg1 start at ', erg_start/fs_eeg)
    
    Cz_a = eel.NDVar(Cz_reref.x[erg_start:], eel.UTS(0, Cz_reref.time.tstep, len(Cz_reref.x[erg_start:])))
    erg_a = eel.NDVar(erg_nd.x[erg_start:], eel.UTS(0, erg_nd.time.tstep, len(erg_nd.x[erg_start:])))

    return Cz_a, erg_a, erg_start



def remove_outliers(y, y1, preds, zerotime=1, stdmul=5, verbose=False):
    '''
    removes outliers by setting 1s before and after high variance segments to zero
    y: eeg NDVar
    y1: filtered eeg NDVar to use for detecting outliers
    preds: list of predictor NDVars
    zerotime: time before and after outlier in seconds
    stdmul: threshold for detecting outliers
    '''
    stdval = y1.std()    
    idxs = np.where(y1.abs().x > stdmul*stdval)[0] # high variance indices
    y2 = y.copy() # for output
    preds2 = [pred.copy() for pred in preds] # for output 
    fs = int(1/y.time.tstep)
    rN = np.zeros_like(y2.x) # zeroed samples
    for i in idxs:
        i1 = np.max([0, i-zerotime*fs]) # 1s before artifact
        i2 = np.min([len(y2.x), i+zerotime*fs]) # 1s after artifact
        rN[i1:i2] = 1
        y2.x[i1:i2] = 0
        for pred2 in preds2:
            pred2.x[i1:i2] = 0
    removed_fraction = 100*np.sum(rN)/len(rN)
    if verbose: print(f'removed {np.sum(rN)} samples, {removed_fraction}%')
    return y2, preds2, removed_fraction


