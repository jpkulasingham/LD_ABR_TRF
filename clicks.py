import eelbrain as eel
import mne, scipy
import numpy as np
import utils

def run_clicks_on_eegfile(eegfilename, data_path, out_path, fsds=4096, reg=0, N=-1, refc=['EXG1', 'EXG2'], plotflag=False, notchflag=True,
                       ch=['Cz'], savestr='ERP', force_make=False, stdmul=3, clickorder=None, hp=30, lp=1000, dBstrs=['72dB', '60dB', '48dB', '36dB'], blockidxs=range(5)):
    
    preprocessed_file = out_path / f'{eegfilename[:-4]}_preprocessed_{ch}_{refc[0]}{refc[1]}.pkl'
    print(preprocessed_file)
    if not preprocessed_file.exists() or force_make:
        eegs, ergs, erg_starts = utils.preprocess_eeg(data_path / eegfilename, ch=ch, refc=refc)
        eegs = eel.filter_data(eegs, 1, None)
        ergs = eel.filter_data(ergs, 1, None)
        if notchflag: eegs.x = mne.filter.notch_filter(eegs.x, 1/eegs.time.tstep, [50*i for i in range(1, 20)], notch_widths=5)
        eegs = eel.resample(eegs, fsds)
        ergs = eel.resample(ergs, fsds)

        ergs1 = ergs - ergs.mean()
        ergs1 /= ergs1.std()
        ergps = ergs1.clip(min=0) # positive rectification
        ergns = -(ergs1.clip(max=0)) # negative rectification
        preds = [ergps, ergns]
        predstrs = ['ergpos', 'ergneg']

        yy = eel.filter_data(eegs, 1, 40)
        eegs, preds, rN = utils.remove_outliers(eegs, yy, preds, stdmul=5, verbose=False)

        eel.save.pickle(dict(eegs=eegs, preds=preds, predstrs=predstrs, ergs=ergs), preprocessed_file)
    else:
        preprocdict = eel.load.unpickle(preprocessed_file)
        eegs, preds, ergs, predstrs = preprocdict['eegs'], preprocdict['preds'], preprocdict['ergs'], preprocdict['predstrs']
    
    Ts = get_click_block_times(ergs)
    Ts = [Ts[b] for b in blockidxs]


    res_dict = {}
    eegs1 = []
    ergs1 = []
    if isinstance(Ts, list):
        for i in range(len(Ts)-1):
            aa = eegs.sub(time=(Ts[i], Ts[i+1]))
            aa = eel.NDVar(aa.x, eel.UTS(0, aa.time.tstep, len(aa)))
            eegs1.append(aa)
            aa = ergs.sub(time=(Ts[i], Ts[i+1]))
            aa = eel.NDVar(aa.x, eel.UTS(0, aa.time.tstep, len(aa)))
            ergs1.append(aa)
    else:
        eegs1.append(eegs)
        ergs1.append(ergs)
    
    eegtrials = []
    triggersA = []
    for i in range(len(eegs1)):
        if clickorder is not None:
            idx = clickorder[i]
        else:
            idx = i
        eeg1 = eegs1[idx].copy()
        erg1 = ergs1[idx].copy()
        eeg1 = eel.filter_data(eeg1, hp, lp)
        erp1, triggers1, eegtrials1 = fit_ERP(eeg1, erg1, -0.01, 0.03, N=N, stdmul=stdmul, plotflag=False, verbose=False)
        if len(eegtrials1)!=5279: print(f'ERROR len(eeg1)={len(eegtrials1)}')
        erp1 = erp1.sub(time=(-0.01,0.03))
        erp1.name = 'ERP'
        res_dict[f'erp{i}'] = erp1
        eegtrials.append(eegtrials1)
        triggersA.append(triggers1)
        
    res_dict['preprocessed_file'] = preprocessed_file
    res_dict['Ts'] = Ts
    res_dict['N'] = N
    eel.save.pickle(res_dict, out_path / f'{savestr}.pkl')

    if plotflag:
        plot_ys = []
        for i, k in enumerate(res_dict.keys()):
            if k in ['Ts', 'N', 'preprocessed_file']:
                continue
            tt = res_dict[k].copy()
            tt -= tt.sub(time=(-0.01, 0)).mean('time')
            tt.name = dBstrs[i]
            plot_ys.append(tt)
        p = eel.plot.UTS([plot_ys])
        p.save(out_path / f'{savestr}.png')
        p.close()

        plot_ys = []
        for i, k in enumerate(res_dict.keys()):
            if k in ['Ts', 'N', 'preprocessed_file']:
                continue
            tt = res_dict[k].copy()
            tt -= tt.sub(time=(-0.01, 0)).mean('time')
            tt.name = dBstrs[i]
            plot_ys.append(tt.sub(time=(0, 0.012)))
        p = eel.plot.UTS([plot_ys])
        p.save(out_path / f'{savestr}_zoom0-12ms.png')
        p.close()

    return res_dict, eegtrials, eegs1, ergs1, triggersA, Ts


def get_click_block_times(ergnd, idxs=range(4), blockgap=1):
    fs = 1/ergnd.time.tstep
    ergf = eel.filter_data(ergnd, 1, None)
    triggersA = find_click_triggers(ergf, fs, verbose=False)
    tdiff = np.diff(triggersA)
    blockgap = 1
    blockidxs = np.where(tdiff > blockgap*fs)[0]
    blocktimes = [triggersA[0]/fs-1] + [triggersA[b+1]/fs-1 for b in blockidxs] + [triggersA[-1]/fs+1]
    return blocktimes
    

def find_click_triggers(x, fs, stdmul=3, gap=0.003, posneg=0, verbose=True):
    x -= x.mean()
    if verbose: print(x.min(), x.max(), stdmul*x.std())
    if posneg == 0:
        trigs = np.where(x.abs().x > stdmul*x.std())[0] # find onset indices
    elif posneg == 1:
        trigs = np.where(x.x > stdmul*x.std())[0] # find onset indices
    elif posneg == -1:
        trigs = np.where(x.x < -stdmul*x.std())[0] # find onset indices
    triggers = trigs[:-1][np.diff(trigs) > gap*fs] # remove indices that are too close together
    if verbose: print(len(triggers))
    return triggers


def fit_ERP(y, x, t1=-0.01, t2=0.03, stdmul=5, gap=0.003, N=-1, plotflag=False, verbose=True):
    '''
    extract ERP
    y: eeg NDVar
    x: click onsets NDVar
    t1: tstart ERP
    t2: tend ERP
    stdmul: threshold to detect onsets
    gap: gap between onsets (seconds)
    '''
    fs = 1/y.time.tstep
    triggers = find_click_triggers(x, fs, stdmul, gap, verbose=verbose)
    
    tt1 = x.time.times[triggers[0]]
    if plotflag: eel.plot.UTS(x.sub(time=(tt1-1, tt1+1)))

    # placeholder for ERP
    erp = y.sub(time=(1+t1, 1+t2)).copy()
    erp = eel.NDVar(erp.x, eel.UTS(t1, y.time.tstep, len(erp)))
    erp.x = np.zeros(len(erp.x))
    # compute ERP
    if N!=-1:
        triggers = triggers[:N]
    if verbose: print('num triggers =', len(triggers))
    divN = 0
    eegtrials = []
    for i, trig in enumerate(triggers):
        tt = x.time.times[trig]
        if tt + t2 > y.time.tmax:
            continue
        else:
            divN += 1
        dd =  y.sub(time=(tt+t1, tt+t2)).x
        eegtrials.append(dd[:len(erp)])
        erp.x += dd[:len(erp)]
    erp.x /= divN

    erp = eel.NDVar(erp.x, eel.UTS(t1, erp.time.tstep, len(erp)))

    return erp, triggers, eegtrials