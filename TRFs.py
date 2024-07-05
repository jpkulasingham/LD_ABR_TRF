import eelbrain as eel
import numpy as np
import mne, pathlib, scipy, importlib, statistics, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathnames import *

importlib.reload(utils)

def extract_channel(eegfile, ch=['Cz'], refc=['EXG1', 'EXG2'], remove_segments=None):
    raw = mne.io.read_raw_bdf(eegfile)
    fs_eeg = raw.info['sfreq']

    erg = raw.get_data(picks=['Erg1'])
    erg_nd = eel.NDVar(erg[0], eel.UTS(0, 1/fs_eeg, len(erg[0])))
    
    eeg = raw.get_data(picks=ch)
    eeg_nd = eel.NDVar(eeg[0,:], eel.UTS(0, 1/fs_eeg, eeg.shape[1]))

    refs = raw.get_data(picks=refc)
    ref_nd = eel.NDVar(refs, (eel.Case, eel.UTS(0, 1/fs_eeg, eeg.shape[1]))).mean('case')

    eeg_reref = eeg_nd - ref_nd

    if remove_segments is not None:
        eeg2 = [eeg_reref.sub(time=(0, remove_segments[0][0]))]
        erg2 = [erg_nd.sub(time=(0, remove_segments[0][0]))]
        for ir in range(len(remove_segments)):
            if ir == len(remove_segments)-1:
                eeg2.append(eeg_reref.sub(time=(remove_segments[ir][1], eeg_reref.time.tmax)))
                erg2.append(erg_nd.sub(time=(remove_segments[ir][1], erg_nd.time.tmax)))
            else:
                eeg2.append(eeg_reref.sub(time=(remove_segments[ir][1], remove_segments[ir+1][0])))
                erg2.append(erg_nd.sub(time=(remove_segments[ir][1], remove_segments[ir+1][0])))
        eeg_reref = eel.concatenate(eeg2)
        erg_nd = eel.concatenate(erg2)

    return eeg_reref, erg_nd


def extract_channel_speechfiles(savestr, eegfilenames, out_path, data_path, force_make=False, ch='Cz', refc=['EXG1', 'EXG2'], remove_segments=None, saveflag=False):
    print(eegfilenames)
    preprocessed_file = out_path / f'{savestr}_preprocessed_{ch}_{refc[0]}{refc[1]}.pkl'
    if not preprocessed_file.exists() or force_make:
        eegsA = []
        ergsA = []
        erg_startsA = []
        for ifile, eegfilename in enumerate(eegfilenames):
            if remove_segments is not None:
                remove_segments1 = remove_segments[ifile]
            else:
                remove_segments1 = None
            eegs, ergs = extract_channel(data_path / eegfilename, ch=ch, refc=refc, remove_segments=remove_segments1)
            eegsA.append(eegs)
            ergsA.append(ergs)
        datadict = dict(eegs=eegsA, ergs=ergsA)
        if saveflag:
            eel.save.pickle(datadict, preprocessed_file)
    else:
        datadict = eel.load.unpickle(preprocessed_file)
    return datadict      



def preprocess_multitrials(eeg, erg, t1=2, t2=62, N=None, manual_trigger_val=None, manual_triggers=None, plotfolder=None, plot_savestr='speech', plotflag=False, 
                        stdmul=20, fsds=4096, notchflag=True):
    '''
    load EDF, rereference and find Erg start time
    eegfile: path to .bdf file
    refc: list of reference channels
    '''
    
    fs_eeg = 1/eeg.time.tstep
    n1 = int(t1*fs_eeg)
    n2 = int(t2*fs_eeg)

    erg_ndf = eel.filter_data(erg, 1, None)
    erg_ndf -= erg_ndf.sub(time=(0,2)).mean()
    ergstd = erg_ndf.sub(time=(0, 2)).std()

    idxs = np.where(erg_ndf.x>stdmul*ergstd)[0]
    idxs = [i for i in idxs if i>3*fs_eeg]
    erg_starts = [idxs[0]]
    for idx in idxs:
        if idx - erg_starts[-1] > (t2+2)*fs_eeg:
            erg_starts.append(idx)

    print('erg_starts', len(erg_starts))
    if N is not None:
        erg_starts = erg_starts[:N]

    print(f'N trials = {len(erg_starts)}')
    print(eeg.time.tmax)


    if plotflag:
        plot_ys = []
        plt.figure(figsize=(10, 20))
        for i, erg_start in enumerate(erg_starts):
            xx = erg.sub(time=(erg_start/fs_eeg-0.2, erg_start/fs_eeg+0.2))
            plt.subplot(len(erg_starts), 1, i+1)
            plt.plot(xx.time.times, xx.x)
            plt.title(f'{plot_savestr} {i} {erg_start/fs_eeg}')
            plt.axvline(erg_start/fs_eeg)
        if plotfolder is not None:
            plotfile = plotfolder / f'{plot_savestr}_ergstarts.png'
            plt.savefig(plotfile)
            plt.close()

    if notchflag:
        eeg.x = mne.filter.notch_filter(eeg.x, 1/eeg.time.tstep, [50*i for i in range(1, 20)], notch_widths=5)

    print('filtering eeg')
    eeg = eel.filter_data(eeg, 1, None)

    eegs = []
    ergs = []
    for i, erg_start in enumerate(erg_starts):
        # print(i, erg_start, n2-n1)
        eegs.append(eel.NDVar(eeg.x[erg_start+n1:erg_start+n2], eel.UTS(0, eeg.time.tstep, n2-n1)))
        ergs.append(eel.NDVar(erg.x[erg_start+n1:erg_start+n2], eel.UTS(0, erg.time.tstep, n2-n1)))

    if fsds:
        print('downsampling', fsds)
        eegs = [eel.resample(eeg, fsds) for eeg in eegs]
        ergs = [eel.resample(erg, fsds) for erg in ergs]

    return eel.combine(eegs), eel.combine(ergs), erg_starts





def preprocess_channel(savestr, savefile, eegfiles, data_path, subject, out_path, ch='Cz', refc=['EXG1', 'EXG2'], remove_segments=None, 
                        force_make=False, force_make_raw=False, save_raw_flag=False, t1=0, t2=62, N=None, plotfolder=None, temp_path=None, plotflag=False, 
                        stdmul=20, fsds=4096, notchflag=True):
    savefile = out_path / f'{savefile}_{fsds}_{t1}-{t2}.pkl'
    if not savefile.exists() or force_make:
        if temp_path is not None:
            preprocessed_file = temp_path / f'{savestr}_preprocessed_{ch}_{refc[0]}{refc[1]}.pkl'
        else:
            preprocessed_file = out_path / f'{savestr}_preprocessed_{ch}_{refc[0]}{refc[1]}.pkl'
        if not preprocessed_file.exists() or force_make_raw:
            datadict_in = extract_channel_speechfiles(savestr, eegfiles, out_path, data_path / subject, saveflag=save_raw_flag,
                            force_make=force_make_raw, ch=ch, refc=refc, remove_segments=remove_segments)
        else:
            datadict_in = eel.load.unpickle(preprocessed_file)
        Neeg = len(datadict_in['eegs'])
        eegsA = []
        ergsA = []
        erg_startsA = []
        for i in range(Neeg):
            eeg = datadict_in['eegs'][i]
            erg = datadict_in['ergs'][i]
            eegs, ergs, erg_starts = preprocess_multitrials(eeg, erg, t1=t1, t2=t2, N=N, plotfolder=out_path, plot_savestr=savefile.stem+f'_eeg{i}', plotflag=plotflag, 
                        stdmul=stdmul, fsds=fsds, notchflag=notchflag)
            eegsA.append(eegs)
            ergsA.append(ergs)
            erg_startsA.append(erg_starts)
        datadict = dict(eegs=eegsA, ergs=ergsA, erg_starts=erg_startsA)
        eel.save.pickle(datadict, savefile)
    else:
        datadict = eel.load.unpickle(savefile)
    return datadict



def get_preds(preds, ks=['rect', 'gt', 'oss', 'ossa', 'zil'], predspath=None):
    if predspath.exists():
        return eel.load.unpickle(predictor_path / 'preds_aligned.pkl')
    else:
        filenames = [f for f in wav_path.glob('M*.wav')]
        print(filenames[0])
        fs = 44100

        if 'oss' in ks:
            print('oss')
            predpath = oss_path
            preds['ossp'] = []
            preds['ossn'] = []
            for i in tqdm(range(80)):
                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_osses_no_adapt_no_headphone_pos.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['ossp'].append(nd)

                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_osses_no_adapt_no_headphone_neg.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['ossn'].append(nd)

            predpath = oss_switching_path
            preds['ossswitchp'] = []
            preds['ossswitchn'] = []
            for i in tqdm(range(80)):
                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_osses_no_adapt_no_headphone_pos.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['ossswitchp'].append(nd)

                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_osses_no_adapt_no_headphone_neg.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['ossswitchn'].append(nd)

            preds = align_pred(preds, 'oss', levdep=False)

        if 'zil' in ks:
            print('zil')
            dd = eel.load.unpickle(zil_path / 'all_4096Hz.pkl')
            for k in dd.keys():
                preds[k] = dd[k]

            dd = eel.load.unpickle(zil_switch_path / 'all_4096Hz.pkl')
            preds['zilswitchp'] = dd['zilpswitching']
            preds['zilswitchn'] = dd['zilnswitching']

            preds = align_pred(preds, 'zil', levdep=True)


        if 'ossa' in ks:
            print('ossa')
            dd = eel.load.unpickle(ossa_path / 'all_4096Hz.pkl')
            for k in dd.keys():
                preds[k] = dd[k]

            dd = eel.load.unpickle(ossa_switch_path / 'all_4096Hz.pkl')
            preds['ossaswitchp'] = dd['ossapswitching']
            preds['ossaswitchn'] = dd['ossanswitching']

            preds = align_pred(preds, 'ossa', levdep=True)

        if 'gt' in ks:
            print('gt')
            preds['gtp'] = []
            preds['gtn'] = []
            predpath = pathlib.Path(gt_path / 'pred001_gt')
            for i in tqdm(range(80)):
                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_gt_pos_72.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['gtp'].append(nd)

                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_gt_neg_72.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['gtn'].append(nd)

            preds['gtswitchp'] = []
            preds['gtswitchn'] = []
            predpath = pathlib.Path(gt_switch_path / 'pred001_gt_switching2')
            for i in tqdm(range(80)):
                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_gt_switching_pos_72.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['gtswitchp'].append(nd)

                mat = scipy.io.loadmat(str(predpath / f'{filenames[i].stem}_gt_switching_neg_72.mat'))
                ndx = mat['outsig'][0,:]
                nd = eel.NDVar(ndx, eel.UTS(0, 1/fs, len(ndx)))
                preds['gtswitchn'].append(nd)

            preds = align_pred(preds, 'gt', levdep=False)


        if 'rect' in ks:
            print('rect')
            preds['rectp'] = []
            preds['rectn'] = []
            preds['wav'] = []
            preds['wav_trigger'] = []
            for i in tqdm(range(80)):
                fs_wav, wav = scipy.io.wavfile.read(filenames[i])
                wavnd = eel.NDVar(wav[:,2], eel.UTS(0, 1/fs_wav, wav.shape[0]))
                preds['wav_trigger'].append(wavnd)
                wavnd = eel.NDVar(wav[:,1], eel.UTS(0, 1/fs_wav, wav.shape[0]))
                preds['wav'].append(wavnd)
                preds['rectp'].append(wavnd.clip(min=0))
                preds['rectn'].append(-(wavnd.clip(max=0)))

            filenames = [f for f in wav_switch_path.glob('M*.wav')]
            fs = 44100
            preds['rectswitchp'] = []
            preds['rectswitchn'] = []
            for i in tqdm(range(80)):
                fs_wav, wav = scipy.io.wavfile.read(filenames[i])
                wavnd = eel.NDVar(wav[:,2], eel.UTS(0, 1/fs_wav, wav.shape[0]))
                wavnd = eel.NDVar(wav[:,1], eel.UTS(0, 1/fs_wav, wav.shape[0]))
                preds['rectswitchp'].append(wavnd.clip(min=0))
                preds['rectswitchn'].append(-(wavnd.clip(max=0)))

            preds = align_pred(preds, 'rect', levdep=False)
        
        return preds





def align_pred(preds_a, k, levdep):
    if not levdep:
        print('aligning non-levdep', k)
        for s in ['p', 'n']:
            kk = k+s
            print(kk)
            preds_a[kk] = eel.combine(preds_a[kk])
            preds_a[kk] = eel.combine([eel.resample(preds_a[kk][i], 2048) for i in range(len(preds_a[kk]))])
            preds_a[kk] = eel.resample(eel.NDVar(preds_a[kk], (eel.Case, eel.UTS(-0.01, (60/60.002)/2048, len(preds_a[kk].time)))), 2048)
            preds_a[kk] = preds_a[kk].sub(time=(0, 60))

            kk = k+'switch'+s
            print(kk)
            preds_a[kk] = eel.combine(preds_a[kk])
            preds_a[kk] = eel.combine([eel.resample(preds_a[kk][i], 2048) for i in range(len(preds_a[kk]))])
            preds_a[kk] = eel.resample(eel.NDVar(preds_a[kk], (eel.Case, eel.UTS(-0.01, (60/60.002)/2048, len(preds_a[kk].time)))), 2048)
            preds_a[kk] = preds_a[kk].sub(time=(0, 60))

    else:
        print('aligning levdep', k)
        for s in ['p', 'n']:
            for ilev in range(4):
                kk = k+s+str(ilev)
                print(kk)
                preds_a[kk] = eel.resample(eel.combine(preds_a[kk]), 2048)
                preds_a[kk] = eel.resample(eel.NDVar(preds_a[kk], (eel.Case, eel.UTS(-0.01, (60/60.002)/2048, len(preds_a[kk].time)))), 2048)
                preds_a[kk] = preds_a[kk].sub(time=(0, 60))
            kk = k+'switch'+s
            print(kk)
            preds_a[kk] = eel.resample(eel.combine(preds_a[kk]), 2048)
            preds_a[kk] = eel.resample(eel.NDVar(preds_a[kk], (eel.Case, eel.UTS(-0.01, (60/60.002)/2048, len(preds_a[kk].time)))), 2048)
            preds_a[kk] = preds_a[kk].sub(time=(0, 60))

    return preds_a




def get_pred_shifts(preds, ks=None):
    
    if ks is None:
        ks = ['rectp', 'gtp', 'ossp', 'ossap0',  'ossap1',  'ossap2',  'ossap3', 'zilp0', 'zilp1', 'zilp2', 'zilp3']

    pred_corr_vals_mean = {}
    pred_corr_vals_std = {}
    pred_corr_lats_mean = {}
    pred_corr_lats_mode = {}
    pred_corr_lats_median = {}
    pred_corr_lats_std = {}
    pred_corr_vals_all = {}
    pred_corr_lats_all = {}
    preds_aligned = {}

    shifts = {}

    for k in ks:
        x1 = preds[k].copy()
        x2 = preds['rectp'].copy()
        corrvals1 = []
        corrlats1 = []
        fs1 = int(1/x1.time.tstep)
        fs2 = int(1/x2.time.tstep)
        print(fs1, fs2)
        N = len(x1[0].x)
        correlation_lags = scipy.signal.correlation_lags(N, N)
        for i in tqdm(range(len(x1))):
            corrsig = scipy.signal.correlate(x1[i].x, x2[i].x)
            corrval = np.max(corrsig)
            corrlat = correlation_lags[np.argmax(corrsig)]
            corrvals1.append(corrval)
            corrlats1.append(corrlat*preds['rectp'][i].time.tstep)
        pred_corr_vals_mean[k] = np.mean(corrvals1)
        pred_corr_vals_std[k] = np.std(corrvals1)
        pred_corr_lats_mean[k] = np.mean(corrlats1)
        pred_corr_lats_mode[k] = max(set(corrlats1), key=corrlats1.count)
        pred_corr_lats_median[k] = statistics.median(corrlats1)
        pred_corr_lats_std[k] = np.std(corrlats1)
        pred_corr_vals_all[k] = corrvals1
        pred_corr_lats_all[k] = corrlats1
        print(f'{k} corr val = {pred_corr_vals_mean[k]:.4f} +- {pred_corr_vals_std[k]:.4f}, corr lat mean={pred_corr_lats_mean[k]*1000:.2f}, median={pred_corr_lats_median[k]*1000:.2f} +- {pred_corr_lats_std[k]*1000:.4f}')
        print(f'{k} corr lat avg = {pred_corr_lats_mode[k]}')

        x1old = x1.copy()
        corrlats1old = corrlats1.copy()
        
        shift = pred_corr_lats_median[k]
        shifts[k] = shift
        corrvals1 = []
        corrlats1 = []
        x1 = preds[k].copy()
        x1 = eel.NDVar(x1.x, (eel.Case, eel.UTS(x1.time.tmin-shift, x1.time.tstep, len(x1.x[0,:]))))
        x1 = x1.sub(time=(1, x1.time.tmax))
        x2 = preds['rectp'].sub(time=(1, x1.time.tmax)).copy()
        correlation_lags = scipy.signal.correlation_lags(len(x1[0]), len(x2[0]))
        for i in tqdm(range(len(x1))):
            corrsig = scipy.signal.correlate(x1[i].x, x2[i].x)
            corrval = np.max(corrsig)
            corrlat = correlation_lags[np.argmax(corrsig)]
            corrvals1.append(corrval)
            corrlats1.append(corrlat*preds['rectp'][i].time.tstep)
        pred_corr_vals_mean[k] = np.mean(corrvals1)
        pred_corr_vals_std[k] = np.std(corrvals1)
        pred_corr_lats_mean[k] = np.mean(corrlats1)
        pred_corr_lats_mode[k] = max(set(corrlats1), key=corrlats1.count)
        pred_corr_lats_median[k] = statistics.median(corrlats1)
        pred_corr_lats_std[k] = np.std(corrlats1)
        pred_corr_vals_all[k] = corrvals1
        pred_corr_lats_all[k] = corrlats1
        print(f'{k} corr val = {pred_corr_vals_mean[k]:.4f} +- {pred_corr_vals_std[k]:.4f}, corr lat mean={pred_corr_lats_mean[k]*1000:.2f}, median={pred_corr_lats_median[k]*1000:.2f} +- {pred_corr_lats_std[k]*1000:.4f}')
        print(f'{k} corr lat avg = {pred_corr_lats_mode[k]}')

    return shifts




def fit_TRFs(preds, options=None, subjects=None):
    if options is None:
        options = [dict(cond='A', pndtype='true', pndsm=0.3, nbins=4, ks=['rect', 'gt', 'oss', 'ossa', 'zil'], signs=['p'])]

    scale = [1, 10**(-12/20), 10**(-24/20), 10**(-36/20)]
    t1 = -0.01
    t2 = 0.03
    datapath = preprocessed_path
    outpath = speech_path 
    outpath.mkdir(exist_ok=True, parents=True)
    
    if subjects is None: subjects = [f'TP{i:04d}' for i in range(1, 25)]

    for opt in options:
        cond = opt['cond']
        pndtype = opt['pndtype']
        pndsm = opt['pndsm']
        nbins = opt['nbins']
        ks = opt['ks']
        signs = opt['signs']
        if 'Ntrials' not in opt.keys():
            Ntrials = -1
        else:
            Ntrials = opt['Ntrials']

        for predk in ks:
            for s in signs:
                outstr = f'test_{cond}_pred{predk}{s}_pnd{pndtype}{pndsm}_nbins{nbins}_Ntrials{Ntrials}'
                for i in tqdm(range(len(subjects))):
                    subject = subjects[i]
                    datadict = eel.load.unpickle(datapath / f'{subject}_speech_preprocessed_Cz_EXG1EXG2_4096_0-62.pkl')
                    eegs = datadict['eegs']
                    eegs = eel.combine(eegs)
                    eegs = eel.resample(eegs, 2048)
                    if predk in ['ossa','zil']:
                        levelflag = True
                    else: 
                        levelflag = False
                    
                    res = align_pred_eeg(subject, eegs, preds, predk, s, levelflag, scale=scale)

                    eegs2 = []
                    preds2 = []
                    rNs = []
                    for eeg, pred1 in zip(res[f'eegs{cond}'], res[f'preds{cond}'][predk+s]):
                        yy = eel.filter_data(eeg, 1, 40)
                        eeg1, pp, rN = utils.remove_outliers(eeg, yy, [pred1], stdmul=5)
                        eegs2.append(eeg1)
                        preds2.append(pp[0])
                        rNs.append(rN)

                    res[f'eegs{cond}'] = eel.combine(eegs2)
                    res[f'preds{cond}'][predk+s] = eel.combine(preds2)
                    outpath1 = outpath / f'{outstr}'
                    outpath1.mkdir(exist_ok=True)
                    eel.save.pickle(rNs, outpath1 / f'{subject}_rNs.pkl')

                    if Ntrials != -1:
                        res[f'eegs{cond}'] = res[f'eegs{cond}'][:Ntrials]
                        res[f'preds{cond}'][predk+s] = res[f'preds{cond}'][predk+s][:Ntrials]
                        res[f'pnds{cond}'] = res[f'pnds{cond}'][:Ntrials]
                        Nblocks = int(np.floor(len(res[f'eegs{cond}'])/4))
                        if Nblocks > 1:
                            res[f'eegs{cond}'] = eel.combine([eel.concatenate(res[f'eegs{cond}'][i*4:i*4+4]) for i in range(int(Nblocks))])
                            res[f'preds{cond}'][predk+s] = eel.combine([eel.concatenate(res[f'preds{cond}'][predk+s][i*4:i*4+4]) for i in range(int(Nblocks))])
                            res[f'pnds{cond}'] = eel.combine([eel.concatenate(res[f'pnds{cond}'][i*4:i*4+4]) for i in range(int(Nblocks))])

                    res[f'preds{cond}'][predk+s].x -= res[f'preds{cond}'][predk+s].sub(time=(0.1,0.5)).mean('case').mean('time')
                    res[f'preds{cond}'][predk+s] = res[f'preds{cond}'][predk+s].clip(min=0)

                    if pndtype == 'true':
                        pnd = res[f'pnds{cond}'].copy()
                    else:
                        predsm = res[f'preds{cond}'][predk+s].smooth('time', pndsm).clip(min=0)
                        pnd = predsm

                    res_s = fit_TRFs_level_wrapper(res[f'eegs{cond}'], res[f'preds{cond}'][predk+s], outpath / f'{outstr}', 
                                            f'{subject}_{predk}{s}', nbins=nbins, pnd=pnd, t1=t1, t2=t2)
            eel.save.pickle(opt, outpath / f'{outstr}/options.pkl')



def align_pred_eeg(subject, eegs, preds, k, s, levelflag, scale=False):
    if levelflag:
        pnd1 = preds[k+s+'0'][0].copy()
    else:
        pnd1 = preds[k+s][0].copy()
    if not scale:
        Pas = [1,1,1,1]
    else:
        Pas = scale

    pnd1.x = np.ones(len(pnd1.x))
    pnds = []
    pndswitch_dict = eel.load.unpickle(pnd_path / 'pnd_dict.pkl')

    
    eegfixed = []
    predsfixed = {}
    pndsfixed = []
    eegswitch = []
    predsswitch = {}
    pndsswitch = []
    predsfixed[k+s] = []
    predsswitch[k+s] = []

    i_s = int(0.5/pnd1.time.tstep)

    randdict = eel.load.unpickle(randomization_path / f'randsettings_{subject}_001.pickle')
    mags = randdict['levels']
    i_randlevel = 0
    for i in range(40):
        if randdict['fastflags'][i] == 0:
            randlev_idx = randdict['randlevels'][i_randlevel]
            randlevel = mags[randlev_idx]
            pnds.append(randlevel*pnd1)
            i_randlevel += 1
            eegfixed.append(eegs[i])
            if levelflag:
                predsfixed[k+s].append(preds[k+s+str(randlev_idx)][i]) 
            else:
                predsfixed[k+s].append(Pas[randlev_idx]*preds[k+s][i])
            pndsfixed.append(pnds[-1])

        else:
            pnds.append(pndswitch_dict['pnds'][i])
            eegswitch.append(eegs[i])
            predsswitch[k+s].append(preds[k+'switch'+s][i])
            pndsswitch.append(pnds[-1])

    randdict = eel.load.unpickle(randomization_path / f'randsettings_{subject}_002.pickle')
    mags = randdict['levels']
    i_randlevel = 0
    if subject == 'TP0003':
        Nt = 20
    elif subject == 'TP0009':
        Nt = 39
    else:
        Nt = 40
    for i in range(Nt):
        if subject=='TP0009' and i>32:
            istim = i+1
        else:
            istim = i
        if randdict['fastflags'][istim+40] == 0:
            randlev_idx = randdict['randlevels'][i_randlevel]
            randlevel = mags[randlev_idx]
            pnds.append(randlevel*pnd1)
            i_randlevel += 1
            eegfixed.append(eegs[i+40])
            if levelflag:
                predsfixed[k+s].append(preds[k+s+str(randlev_idx)][istim+40])
            else:
                predsfixed[k+s].append(Pas[randlev_idx]*preds[k+s][istim+40])       
            pndsfixed.append(pnds[-1])
        else:
            pnds.append(pndswitch_dict['pnds'][istim+40])
            eegswitch.append(eegs[i+40])
            predsswitch[k+s].append(preds[k+'switch'+s][istim+40])
            pndsswitch.append(pnds[-1])

    eegfixed = eel.combine(eegfixed).sub(time=(0, 60))
    eegswitch = eel.combine(eegswitch).sub(time=(0, 60))
    predsfixed[k+s] = eel.combine(predsfixed[k+s]).sub(time=(0, 60))
    predsswitch[k+s] = eel.combine(predsswitch[k+s]).sub(time=(0, 60))
    pndsfixed = eel.combine(pndsfixed).sub(time=(0, 60))
    pndsswitch = eel.combine(pndsswitch).sub(time=(0, 60))

    predsfixed1 = {}
    predsswitch1 = {}

    if subject == 'TP0003':
        Ntrials = 10
        Nblocks = 6
        eegfixed1 = eel.combine([eel.concatenate(eegfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsfixed1[k+s] = eel.combine([eel.concatenate(predsfixed[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsswitch1[k+s] = eel.combine([eel.concatenate(predsswitch[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsfixed1 = eel.combine([eel.concatenate(pndsfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        eegswitch1 = eel.combine([eel.concatenate(eegswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsswitch1 = eel.combine([eel.concatenate(pndsswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
    elif subject == 'TP0009':
        Ntrials = 9
        Nblocks = 8
        eegfixed1 = eel.combine([eel.concatenate(eegfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsfixed1[k+s] = eel.combine([eel.concatenate(predsfixed[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsswitch1[k+s] = eel.combine([eel.concatenate(predsswitch[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsfixed1 = eel.combine([eel.concatenate(pndsfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        eegswitch1 = eel.combine([eel.concatenate(eegswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsswitch1 = eel.combine([eel.concatenate(pndsswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
    else:
        Ntrials = 10
        Nblocks = 8
        eegfixed1 = eel.combine([eel.concatenate(eegfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsfixed1[k+s] = eel.combine([eel.concatenate(predsfixed[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        predsswitch1[k+s] = eel.combine([eel.concatenate(predsswitch[k+s][i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsfixed1 = eel.combine([eel.concatenate(pndsfixed[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        eegswitch1 = eel.combine([eel.concatenate(eegswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])
        pndsswitch1 = eel.combine([eel.concatenate(pndsswitch[i*Ntrials:i*Ntrials+Ntrials]) for i in range(int(Nblocks/2))])

    eegsA = eel.combine([eegfixed1, eegswitch1])

    predsA = {}
    predsA[k+s] = eel.combine([predsfixed1[k+s], predsswitch1[k+s]])
    pndsA = eel.combine([pndsfixed1, pndsswitch1])

    res = {}
    res['eegsA'], res['predsA'], res['pndsA'] = eegsA, predsA, pndsA
    res['eegsfixed1'], res['predsfixed1'], res['predsswitch1'], res['pndsfixed1'], res['eegsswitch1'], res['pndsswitch1'] = eegfixed1, predsfixed1, predsswitch1, pndsfixed1, eegswitch1, pndsswitch1
    res['predsswitch'], res['eegsswitch'], res['pndsswitch'], res['predsfixed'], res['eegsfixed'], res['pndsfixed'] = predsswitch, eegswitch, pndsswitch, predsfixed, eegfixed, pndsfixed
    del eegfixed, pndsfixed, eegswitch, pndsswitch
    return res





def fit_TRFs_level_wrapper(eegs, pred, res_path, savestr, nbins=10, pnd=None, t1=-0.01, t2=0.03):
    
    res_path.mkdir(exist_ok=True, parents=True)
       
    ys = eegs.copy()
    timedim = eel.UTS(0, ys.time.tstep, len(ys.time))
    if pnd is None:
        pnd = pred.copy()
        
    bins = []
    if pnd.has_case:
        xx = eel.concatenate(pnd).x.copy()
    else:
        xx = pnd.x.copy()
    xx=xx[xx!=0]
    X = np.sort(xx)
    N = len(X)
    Nlev = int(N/nbins)
    for i in range(1, nbins):
        val = np.mean([X[int((i-0.5)*Nlev)],X[int((i+0.5)*Nlev)]])
        bins.append(val)

    bins = [0] + bins

    ldtrfs, ldpreds, bin_rmss = fit_trf_levels(ys.copy(), pred.copy(), pnd.copy(), nbins, bins=bins, 
                                                        t1=t1, t2=t2, verbose=True)
    

    eel.save.pickle(dict(ldtrfs=ldtrfs, bins=bins, bin_rmss=bin_rmss), res_path / f'{savestr}_trfs.pkl')

    resdict = dict(ldtrfs=ldtrfs, pnd=pnd, pred=pred, ldpreds=ldpreds, ys=ys, bins=bins, bin_rmss=bin_rmss)
    return resdict



def split_pred_bins(x_in: eel.NDVar, p_in: eel.NDVar, bins):
    x = x_in.copy()
    p = p_in.x.copy()
    
    xsm = x.smooth('time', 0.3).x
    x = x.x
    x -= np.min(x)

    bin_rmss = []
    for i in range(len(bins)):
        rmss = []
        for j in range(len(x)):
            if i==len(bins)-1:
                psm = xsm[j][p[j]>=bins[i]].copy()
            else:
                idx = np.all([bins[i]<=p[j], p[j]<bins[i+1]], axis=0)
                psm = xsm[j][idx].copy()
            rmss.append(np.sqrt(np.mean(psm**2)))
        bin_rmss.append(np.mean(rmss))

    xs = []
    for i in range(len(bins)):
        xs1 = []    
        for j in range(len(x)):
            x1 = x[j].copy()
            multflag = np.ones(x1.shape)
            multflag /= bin_rmss[i]
            if i==len(bins)-1: 
                multflag[(p[j]<bins[i])] = 0
            else: 
                multflag[(p[j] < bins[i]) | (p[j] >= bins[i+1])] = 0
            x1 *= multflag
            x1[x1<=0] = min(x1[x1>0])
            xs1.append(x1)
        xs.append(xs1)
    xs = np.asarray(xs)

    for i in range(len(xs)):
        for j in range(len(xs[i])):
            xs[i,j,:] -= np.min(xs[i,j,:])

    return np.squeeze(np.asarray(xs)), bin_rmss



def fit_trf_levels(ynd, xnd, pnd, nlevels, bins=None, t1=-0.01, t2=0.03, edges=0.005, 
                    logger=None, verbose=False, lfreq=30, hfreq = 1000):
    
    if len(xnd.x.shape) == 1:
        xnd = eel.NDVar(xnd.x[np.newaxis, :], (eel.Case, xnd.time))
        ynd = eel.NDVar(ynd.x[np.newaxis, :], (eel.Case, xnd.time))
        pnd = eel.NDVar(pnd.x[np.newaxis, :], (eel.Case, xnd.time))

    xs, bin_rmss = split_pred_bins(xnd, pnd, bins)
    Nlev = len(xs)
    
    balphas = []
    trfcorrs = []
    
    ynd = eel.filter_data(ynd, lfreq, hfreq)
    preds = []
    for ilev in range(Nlev):
        pred = xnd.copy()
        pred.x = xs[ilev].copy()
        preds.append(pred)

    ntrials = len(xnd)
    tt = []
    for nt in range(ntrials):
        X = []
        i1 = int(t1/ynd.time.tstep)
        i2 = int(t2/ynd.time.tstep)
        edges = 0
        K = i2-i1
        Xsq2 = np.zeros((K*Nlev, K*Nlev))
        for ilev in range(Nlev):
            X1 = x_to_X(preds[ilev][nt].x, i1, i2, edges)
            X1[preds[ilev][nt].x==0,:] = 0
            X.append(X1)
        X = np.concatenate(X, axis=1)
        tt1 = np.linalg.pinv(X.T@X)@X.T@ynd[nt].x
        tt1 /= ynd[nt].std()
        tt.append(tt1)
    tt = np.mean(np.asarray(tt), axis=0)
    nlev = int(len(tt)/Nlev)
    fit_trfs = eel.combine([eel.NDVar(tt[i*nlev:(i+1)*nlev], eel.UTS(t1, ynd.time.tstep, nlev)) for i in range(Nlev)])
    
    return fit_trfs, preds, bin_rmss



def x_to_X(
    x: np.ndarray, 
    i1: int = 0, 
    i2: int = 50, 
    edges: int = 0,
) -> np.ndarray:
    '''
    Convert 1D predictor to lagged matrix
    
    Parameters
    ---------------------------
    x : 1D predictor
    i1 : start lag sample
    i2 : end lag sample
    edges : number of edge samples
    
    Returns
    ----------------------------
    X : Lagged matrix
    
    '''
    K = i2 - i1
    N = len(x)
    X1 = np.zeros((K+2*edges, N))
    i=0
    for k in range(i1-edges, i2+edges):
        if k < 0:
            X1[i, :] = np.concatenate([x[abs(k):], np.zeros(abs(k))])
        elif k == 0:
            X1[i, :] = x
        else:
            X1[i, :] = np.concatenate([np.zeros(k), x[:-k]])
        i += 1
    return X1.T
