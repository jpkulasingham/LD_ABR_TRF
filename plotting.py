import eelbrain as eel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib, scipy, statsmodels, pathlib
from tqdm import tqdm
from pathnames import *
colors = [(0, 0.2, 0.7), (0, 0.5, 0.8), (0.4, 0.8, 1), (0.4, 0.9, 0.7)]




def get_trfs(speechfolder, pred, badsubjs, shift, twin=0.002, pkwin=(0.004, 0.01), pkwin_adjust=[-0.001,0,0,0], trfkey='ldtrfs', binskey='bins'):
    trfs = []
    pklats = []
    pkamps = []
    pklatnegs = []
    pk2pkamps = []
    pkampnegs = []
    binsA = []
    subjects = []
    for isub in tqdm(range(24)):
        subject = f'TP00{isub+1:02d}'
        if subject in badsubjs:
            continue
        res = eel.load.unpickle(speechfolder / f'{subject}_{pred}_trfs.pkl')
        if 'trfsA' in res.keys():
            trf = res['trfsA']['ldtrfs']
        else:
            trf = res['ldtrfs']
        binsA.append(res['bin_rmss'])
        trf = eel.resample(trf.sub(time=(-0.01, 0.03)), 8000).smooth('time', twin)
        trf = eel.NDVar(trf.x, (eel.Case, eel.UTS(-0.01+shift, trf.time.tstep, len(trf[0]))))
        trf = eel.combine([trf[-i] for i in range(1, len(trf)+1)])
        trfs.append(trf)
        
        pklats1, pkamps1, pk2pkamps1, pklatnegs1, pkampnegs1 = get_pk(trf, pkwin, pkwin_adjust)
       
        pklats.append(pklats1)
        pkamps.append(pkamps1)
        pk2pkamps.append(pk2pkamps1)
        pklatnegs.append(pklatnegs1)
        pkampnegs.append(pkampnegs1)
        subjects.append(subject)
    return dict(trfs=trfs, pklats=pklats, pkamps=pkamps, pk2pkamps=pk2pkamps, pklatnegs=pklatnegs, pkampnegs=pkampnegs, subjects=subjects, bins=binsA)


def get_pk(trf, pkwin, pkwin_adjust):
    pklats1 = []
    pkamps1 = []
    pk2pkamps1 = []
    pklatnegs1 = []
    pkampnegs1 = []
    for i in range(len(trf)):
        ttpk = trf[i].sub(time=(pkwin[0]+pkwin_adjust[i], pkwin[1]+pkwin_adjust[i]))
        maxima = scipy.signal.argrelextrema(ttpk.x, np.greater)
        if len(maxima[0])==0:
            pkidx = np.argmax(ttpk.x)
        else:
            maxima_v = []
            for m in maxima[0]:
                maxima_v.append(ttpk.x[m])
            pkidx = maxima[0][np.argmax(maxima_v)]
        ttpk2 = trf[i].sub(time=(pkwin[0]+pkwin_adjust[i], pkwin[1]+pkwin_adjust[i]+0.005))
        ttpk3 = ttpk2.x[pkidx:]    
        minima = scipy.signal.argrelextrema(ttpk3, np.less)
        if len(minima[0])==0:
            pkidx2 = np.argmin(ttpk3)
        else:
            pkidx2 = minima[0][0]
        pkamp = ttpk.x[pkidx]-ttpk3[pkidx2]
        pkampnegs1.append(ttpk3[pkidx2])
        pklats1.append(ttpk.time.times[pkidx])
        pkamps1.append(ttpk.sub(time=(pklats1[-1])))
        pk2pkamps1.append(pkamp)
        pklatnegs1.append(ttpk2.time.times[pkidx:][pkidx2])

    return pklats1, pkamps1, pk2pkamps1, pklatnegs1, pkampnegs1



def get_waveVsnr(res, levels=4):
    Nsubjs = len(res['trfs'])
    snrs = []
    npows = []
    spows = []
    for isubj in range(Nsubjs):
        snrs1, npows1, spows1 = [], [], []
        for ilev in range(levels):
            tt = res['trfs'][isubj][ilev].copy()
            pktime = res['pklats'][isubj][ilev]
            spow = tt.sub(time=(pktime-0.0025, pktime+0.0025)).rms()**2
            npow = 0.5*(tt.sub(time=(-0.005, 0)).rms()**2+tt.sub(time=(-0.01, -0.005)).rms()**2)
            snr = 10*np.log10(np.max([(spow-npow)/npow, 10**(-0.5)]))
            snrs1.append(snr)
            npows1.append(npow)
            spows1.append(spow)
        snrs.append(snrs1)
        npows.append(npows1)
        spows.append(spows1)
    return snrs, spows, npows


def get_clicks(badsubjs=['TP0015', 'TP0020', 'TP0022'], twin=0.002, pkwin=(0.004, 0.011), shift=-0.0009):
    clicks = []
    click_pklats = []
    click_pkamps = []
    click_pk2pkamps = []
    click_pklatnegs = []
    click_pkampnegs = []
    subjects1 = []
    for isub in tqdm(range(24)):
        subject = f'TP00{isub+1:02d}'
        if subject in badsubjs:
            continue
        res = eel.load.unpickle(click_path / f'{subject}_ERP.pkl')
        erp = eel.combine([res[f'erp{i}'] for i in range(4)])
        erp = eel.resample(erp.sub(time=(-0.01, 0.025)), 8192).smooth('time', twin)
        erp = eel.NDVar(erp.x, (eel.Case, eel.UTS(-0.01+shift, erp.time.tstep, len(erp[0]))))
        erp = eel.combine([erp[i] for i in range(4)])
        clicks.append(erp)

        pklats1, pkamps1, pk2pkamps1, pklatnegs1, pkampnegs1 = get_pk(erp, pkwin, pkwin_adjust=np.zeros(len(erp)))
        
        click_pklats.append(pklats1)
        click_pkamps.append(pkamps1)
        click_pk2pkamps.append(pk2pkamps1)
        click_pklatnegs.append(pklatnegs1)
        click_pkampnegs.append(pkampnegs1)
        subjects1.append(subject)
    return clicks, click_pklats, click_pkamps, click_pk2pkamps, click_pklatnegs, click_pkampnegs, subjects1


def plot_fig1(res, ks, tstrs, savefolder=None, savestr='ALL', levels=4, plotwin=(-0.005,0.015), ylim=None):
    f, axs = plt.subplots(2, 3, figsize=(25, 10))
    for ik, k in enumerate(ks):
        ax = axs[int((ik)/3), int((ik)%3)]
        trfs = [t.sub(time=plotwin) for t in res['trfs'][k]]
        if k=='clicks':
            ylabel = 'Amplitude [uV]'
            legendstr = ['66 dBA', '60 dBA', '48 dBA', '36 dBA']
        else:
            ylabel = 'Amplitude [a.u.]'
            legendstr = ['72 dBA', '60 dBA', '48 dBA', '36 dBA']
        plot_avgtrf(ax, trfs, legendstr, tstrs[ik], ylabel, levels=levels, colors=colors, ylim=ylim)

    plt.subplots_adjust(hspace=0.5)
    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig1.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig1.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig1.tiff', bbox_inches='tight', dpi=300)


def plot_avgtrf(ax, trfs, legendstr, tstr, ylabel, levels=4, colors=colors, ylim=None):
    trfm = np.mean(np.asarray(trfs), axis=0)
    trfsem = np.std(np.asarray(trfs), axis=0)/np.sqrt(len(trfs))
    for i in range(levels):
        ax.plot(trfs[0].time.times*1000, trfm[i,:], color=colors[i])
        ax.fill_between(trfs[0].time.times*1000, trfm[i,:]-trfsem[i, :], trfm[i,:]+trfsem[i, :], color=colors[i], alpha=0.15)
    ax.axvline(0, color='k')
    ax.set_xlim([-5, 15])
    ax.set_xlabel('Time [ms]', fontsize=15)
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(custom_lines, legendstr, fontsize=13, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([-5,0,5,10,15])
    ax.set_xticklabels([-5,0,5,10,15], fontsize=14)
    ax.set_title(tstr, fontsize=17)
    if ylim is not None: ax.set_ylim(ylim)



def plot_fig2(res, ks, tstrs, savefolder=None, savestr='ALL', levels=4, pkstr='pkamps', ylim=0.0015):
    clicklevs = [0, 1, 2, 2.5]
    levs = [0, 1, 2 ,3]
    clickdBs = ['66', '60', '48', '36']
    levdBs = ['72', '60', '48', '36']
    levdBs = levdBs[::-1]
    clickdBs = clickdBs[::-1]
    plt.figure(figsize=(23, 6))
    for ik, k in enumerate(ks):
        if k == 'clicks':
            xdim = clicklevs
            xstrs = clickdBs
        else:
            xdim = levs
            xstrs = levdBs
        pklats = [x[::-1] for x in res['pklats'][k].copy()]
        pkamps = [x[::-1] for x in res[pkstr][k].copy()]
        Nsubj = len(pklats)

        pklatslev = [[pklats[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]
        pkampslev = [[pkamps[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]

        plt.subplot(2,6,ik+1)
        cmap = matplotlib.cm.get_cmap('winter')
        reds = 0
        for i in range(Nsubj):
            if pklats[i][0] - pklats[i][-1] > 0:
                color = 'g'
                lw = 1
            else:
                color = 'r'
                lw = 2
                reds += 1
                print('lat', k, i, pklats[i][0], pklats[i][-1])
            plt.plot(xdim, [p*1000 for p in pklats[i]], color=color, linewidth=lw, alpha=0.5)
        medianprops = dict(color="black",linewidth=1.5)
        plt.boxplot(pklatslev, positions=xdim, medianprops=medianprops);
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_xticklabels(xstrs, fontsize=10)
        if ik==0: plt.ylabel('Latency [ms]', fontsize=15)
        plt.title(tstrs[ik], fontsize=17)
        plt.xlim([-0.5, 3.5])
        plt.ylim([3, 12])

        plt.subplot(2,6,6+ik+1)
        medianprops = dict(color="black",linewidth=1.5)
        plt.boxplot(pkampslev, positions=xdim, medianprops=medianprops)
        reds = 0
        for i in range(Nsubj):
            if pkamps[i][0] - pkamps[i][-1] > 0:
                color = 'r'
                reds += 1
                lw = 2
                print('amp', k, i, pkamps[i][0], pkamps[i][-1])
            else:
                color = 'g'
                lw = 1
            plt.plot(xdim, [p*1000 for p in pkamps[i]], color=color, linewidth=lw, alpha=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_xticklabels(xstrs, fontsize=10)
        plt.gca().set_yticks([0, 0.0005, 0.001])
        plt.gca().set_yticklabels([0, 1, 2])
        plt.ylim([-0.0002, ylim])
        if ik==0: plt.ylabel('Amplitude [a.u.]', fontsize=15)
        plt.subplots_adjust(wspace=0.5)
        plt.xlim([-0.5, 3.5])

        pklatslev = [[pklats[j][i]*1000 for j in range(Nsubj)] for i in [0, -1]]
        pkampslev = [[pkamps[j][i]*1000 for j in range(Nsubj)] for i in [0, -1]]

    plt.subplots_adjust(hspace=0.2, wspace=0.4)
    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig2_{pkstr}.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig2_{pkstr}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig2_{pkstr}.tiff', bbox_inches='tight', dpi=300)



def plot_fig3(res, savefolder=None, k='gt', savestr='ALL', ampstr='pkamps'):
    markers = ['o', '*', 'x', '+']
    markersizes = [40, 50, 60, 80]
    markerwidths = [1, 1, 2.5, 2.5]
    plt.figure(figsize=(20, 12))
    ps = []
    ts = []
    xlats = []
    ylats = []
    xamps = []
    yamps = []
    dBs = [72, 60, 48, 36]
    click_pklats = [res['pklats']['clicks'][i].copy() for i in range(21)]
    click_pkamps = [res[ampstr]['clicks'][i].copy() for i in range(21)]
    pklats = [res['pklats'][k][i].copy() for i in range(21)]
    pkamps = [res[ampstr][k][i].copy() for i in range(21)]

    for ilev in range(4):
        plt.subplot(2,3,1)
        x = [click_pklats[i][ilev]*1000 for i in range(len(click_pklats))]
        y = [pklats[i][ilev]*1000 for i in range(len(pklats))]
        plt.scatter(x, y, color=colors[ilev], marker=markers[ilev], linewidths=markerwidths[ilev], s=markersizes[ilev])
        plt.xlabel('Click ERP latencies [ms]', fontsize=15)
        plt.ylabel(f'Speech GT TRF latencies [ms]', fontsize=15)
        plt.gca().set_xticks([5, 6, 7, 8, 9, 10])
        plt.gca().set_xticklabels([5, 6, 7, 8, 9, 10], fontsize=12)
        plt.gca().set_yticks([5, 6, 7, 8, 9, 10])
        plt.gca().set_yticklabels([5, 6, 7, 8, 9, 10], fontsize=12)
        xlats += x
        ylats += y

        plt.subplot(2,3,4)
        x = [click_pkamps[i][ilev] for i in range(len(click_pkamps))]
        y = [pkamps[i][ilev] for i in range(len(pkamps))]
        plt.scatter(x, y, color=colors[ilev], marker=markers[ilev], linewidths=markerwidths[ilev], s=markersizes[ilev])
        plt.xlabel('Click ERP amplitudes [a.u.]', fontsize=15)
        plt.ylabel(f'Speech GT TRF amplitudes [a.u.]', fontsize=15)
        plt.gca().set_xticks([2e-7, 4e-7, 6e-7, 8e-7])
        plt.gca().set_xticklabels([2, 4, 6, 8], fontsize=12)
        plt.gca().set_yticks([2e-7, 4e-7, 6e-7, 8e-7])
        plt.gca().set_yticklabels([2, 4, 6, 8], fontsize=12)
        xamps += x
        yamps += y
    
    plt.subplot(2,3,1)
    plt.legend(['72/66 dBA', '60 dBA', '48 dBA', '36 dBA'], fontsize=13)
    plt.subplot(2,3,4)
    plt.legend(['72/66 dBA', '60 dBA', '48 dBA', '36 dBA'], fontsize=13)

    t, p = scipy.stats.pearsonr(xlats, ylats)
    ps.append(p)
    ts.append(t)

    t, p = scipy.stats.pearsonr(xamps, yamps)
    ps.append(p)
    ts.append(t)

    [_, pcorrected, _,_] = statsmodels.stats.multitest.multipletests(ps)

    amp_slopes, amp_intercepts, lat_slopes, lat_intercepts = fit_slopes(pkamps, pklats, [72, 60, 48, 36])
    click_amp_slopes, click_amp_intercepts, click_lat_slopes, click_lat_intercepts = fit_slopes(click_pkamps, click_pklats, [66, 60, 48, 36])

    for i, (x, y) in enumerate(zip([click_lat_slopes, click_lat_intercepts], [lat_slopes, lat_intercepts])):
        plt.subplot(2,3,i+2)
        plt.scatter(x, y, color='k')
        print(len(x), len(y))
        t, p = scipy.stats.pearsonr(x, y)
        plt.xlabel('click ERP')
        plt.ylabel(f'GT TRF')
        ps.append(p)
        ts.append(t)

    for i, (x, y) in enumerate(zip([click_amp_slopes, click_amp_intercepts], [amp_slopes, amp_intercepts])):
        plt.subplot(2,3,3+i+2)
        print(len(x), len(y))
        plt.scatter(x, y, color='k')
        t, p = scipy.stats.pearsonr(x, y)
        ps.append(p)
        ts.append(t)

    [_, pcorrected, _,_] = statsmodels.stats.multitest.multipletests(ps)

    pcorrstrs = [f'p={p:.4f}' if p>0.001 else 'p<0.001' for p in pcorrected]

    plt.subplot(2,3,1)
    plt.title(f'Wave V Peak Latencies\nr={ts[-2]:.3f}, {pcorrstrs[0]}', fontsize=17)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot([5.5,10.5], [5.5,10.5], color='k', linestyle='dashed')
    plt.subplot(2,3,4)
    plt.title(f'Wave V Peak Amplitudes\nr={ts[-1]:.3f}, {pcorrstrs[1]}', fontsize=17)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot([1e-7,9e-7], [1e-7,9e-7], color='k', linestyle='dashed')

    plt.subplot(2,3,2)
    plt.title(f'Fitted Latency Slopes\nr={ts[2]:.3f}, {pcorrstrs[2]}', fontsize=17)
    plt.xlabel('Click ERP latency slopes [ms/dBA]', fontsize=15)
    plt.ylabel(f'GT TRF latency slopes [ms/dBA]', fontsize=15)
    plt.plot([-0.015, -0.07], [-0.015, -0.07], color='k', linestyle='dashed')
    plt.gca().set_xticks([-0.02, -0.04, -0.06])
    plt.gca().set_xticklabels([-0.02, -0.04, -0.06], fontsize=13)
    plt.gca().set_yticks([-0.02, -0.04, -0.06])
    plt.gca().set_yticklabels([-0.02, -0.04, -0.06], fontsize=13)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2,3,3)
    plt.title(f'Fitted Latency Intercepts\nr={ts[3]:.3f}, {pcorrstrs[3]}', fontsize=17)
    plt.xlabel('Click ERP latency intercepts [ms]', fontsize=15)
    plt.ylabel(f'GT TRF latency intercepts [ms]', fontsize=15)
    plt.plot([7.5, 11.5], [7.5, 11.5], color='k', linestyle='dashed')
    plt.gca().set_xticks([8, 9, 10, 11])
    plt.gca().set_xticklabels([8, 9, 10, 11], fontsize=13)
    plt.gca().set_yticks([8, 9, 10, 11])
    plt.gca().set_yticklabels([8, 9, 10, 11], fontsize=13)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.subplot(2,3,5)
    plt.title(f'Fitted Amplitude Slopes\nr={ts[4]:.3f}, {pcorrstrs[4]}', fontsize=17)
    plt.xlabel('Click ERP amplitude slopes [a.u./dBA]', fontsize=15)
    plt.ylabel(f'GT TRF amplitude slopes [a.u./dBA]', fontsize=15)
    plt.plot([-0.005, 0.025], [-0.005, 0.025], color='k', linestyle='dashed')
    plt.gca().set_xticks([0, 0.01, 0.02])
    plt.gca().set_xticklabels([0, 0.01, 0.02], fontsize=13)
    plt.gca().set_yticks([0, 0.01, 0.02])
    plt.gca().set_yticklabels([0, 0.01, 0.02], fontsize=13)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2,3,6)
    plt.title(f'Fitted Amplitude Intercepts\nr={ts[5]:.3f}, {pcorrstrs[5]}', fontsize=17)
    plt.xlabel('Click ERP amplitude intercepts [a.u.]', fontsize=15)
    plt.ylabel(f'GT TRF amplitude intercepts [a.u.]', fontsize=15)
    plt.plot([-0.45, 0.45], [-0.45, 0.45], color='k', linestyle='dashed')
    plt.gca().set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    plt.gca().set_xticklabels([-0.4, -0.2, 0, 0.2, 0.4], fontsize=13)
    plt.gca().set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    plt.gca().set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4], fontsize=13)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig3_{ampstr}.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig3_{ampstr}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig3_{ampstr}.tiff', bbox_inches='tight', dpi=300)

    return dict(click_amp_slopes=click_amp_slopes, click_amp_intercepts=click_amp_intercepts, click_lat_slopes=click_lat_slopes, click_lat_intercepts=click_lat_intercepts,
                amp_slopes=amp_slopes, amp_intercepts=amp_intercepts, lat_slopes=lat_slopes, lat_intercepts=lat_intercepts,)


def fit_slopes(pkamps, pklats, leveldim):
    amp_slopes, amp_intercepts, lat_slopes, lat_intercepts = [], [], [], []
    pkampmax = np.max(np.max(np.asarray(pkamps)))
    pkamps = np.asarray(pkamps)/pkampmax

    for i in range(len(pkamps)):
        linres = scipy.stats.linregress(leveldim, pkamps[i])
        amp_slopes.append(linres.slope)
        amp_intercepts.append(linres.intercept)

        linres = scipy.stats.linregress(leveldim,  np.asarray(pklats[i])*1000)
        lat_slopes.append(linres.slope)
        lat_intercepts.append(linres.intercept)

    return amp_slopes, amp_intercepts, lat_slopes, lat_intercepts



def plot_fig4(res, ks, tstrs, levels=4, ylim=None, savefolder=None, plotwin=(-0.005, 0.015), savestr='ALL', ampstr='pkamps', ampylim=0.0015):
    plt.figure(figsize=(20, 10))
    xdim = [0,1,2,3]
    xstrs = ['72', '60', '48', '36']
    xstrs = xstrs[::-1]

    for ik, k in enumerate(ks):
        plt.subplot(len(ks), 2, ik*2+1)
        ax = plt.gca()
        trfs = [t.sub(time=plotwin) for t in res['trfs'][k]]
        ylabel = 'Amplitude [a.u.]'
        legendstr = ['72 dBA', '60 dBA', '48 dBA', '36 dBA']
        plot_avgtrf(ax, trfs, legendstr, 'Speech GT TRF', ylabel, levels=levels, colors=colors, ylim=ylim)
        plt.text(-9, -4e-7, tstrs[ik], fontsize=22, rotation='vertical', horizontalalignment='center')

        pklats = [x[::-1] for x in res['pklats'][k].copy()]
        pkamps = [x[::-1] for x in res[ampstr][k].copy()]
        Nsubj = len(pklats)

        pklatslev = [[pklats[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]
        pkampslev = [[pkamps[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]

        plt.subplot(len(ks), 4, ik*4+3)
        cmap = matplotlib.cm.get_cmap('winter')
        for i in range(Nsubj):
            if pklats[i][0] - pklats[i][-1] > 0:
                color = 'g'
                lw = 1
            else:
                color = 'r'
                lw = 2
            plt.plot(xdim, [p*1000 for p in pklats[i]], color=color, alpha=0.5, lw=lw)
        medianprops = dict(color="black",linewidth=1.5)
        plt.boxplot(pklatslev, positions=xdim, medianprops=medianprops);
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_xticklabels(xstrs, fontsize=10)
        plt.ylabel('Latency [ms]', fontsize=12)
        plt.xlabel('Intensity [dBA]', fontsize=12)
        plt.title('Latency', fontsize=17)
        plt.xlim([-0.5, 3.5])
        plt.ylim([4, 11])

        plt.subplot(len(ks), 4, ik*4+4)
        medianprops = dict(color="black",linewidth=1.5)
        plt.boxplot(pkampslev, positions=xdim, medianprops=medianprops);
        for i in range(Nsubj):
            if pkamps[i][0] - pkamps[i][-1] > 0:
                color = 'r'
                lw = 2
            else:
                color = 'g'
                lw = 1
            plt.plot(xdim, [p*1000 for p in pkamps[i]], color=color, alpha=0.5, lw=lw)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_xticklabels(xstrs, fontsize=10)
        plt.gca().set_yticks([0, 0.0005, 0.001])
        plt.gca().set_yticklabels([0, 1, 2])
        plt.ylim([-0.0002, ampylim])
        plt.ylabel('Amplitude [a.u.]', fontsize=12)
        plt.xlabel('Intensity [dBA]', fontsize=12)
        plt.title('Amplitude', fontsize=17)
        plt.xlim([-0.5, 3.5])

    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig4_{ampstr}.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig3_{ampstr}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig3_{ampstr}.tiff', bbox_inches='tight', dpi=300)




def plot_fig5(pkampsA, pklatsA, snrsA, levels=4, colors=None, dBlevs=['72 dBA', '60 dBA', '48 dBA', '36 dBA'], hline=0, savefolder=None, savestr='ALL'):
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    if colors is None:
        colors = [(0, 0.2, 0.7, 0.7), (0, 0.5, 0.8, 0.7), (0.4, 0.8, 1, 0.7), (0.4, 0.9, 0.7, 0.7)]
        colors = colors[::-1]
    amp_slopesA = []
    amp_interceptsA = []
    lat_slopesA = []
    lat_interceptsA = []

    for ntrial in range(len(pkampsA)):
        amp_slopes, amp_intercepts, lat_slopes, lat_intercepts = fit_slopes(pkampsA[ntrial], pklatsA[ntrial], [72, 60, 48, 36])
        amp_slopesA.append(amp_slopes)
        amp_interceptsA.append(amp_intercepts)
        lat_slopesA.append(lat_slopes)
        lat_interceptsA.append(lat_intercepts)
        
    plt.figure(figsize=(20, 10))
    plt.subplot(2,2,1)
    bplot = plt.boxplot(lat_slopesA, widths=0.2, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor((0,0.7,1,0.5))
        patch.set_linewidth(0)
    for m in bplot['medians']:
        m.set_color('black')
        m.set_linewidth(3)
    for i in range(len(amp_slopesA[0])):
        plt.plot(range(1,11), [lat_slopesA[j][i] for j in range(10)], lw=1, color=(0,0.4,1,0.4))
    plt.title('Wave V Latency Slopes and Data Length', fontsize=16)
    plt.xlabel('Datalength [mins]', fontsize=14)
    plt.axhline(0, color='k')
    plt.gca().set_xticklabels([f'{i*4}' for i in range(1,11)], fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2,2,2)
    bplot = plt.boxplot(amp_slopesA, widths=0.2, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor((0,0.7,1,0.5))
        patch.set_linewidth(0)
    for m in bplot['medians']:
        m.set_color('black')
        m.set_linewidth(3)
    for i in range(len(amp_slopesA[0])):
        plt.plot(range(1,11), [amp_slopesA[j][i] for j in range(10)], lw=1, color=(0,0.4,1,0.4))
    plt.title('Wave V Amplitude Slopes and Data Length', fontsize=16)
    plt.xlabel('Datalength [mins]', fontsize=14)
    plt.axhline(0, color='k')
    plt.gca().set_xticklabels([f'{i*4}' for i in range(1,11)], fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.subplot(2,1,2)
    patches = []
    L = len(snrsA)
    for ilev in range(levels):
        snr1 = [[snrsA[ilen][isubj][::-1][ilev] for isubj in range(snrsA.shape[1])] for ilen in range(L)]
        bplot = plt.boxplot(snr1, positions=[i*10+ilev for i in range(L)], patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[ilev])
            patch.set_linewidth(0)
        for m in bplot['medians']:
            m.set_color('black')
            m.set_linewidth(3)
    patches = [mpatches.Patch(color=colors[::-1][ilev], label=dBlevs[ilev]) for ilev in range(4)]
    ax = plt.gca()
    ax.legend(handles=patches, loc='upper left', fontsize=12)
    ax.set_xlim([-1, 95])
    ax.set_xticks([i*10+1.5 for i in range(L)])
    ax.set_xticklabels([i+1 for i in range(L)], fontsize=12)
    plt.yticks(fontsize=12)
    for v in [5, 10, 15, 20]:
        plt.axhline(v, color='k', alpha=0.2, linestyle='dashed')
    if hline is not None: plt.axhline(hline, color='k', linestyle='dashed')
    ax.set_ylabel('SNR [dB]', fontsize=14)
    ax.set_xlabel('Data length per level [minutes]', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Wave V SNR and Data Length', fontsize=16)

    plt.subplots_adjust(hspace=0.5)

    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig5.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig5.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig5.tiff', bbox_inches='tight', dpi=300)





def plot_fig6(res, bins, k, tstr, ylim=None, savefolder=None, plotwin=(-0.005, 0.015), savestr='ALL', levels=8, ampstr='pkamps', ampylim=0.0015):
    cmap = matplotlib.cm.get_cmap('winter')
    colors10 = [cmap(i/levels) for i in range(levels)]
    plt.figure(figsize=(12, 10))
    xdim = bins
    xstrs = [f'{int(p)}' for p in bins]
    colors = []

    plt.subplot(2, 1, 1)
    ax = plt.gca()        
    trfs = [t.sub(time=plotwin) for t in res['trfs'][k]]
    ylabel = 'Amplitude [a.u.]'
    legendstr = list(range(levels, 0, -1))
    plot_avgtrf(ax, trfs, legendstr, tstr, ylabel, levels=levels, colors=colors10, ylim=ylim)
    ax.legend([])

    pklats = [x[::-1] for x in res['pklats'][k].copy()]
    pkamps = [x[::-1] for x in res[ampstr][k].copy()]
    Nsubj = len(pklats)

    pklatslev = [[pklats[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]
    pkampslev = [[pkamps[j][i]*1000 for j in range(Nsubj)] for i in range(levels)]

    plt.subplot(2, 2, 3)
    cmap = matplotlib.cm.get_cmap('winter')
    for i in range(Nsubj):
        if pklats[i][0] - pklats[i][-1] > 0:
            color = 'g'
            lw = 1
        else:
            color = 'r'
            lw = 2
        plt.plot(xdim, [p*1000 for p in pklats[i]], color=color, alpha=0.5, lw=lw)
    medianprops = dict(color="black",linewidth=1.5)
    plt.boxplot(pklatslev, positions=xdim, medianprops=medianprops, widths=[2 for _ in range(len(xdim))]);
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_xticks(xdim)
    plt.gca().set_xticklabels(xstrs, fontsize=10)
    plt.ylabel('Latency [ms]', fontsize=12)
    plt.xlabel('Intensity level [dBA]', fontsize=12)
    plt.title('Latency', fontsize=17)
    plt.xlim([25, 75])
    plt.ylim([4, 11])

    plt.subplot(2, 2, 4)
    medianprops = dict(color="black",linewidth=1.5)
    plt.boxplot(pkampslev, positions=xdim, medianprops=medianprops, widths=[2 for _ in range(len(xdim))]);
    for i in range(Nsubj):
        if pkamps[i][0] - pkamps[i][-1] > 0:
            color = 'r'
            lw = 2
        else:
            color = 'g'
            lw = 1
        plt.plot(xdim, [p*1000 for p in pkamps[i]], color=color, alpha=0.5, lw=lw)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_xticks(xdim)
    plt.gca().set_xticklabels(xstrs, fontsize=10)
    plt.gca().set_yticks([0, 0.0005, 0.001])
    plt.gca().set_yticklabels([0, 1, 2])
    plt.ylim([-0.0002, ampylim])
    plt.ylabel('Amplitude [a.u.]', fontsize=12)
    plt.xlabel('Intensity level [dBA]', fontsize=12)
    plt.subplots_adjust(wspace=0.5)
    plt.title('Amplitude', fontsize=17)
    plt.xlim([25, 75])

    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    if savefolder is not None:
        plt.savefig(savefolder / f'{savestr}_fig6_{ampstr}.png', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig6_{ampstr}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(savefolder / f'{savestr}_fig6_{ampstr}.tiff', bbox_inches='tight', dpi=300)



def plot_indiv_trfs(res, ks, tstrs, savefolder=None, colors1=colors, legendstr=None, savestr='ALL', levels=4, plotwin=(-0.005,0.015), ylim=None, badsubj_idx=None):
    for ik, k in enumerate(ks):
        f, axs = plt.subplots(6, 4, figsize=(13, 18))
        trfs = [t.sub(time=plotwin) for t in res['trfs'][k]]
        pklats = res['pklats'][k].copy()
        if k=='clicks':
            ylabel = 'Amplitude [uV]'
            if legendstr is None: legendstr = ['66 dBA', '60 dBA', '48 dBA', '36 dBA']
        else:
            ylabel = 'Amplitude [a.u.]'
            if legendstr is None: legendstr = ['72 dBA', '60 dBA', '48 dBA', '36 dBA']
        for isubj in range(len(trfs)):
            ax = axs[int((isubj)/4), int((isubj)%4)]
            for i in range(levels):
                ax.plot(trfs[0].time.times*1000, trfs[isubj][i].x, color=colors1[i])
                ax.scatter(res['pklats'][k][isubj][i]*1000, res['pkamps'][k][isubj][i], color=colors1[i], marker='x', s=40)
                # ax.scatter(res['pklatnegs'][k][isubj][i]*1000, res['pkampnegs'][k][isubj][i], color=colors1[i], marker='x', s=40)
            if badsubj_idx is not None and badsubj_idx == isubj:
                ax.set_title(f'Participant {isubj+1:02d}', fontsize=13, color='r')
            else:
                ax.set_title(f'Participant {isubj+1:02d}', fontsize=13)
            ax.axvline(0, color='k')
            ax.set_xlim([-5, 15])
            ax.set_xticks([-5,0,5,10,15])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time [ms]', fontsize=12)
            ax.set_xticklabels([-5,0,5,10,15], fontsize=14)
            if ylim is not None: ax.set_ylim(ylim)

        custom_lines = [Line2D([0], [0], color=color, lw=3) for color in colors1]
        axs[5,3].legend(custom_lines, legendstr, fontsize=12, loc='upper right')

        for i in range(6):
            ax = axs[i, 0]
            ax.set_ylabel(ylabel, fontsize=12)

        for i in range(2):
            ax = axs[5, i+2]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        plt.subplots_adjust(hspace=0.8)
        if savefolder is not None:
            plt.savefig(savefolder / f'{savestr}_indivTRFs_{k}.png', bbox_inches='tight', dpi=300)
            plt.savefig(savefolder / f'{savestr}_indivTRFs_{k}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(savefolder / f'{savestr}_indivTRFs_{k}.tiff', bbox_inches='tight', dpi=300)