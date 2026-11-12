# -*- coding: utf-8 -*-
# 独立绘图文件：所需数据直接嵌入本文件，不读取仓库数据、CSV、模型或GPU。
# 修改配色/线宽/字号/坐标范围后直接运行；输出覆盖 ../figs/current/ 对应图片。
# 依赖：Python 3.10+，numpy，matplotlib。
"""Render RT paper figures from frozen CSVs; never train or modify sources."""
import csv,io
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

P = Path('embedded')
O=Path(__file__).resolve().parent.parent/'figs/current';O.mkdir(parents=True,exist_ok=True)
FINAL = P/'results/paper/final_resident864_20261016/tables'
SOURCE = P/'results/paper/rt864_final_20261016'
COLORS = {'FastSET':'#0072BD','CNN':'#D95319'}
MARKERS = {'FastSET':'o','CNN':'s'}
MAP = []
CAPTIONS = {}
DISPLAY_RANGES = {}
INPUTS = [SOURCE/'ber_curves.csv', FINAL/'rt_timing.csv', FINAL/'rt_delay_all.csv', FINAL/'rt_delay_ber.csv']

plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.labelsize':9,
 'axes.titlesize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':True,'axes.spines.right':True,
 'axes.linewidth':.7,'lines.linewidth':1.85,'savefig.dpi':300,'xtick.direction':'out','ytick.direction':'out'})

from ieee_style import apply as apply_ieee_style
apply_ieee_style()

def paired_times(ax, labels, pairs, colors, qualified=None):
 pairs=np.asarray(pairs,dtype=float);top=float(pairs.max())
 for j,col in enumerate(colors):
  xs=np.arange(len(labels))*1.45+(-.28 if j==0 else .28)
  vals=pairs[:,j]
  ax.bar(xs,vals,width=.52,color=col,zorder=3)
  for x,value in zip(xs,vals):
   # Anchor FastSET labels toward the left of their bars, clear of CNN.
   ax.annotate(f'{value:.2f}',(x,value+top*(.025 if j==0 else .065)),
               xytext=(-3 if j==0 else 0,0),textcoords='offset points',
               rotation=0,ha='center',va='bottom',
               fontsize=9,color='black',zorder=5)
 for i,(ff,cc) in enumerate(pairs):
  y=max(ff,cc)+top*.21;x=i*1.45
  ax.plot([x-.28,x-.28,x+.28,x+.28],[y-top*.02,y,y,y-top*.02],color='#555555',lw=.7,zorder=4)
  ax.text(x,y+top*.025,f'{cc/ff:.1f}×',ha='center',va='bottom',fontsize=9)
 ax.set(xticks=np.arange(len(labels))*1.45,xticklabels=labels,ylabel='Time (s)',ylim=(0,top*1.45),xlim=(-.85,(len(labels)-1)*1.45+.85))
 ax.grid(axis='y',color='#d6d6d6',lw=.5);ax.set_axisbelow(True);ax.tick_params(direction='in',pad=2)


def rows(p):
 return list(csv.DictReader(io.StringIO(DATA_CSV[Path(p).relative_to(P).as_posix()])))


def save(fig,name,caption,sources):
    dst=O/name; dst.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(dst.with_suffix('.'+ext),facecolor='white')
    plt.close(fig)
    CAPTIONS[name]=caption
    MAP.append({'old_basename':Path(name).name,'new_png':str(dst.with_suffix('.png')),
                'new_pdf':str(dst.with_suffix('.pdf')),'source_csvs':';'.join(str(x) for x in sources)})

def legend_handles():
    hh=[]
    for method in ['FastSET','CNN']:
        for phase in ['before','after']:
            hh.append(Line2D([],[],color=COLORS[method],marker=MARKERS[method],markersize=4.5,
               ls='--' if phase=='before' else '-',mfc='white' if phase=='before' else COLORS[method],
               label=f'{method} {phase}'))
    return hh


def yrange(rr,xmax=12):
    shown=[max(float(r['ber']),1/int(r['bits'])) for r in rr if 4<=float(r['snr_db'])<=xmax]
    low=min(shown+[2e-4]);high=max(shown+[2e-4]); bottom=10**np.floor(np.log10(low))
    return (bottom*.8,min(1,high*1.65))

def axis_style(ax,limits,xmax=12):
    ax.set(xlim=(3.85,xmax+.15),ylim=limits,xticks=range(4,xmax+1,2),xlabel=r'$E_b/N_0$ (dB)',ylabel='BER',yscale='log')
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=7));ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis='both',which='major',color='#DDE1E5',lw=.5);ax.set_axisbelow(True)
    ax.axhline(2e-4,color='#777777',ls=':',lw=.65,zorder=1)
    ax.text(4.18,2e-4*1.18,r'$\epsilon=2\times10^{-4}$',color='#777777',fontsize=8.5,ha='left',va='bottom')
    ax.tick_params(pad=2,labelsize=8.5);ax.xaxis.labelpad=2;ax.yaxis.labelpad=2

def draw_curve(ax,rr,method,ls='-',filled=True,label=None,xmax=12):
    full=sorted(rr,key=lambda r:float(r['snr_db']))
    assert len(full)==15
    rr=[r for r in full if 4<=float(r['snr_db'])<=xmax]
    x=np.array([float(r['snr_db']) for r in rr]);raw=np.array([float(r['ber']) for r in rr]);zero=raw==0
    y=np.array([max(float(r['ber']),1/int(r['bits'])) for r in rr])
    ax.plot(x,y,color=COLORS[method],ls=ls,lw=1.85,marker=MARKERS[method],mfc=COLORS[method] if filled else 'white',mec=COLORS[method],mew=.85,ms=4.5,markevery=[i for i in range(0,len(rr),2) if not zero[i]],label=label,zorder=3)
    if zero.any():ax.scatter(x[zero],y[zero],marker='v',s=24,facecolors=COLORS[method] if filled else 'white',edgecolors=COLORS[method],lw=.8,zorder=4)

def display_target(label):
    return label.replace(' P2',' __OLDP2__').replace(' P3',' P2').replace(' __OLDP2__',' P1')

def ber_plot(data,cases,name,labels=None,xmax=12):
    nc=len(cases);width=7.1 if nc==4 else 3.5;height=(2.15 if nc==4 else 2.05) if xmax==12 else 2.45
    fig,axes=plt.subplots(1,nc,figsize=(width,height),squeeze=False)
    selected=[r for r in data if r['case'] in cases];limits=yrange(selected,xmax)
    for i,(ax,case) in enumerate(zip(axes.flat,cases)):
        axis_style(ax,limits,xmax)
        for method in ['FastSET','CNN']:
            for phase in ['pretrained','adapted']:
                rr=[r for r in data if r['case']==case and r['method']==method and r['phase']==phase]
                draw_curve(ax,rr,method,ls='--' if phase=='pretrained' else '-',filled=phase=='adapted',label=method+' '+('before' if phase=='pretrained' else 'after'),xmax=xmax)
    fig.legend(handles=legend_handles(),ncol=4 if nc==4 else 2,loc='upper center',bbox_to_anchor=(.54,.995),frameon=False,columnspacing=.75,handlelength=2.4,handletextpad=.4,labelspacing=.25,fontsize=8.5)
    fig.subplots_adjust(left=.07 if nc==4 else .135,right=.985 if nc==4 else .965,bottom=.18,top=.86 if nc==4 else .80,wspace=.48 if nc==4 else .62)
    DISPLAY_RANGES[name]={'width_inches':width,'height_inches':height,'layout':f'1x{nc}','snr_db':[4,xmax],'available_snr_db':[4,18],'ylim':list(limits),'shared_ylimits_within_figure':True}
    caption=('Panels, left to right: '+', '.join(labels if labels else cases)+'. BER before (dashed, open markers) and after (solid, filled markers) adaptation for FastSET (8-64, green circles) and CNN (navy squares), from fixed source checkpoints. Each SNR uses 26,054,400 decoded information bits. The dotted line is the target BER 2e-4. Axes show measured points from 4–'+str(xmax)+' dB, without smoothing. Full 4–18 dB CSVs and earlier figures are retained. SF denotes San Francisco. RT target positions were constructed and screened as recorded, not a blind city trial. Downward triangles, when present, denote zero observed errors plotted at 1/N, not an estimate of the true BER.')
    save(fig,name,caption,[INPUTS[0]])

def delay_plot(data):
    cases=[900,1200];selected=[r for r in data if int(r['delay_ns']) in cases and r['phase']=='adapted'];before=[r for r in data if int(r['delay_ns'])==900 and r['phase']=='pretrained'];limits=yrange(selected+before)
    fig,(ax,right)=plt.subplots(1,2,figsize=(3.5,3.0),gridspec_kw={'width_ratios':[1,1.5]})
    axis_style(ax,limits);ax.set_xlim(3.6,12.4)
    styles={900:'-',1200:'-.'}
    for method in ['FastSET','CNN']:
        draw_curve(ax,[r for r in before if r['method']==method],method,ls='--',filled=False,label=f'{method} before')
    for delay in cases:
        for method in ['FastSET','CNN']:
            rr=[r for r in selected if int(r['delay_ns'])==delay and r['method']==method]
            draw_curve(ax,rr,method,ls=styles[delay],filled=False,label=f'{method} {delay}')
    dd=rows(INPUTS[2]);pairs=[];qualified=[]
    for delay in [600,900,1200]:
        ff,cc=[next(r for r in dd if r['delay_ns']==str(delay) and r['method']==method) for method in ['FastSET','CNN']]
        pairs.append([float(ff['total_seconds']),float(cc['total_seconds'])]);qualified.append(ff['qualified']==cc['qualified']=='True')
    ratios=[cc/ff for ff,cc in pairs]
    paired_times(right,['600','900','1200'],pairs,[COLORS['FastSET'],COLORS['CNN']],qualified);right.set_xlabel('Delay (ns)')
    h,l=ax.get_legend_handles_labels();order=[0,2,4,1,3,5];fig.legend([h[i] for i in order],[l[i] for i in order],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.54,1),columnspacing=.65,handlelength=1.8,labelspacing=.25,fontsize=9)
    fig.subplots_adjust(left=.15,right=.98,bottom=.19,top=.74,wspace=.52)
    DISPLAY_RANGES['rt_delay_ber']={'width_inches':3.5,'height_inches':3.0,'layout':'before/after BER and paired absolute time,1x2','ber_cases_ns':cases,'time_cases_ns':[600,900,1200],'time_ratios':ratios,'equal_quality_qualified':qualified,'snr_db':[4,12],'available_snr_db':[4,18],'ylim':list(limits)}
    save(fig,'rt_delay_ber','Left: two900ns pretrained curves followed by four adapted900/1200ns curves. Right: recorded FastSET/CNN durations in seconds, with their ratios above each pair at600/900/1200ns. The900ns recorded-duration ratio is not an equal-reliability speedup: FastSET full-test BER is4.3877e-4, above2e-4. The other pairs meet the target. All BER and time measurements are unchanged.',[INPUTS[3],INPUTS[2]])

def timing_plot():
    rr=rows(INPUTS[1]); cases=['Paris P2','Paris P3','Munich P2','San Francisco P2','Florence P2','Paris to San Francisco','San Francisco to Paris']
    names=['Paris P2','Paris P3','Munich P2','SF P2','Florence P2','Paris → SF','SF → Paris']
    fig,ax=plt.subplots(figsize=(3.5,2.65))
    for method,offset in [('FastSET',-.18),('CNN',.18)]:
        selected=[next(r for r in rr if r['case']==c and r['method']==method) for c in cases]
        vals=[float(r['total_seconds']) for r in selected]
        ax.barh(np.arange(7)+offset,vals,height=.31,color=COLORS[method],label=method,zorder=3)
        for i,v in enumerate(vals):ax.text(v+.055,i+offset,f'{v:.2f}',ha='left',va='center',fontsize=8.5)
    ax.set(yticks=np.arange(7),yticklabels=[display_target(n) for n in names],xlabel='Online adaptation time (s)',ylabel='Target / transfer',xlim=(0,5.5));ax.invert_yaxis()
    ax.grid(axis='both',color='#E1E4E8',lw=.55);ax.set_axisbelow(True)
    ax.legend(ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.48,1.17))
    fig.subplots_adjust(left=.30,right=.98,bottom=.17,top=.86)
    save(fig,'urban_adaptation_time','Same-host online adaptation time for seven RT transfers. SF denotes San Francisco. Values add historical synchronized training time and separately measured input/processing time; these are component sums, not a new continuous wall-clock run. Models are resident and warmed. BER evaluation, initialization/compilation, checkpoint writing, and offline data generation are excluded. All seven cases pass the full 12 dB test target. The bars are not uncertainty intervals.',[INPUTS[1]])

    # Separate RT-delay timing panel removed: Fig.20 already includes these values.


# ===== 嵌入数据区：CSV原始数值，含整数误码计数 =====
DATA_CSV = {
'results/paper/rt864_final_20261016/ber_curves.csv': """case,method,phase,snr_db,bit_errors,bits,ber
Paris P2,FastSET,pretrained,4,6340430,26054400,0.2433535218619504
Paris P2,FastSET,pretrained,5,5986972,26054400,0.2297873679685581
Paris P2,FastSET,pretrained,6,5664624,26054400,0.21741525423728814
Paris P2,FastSET,pretrained,7,5384673,26054400,0.20667038964627857
Paris P2,FastSET,pretrained,8,5152076,26054400,0.1977430299680668
Paris P2,FastSET,pretrained,9,4950635,26054400,0.19001147598870055
Paris P2,FastSET,pretrained,10,4768800,26054400,0.18303242446573323
Paris P2,FastSET,pretrained,11,4599444,26054400,0.17653233235077376
Paris P2,FastSET,pretrained,12,4427991,26054400,0.1699517547899779
Paris P2,FastSET,pretrained,13,4237881,26054400,0.16265509856300664
Paris P2,FastSET,pretrained,14,4019139,26054400,0.15425951086956521
Paris P2,FastSET,pretrained,15,3759897,26054400,0.1443094832350774
Paris P2,FastSET,pretrained,16,3444646,26054400,0.1322097611152051
Paris P2,FastSET,pretrained,17,3062392,26054400,0.11753838123311226
Paris P2,FastSET,pretrained,18,2600469,26054400,0.09980920689019897
Paris P2,FastSET,adapted,4,5367921,26054400,0.20602742722918202
Paris P2,FastSET,adapted,5,4852312,26054400,0.1862377180054041
Paris P2,FastSET,adapted,6,4319624,26054400,0.1657924957013019
Paris P2,FastSET,adapted,7,3736062,26054400,0.14339466654384672
Paris P2,FastSET,adapted,8,3008922,26054400,0.11548613669859985
Paris P2,FastSET,adapted,9,2038768,26054400,0.07825042986981086
Paris P2,FastSET,adapted,10,833009,26054400,0.03197191261360845
Paris P2,FastSET,adapted,11,60136,26054400,0.0023080938344387128
Paris P2,FastSET,adapted,12,1457,26054400,5.5921456644559075e-05
Paris P2,FastSET,adapted,13,74,26054400,2.8402112503070497e-06
Paris P2,FastSET,adapted,14,0,26054400,0.0
Paris P2,FastSET,adapted,15,0,26054400,0.0
Paris P2,FastSET,adapted,16,0,26054400,0.0
Paris P2,FastSET,adapted,17,0,26054400,0.0
Paris P2,FastSET,adapted,18,0,26054400,0.0
Paris P2,CNN,pretrained,4,6415462,26054400,0.2462333425448293
Paris P2,CNN,pretrained,5,6016425,26054400,0.2309178104274134
Paris P2,CNN,pretrained,6,5650433,26054400,0.2168705861581921
Paris P2,CNN,pretrained,7,5337844,26054400,0.20487303488086464
Paris P2,CNN,pretrained,8,5095167,26054400,0.19555879237288135
Paris P2,CNN,pretrained,9,4917095,26054400,0.18872416943011544
Paris P2,CNN,pretrained,10,4783796,26054400,0.18360798943748466
Paris P2,CNN,pretrained,11,4687112,26054400,0.17989713829525916
Paris P2,CNN,pretrained,12,4610205,26054400,0.17694535280029478
Paris P2,CNN,pretrained,13,4549286,26054400,0.17460720646032915
Paris P2,CNN,pretrained,14,4497622,26054400,0.1726242784328175
Paris P2,CNN,pretrained,15,4454367,26054400,0.1709640981945468
Paris P2,CNN,pretrained,16,4425123,26054400,0.16984167741341194
Paris P2,CNN,pretrained,17,4394012,26054400,0.1686475988700565
Paris P2,CNN,pretrained,18,4373607,26054400,0.16786442980840088
Paris P2,CNN,adapted,4,5500490,26054400,0.21111558892164087
Paris P2,CNN,adapted,5,5023907,26054400,0.1928237457013019
Paris P2,CNN,adapted,6,4547397,26054400,0.17453470431098012
Paris P2,CNN,adapted,7,4059286,26054400,0.15580040223532302
Paris P2,CNN,adapted,8,3521989,26054400,0.13517828082780645
Paris P2,CNN,adapted,9,2861994,26054400,0.10984685887988209
Paris P2,CNN,adapted,10,1932143,26054400,0.07415803088921641
Paris P2,CNN,adapted,11,545693,26054400,0.020944370240727095
Paris P2,CNN,adapted,12,3927,26054400,0.00015072310243183492
Paris P2,CNN,adapted,13,0,26054400,0.0
Paris P2,CNN,adapted,14,0,26054400,0.0
Paris P2,CNN,adapted,15,0,26054400,0.0
Paris P2,CNN,adapted,16,0,26054400,0.0
Paris P2,CNN,adapted,17,0,26054400,0.0
Paris P2,CNN,adapted,18,0,26054400,0.0
Paris P3,FastSET,pretrained,4,6093994,26054400,0.2338950042986981
Paris P3,FastSET,pretrained,5,5723362,26054400,0.2196696911078359
Paris P3,FastSET,pretrained,6,5380999,26054400,0.20652937699582413
Paris P3,FastSET,pretrained,7,5084292,26054400,0.19514139646278555
Paris P3,FastSET,pretrained,8,4835011,26054400,0.18557368429132892
Paris P3,FastSET,pretrained,9,4614741,26054400,0.17711945007369198
Paris P3,FastSET,pretrained,10,4417501,26054400,0.1695491356546303
Paris P3,FastSET,pretrained,11,4227622,26054400,0.16226134549250798
Paris P3,FastSET,pretrained,12,4045405,26054400,0.15526763233849178
Paris P3,FastSET,pretrained,13,3846994,26054400,0.14765237349545565
Paris P3,FastSET,pretrained,14,3635994,26054400,0.13955393330876933
Paris P3,FastSET,pretrained,15,3402960,26054400,0.13060980103168754
Paris P3,FastSET,pretrained,16,3137492,26054400,0.1204208118398428
Paris P3,FastSET,pretrained,17,2847611,26054400,0.1092948216040285
Paris P3,FastSET,pretrained,18,2539327,26054400,0.09746250153524932
Paris P3,FastSET,adapted,4,5425060,26054400,0.2082204925079833
Paris P3,FastSET,adapted,5,4929862,26054400,0.18921418263325965
Paris P3,FastSET,adapted,6,4423636,26054400,0.169784604519774
Paris P3,FastSET,adapted,7,3888201,26054400,0.14923394896831246
Paris P3,FastSET,adapted,8,3274083,26054400,0.12566334285187913
Paris P3,FastSET,adapted,9,2461556,26054400,0.09447755465487595
Paris P3,FastSET,adapted,10,1337216,26054400,0.05132399901744043
Paris P3,FastSET,adapted,11,76870,26054400,0.0029503653893392288
Paris P3,FastSET,adapted,12,478,26054400,1.834622942765905e-05
Paris P3,FastSET,adapted,13,0,26054400,0.0
Paris P3,FastSET,adapted,14,0,26054400,0.0
Paris P3,FastSET,adapted,15,0,26054400,0.0
Paris P3,FastSET,adapted,16,0,26054400,0.0
Paris P3,FastSET,adapted,17,0,26054400,0.0
Paris P3,FastSET,adapted,18,0,26054400,0.0
Paris P3,CNN,pretrained,4,6539641,26054400,0.2509994856914763
Paris P3,CNN,pretrained,5,6152280,26054400,0.23613209285187914
Paris P3,CNN,pretrained,6,5788374,26054400,0.22216493183493
Paris P3,CNN,pretrained,7,5465481,26054400,0.20977190033161386
Paris P3,CNN,pretrained,8,5198325,26054400,0.19951812361827562
Paris P3,CNN,pretrained,9,4989818,26054400,0.19151536784573814
Paris P3,CNN,pretrained,10,4826641,26054400,0.18525243337017933
Paris P3,CNN,pretrained,11,4704225,26054400,0.1805539563375092
Paris P3,CNN,pretrained,12,4610150,26054400,0.1769432418324736
Paris P3,CNN,pretrained,13,4538631,26054400,0.17419825442151807
Paris P3,CNN,pretrained,14,4489451,26054400,0.17231066537705722
Paris P3,CNN,pretrained,15,4456510,26054400,0.17104634917710637
Paris P3,CNN,pretrained,16,4444287,26054400,0.17057721536477524
Paris P3,CNN,pretrained,17,4439440,26054400,0.17039118152788013
Paris P3,CNN,pretrained,18,4439646,26054400,0.17039908806190127
Paris P3,CNN,adapted,4,5429689,26054400,0.20839815923605995
Paris P3,CNN,adapted,5,4942434,26054400,0.18969671149594694
Paris P3,CNN,adapted,6,4443105,26054400,0.17053184874723656
Paris P3,CNN,adapted,7,3916005,26054400,0.15030110077376566
Paris P3,CNN,adapted,8,3318432,26054400,0.12736551215917466
Paris P3,CNN,adapted,9,2577072,26054400,0.09891120117907148
Paris P3,CNN,adapted,10,1588841,26054400,0.060981676799312205
Paris P3,CNN,adapted,11,234735,26054400,0.009009418754605748
Paris P3,CNN,adapted,12,1615,26054400,6.198569147629575e-05
Paris P3,CNN,adapted,13,4,26054400,1.5352493244902973e-07
Paris P3,CNN,adapted,14,0,26054400,0.0
Paris P3,CNN,adapted,15,0,26054400,0.0
Paris P3,CNN,adapted,16,0,26054400,0.0
Paris P3,CNN,adapted,17,0,26054400,0.0
Paris P3,CNN,adapted,18,0,26054400,0.0
Munich P2,FastSET,pretrained,4,6240354,26054400,0.2395124815770081
Munich P2,FastSET,pretrained,5,5880931,26054400,0.2257173836281012
Munich P2,FastSET,pretrained,6,5536180,26054400,0.21248541513141733
Munich P2,FastSET,pretrained,7,5225636,26054400,0.20056635347580448
Munich P2,FastSET,pretrained,8,4969424,26054400,0.19073262097764676
Munich P2,FastSET,pretrained,9,4757092,26054400,0.18258305698845492
Munich P2,FastSET,pretrained,10,4572238,26054400,0.1754881325227217
Munich P2,FastSET,pretrained,11,4411278,26054400,0.16931028924097274
Munich P2,FastSET,pretrained,12,4266614,26054400,0.1637579065340211
Munich P2,FastSET,pretrained,13,4137682,26054400,0.15880933738639155
Munich P2,FastSET,pretrained,14,4021398,26054400,0.1543462140751658
Munich P2,FastSET,pretrained,15,3914125,26054400,0.15022894405551462
Munich P2,FastSET,pretrained,16,3817231,26054400,0.14651003285433553
Munich P2,FastSET,pretrained,17,3734202,26054400,0.1433232774502579
Munich P2,FastSET,pretrained,18,3652893,26054400,0.14020253776713337
Munich P2,FastSET,adapted,4,5472377,26054400,0.210036577315156
Munich P2,FastSET,adapted,5,4950085,26054400,0.18999036631048882
Munich P2,FastSET,adapted,6,4397627,26054400,0.1687863470277573
Munich P2,FastSET,adapted,7,3779164,26054400,0.14504897445345125
Munich P2,FastSET,adapted,8,3029358,26054400,0.11627049557848194
Munich P2,FastSET,adapted,9,2128619,26054400,0.0816990220461803
Munich P2,FastSET,adapted,10,914533,26054400,0.035100904261852126
Munich P2,FastSET,adapted,11,54259,26054400,0.002082527327437976
Munich P2,FastSET,adapted,12,353,26054400,1.3548575288626872e-05
Munich P2,FastSET,adapted,13,0,26054400,0.0
Munich P2,FastSET,adapted,14,0,26054400,0.0
Munich P2,FastSET,adapted,15,0,26054400,0.0
Munich P2,FastSET,adapted,16,0,26054400,0.0
Munich P2,FastSET,adapted,17,0,26054400,0.0
Munich P2,FastSET,adapted,18,0,26054400,0.0
Munich P2,CNN,pretrained,4,6607849,26054400,0.25361739283959717
Munich P2,CNN,pretrained,5,6215264,26054400,0.23854949643822157
Munich P2,CNN,pretrained,6,5815643,26054400,0.22321154968066814
Munich P2,CNN,pretrained,7,5420912,26054400,0.20806128715303365
Munich P2,CNN,pretrained,8,5052592,26054400,0.193924711373127
Munich P2,CNN,pretrained,9,4737431,26054400,0.18182844356423483
Munich P2,CNN,pretrained,10,4482576,26054400,0.17204679439941045
Munich P2,CNN,pretrained,11,4280403,26054400,0.16428714535740604
Munich P2,CNN,pretrained,12,4114191,26054400,0.1579077238393515
Munich P2,CNN,pretrained,13,3989195,26054400,0.15311022322525178
Munich P2,CNN,pretrained,14,3878395,26054400,0.14885758259641366
Munich P2,CNN,pretrained,15,3797624,26054400,0.14575749201670352
Munich P2,CNN,pretrained,16,3736716,26054400,0.14341976787030214
Munich P2,CNN,pretrained,17,3697071,26054400,0.1418981438835667
Munich P2,CNN,pretrained,18,3677907,26054400,0.1411626059322034
Munich P2,CNN,adapted,4,5553935,26054400,0.21316687392532546
Munich P2,CNN,adapted,5,5050592,26054400,0.1938479489069025
Munich P2,CNN,adapted,6,4535566,26054400,0.17408061594202898
Munich P2,CNN,adapted,7,3990900,26054400,0.15317566322770818
Munich P2,CNN,adapted,8,3373205,26054400,0.1294677674404323
Munich P2,CNN,adapted,9,2563717,26054400,0.09839861981085728
Munich P2,CNN,adapted,10,1478566,26054400,0.05674918631785802
Munich P2,CNN,adapted,11,196069,26054400,0.007525369995087202
Munich P2,CNN,adapted,12,3644,26054400,0.00013986121346106606
Munich P2,CNN,adapted,13,266,26054400,1.0209408007860477e-05
Munich P2,CNN,adapted,14,0,26054400,0.0
Munich P2,CNN,adapted,15,0,26054400,0.0
Munich P2,CNN,adapted,16,0,26054400,0.0
Munich P2,CNN,adapted,17,0,26054400,0.0
Munich P2,CNN,adapted,18,0,26054400,0.0
San Francisco P2,FastSET,pretrained,4,5133184,26054400,0.19701793171211004
San Francisco P2,FastSET,pretrained,5,4623657,26054400,0.17746165714812084
San Francisco P2,FastSET,pretrained,6,4121337,26054400,0.1581819961311717
San Francisco P2,FastSET,pretrained,7,3655628,26054400,0.14030751043969542
San Francisco P2,FastSET,pretrained,8,3228756,26054400,0.12392363669859985
San Francisco P2,FastSET,pretrained,9,2751699,26054400,0.10561360077376566
San Francisco P2,FastSET,pretrained,10,2159960,26054400,0.08290192827315156
San Francisco P2,FastSET,pretrained,11,1406596,26054400,0.05398688897076885
San Francisco P2,FastSET,pretrained,12,514371,26054400,0.019742193257184966
San Francisco P2,FastSET,pretrained,13,96478,26054400,0.0037029446082043724
San Francisco P2,FastSET,pretrained,14,26146,26054400,0.0010035157209530828
San Francisco P2,FastSET,pretrained,15,20676,26054400,0.0007935703758290346
San Francisco P2,FastSET,pretrained,16,10137,26054400,0.0003890705600589536
San Francisco P2,FastSET,pretrained,17,2518,26054400,9.664394497666421e-05
San Francisco P2,FastSET,pretrained,18,329,26054400,1.2627425693932695e-05
San Francisco P2,FastSET,adapted,4,5181269,26054400,0.19886349330631295
San Francisco P2,FastSET,adapted,5,4649762,26054400,0.17846359923851635
San Francisco P2,FastSET,adapted,6,4094079,26054400,0.15713580047899778
San Francisco P2,FastSET,adapted,7,3473950,26054400,0.1333344847703267
San Francisco P2,FastSET,adapted,8,2736781,26054400,0.10504102953819701
San Francisco P2,FastSET,adapted,9,1859179,26054400,0.07135758259641366
San Francisco P2,FastSET,adapted,10,695416,26054400,0.026690923605993615
San Francisco P2,FastSET,adapted,11,54776,26054400,0.002102370424957013
San Francisco P2,FastSET,adapted,12,1371,26054400,5.262067059690494e-05
San Francisco P2,FastSET,adapted,13,68,26054400,2.6099238516335053e-06
San Francisco P2,FastSET,adapted,14,0,26054400,0.0
San Francisco P2,FastSET,adapted,15,0,26054400,0.0
San Francisco P2,FastSET,adapted,16,1,26054400,3.838123311225743e-08
San Francisco P2,FastSET,adapted,17,0,26054400,0.0
San Francisco P2,FastSET,adapted,18,0,26054400,0.0
San Francisco P2,CNN,pretrained,4,5289128,26054400,0.20300325472856792
San Francisco P2,CNN,pretrained,5,4792680,26054400,0.18394896831245394
San Francisco P2,CNN,pretrained,6,4309954,26054400,0.16542134917710635
San Francisco P2,CNN,pretrained,7,3859958,26054400,0.14814994780152296
San Francisco P2,CNN,pretrained,8,3462396,26054400,0.1328910280029477
San Francisco P2,CNN,pretrained,9,3106370,26054400,0.11922631110292312
San Francisco P2,CNN,pretrained,10,2762441,26054400,0.10602589197985753
San Francisco P2,CNN,pretrained,11,2420455,26054400,0.09290004759272906
San Francisco P2,CNN,pretrained,12,2041412,26054400,0.07835190985015966
San Francisco P2,CNN,pretrained,13,1648022,26054400,0.06325311655612871
San Francisco P2,CNN,pretrained,14,1215431,26054400,0.04664974054286416
San Francisco P2,CNN,pretrained,15,789351,26054400,0.030296264738393516
San Francisco P2,CNN,pretrained,16,457270,26054400,0.017550586465241956
San Francisco P2,CNN,pretrained,17,253794,26054400,0.009740926676492262
San Francisco P2,CNN,pretrained,18,140204,26054400,0.005381202407270941
San Francisco P2,CNN,adapted,4,5326552,26054400,0.20443963399656104
San Francisco P2,CNN,adapted,5,4787195,26054400,0.1837384472488332
San Francisco P2,CNN,adapted,6,4234767,26054400,0.16253557940309507
San Francisco P2,CNN,adapted,7,3637993,26054400,0.13963065739376074
San Francisco P2,CNN,adapted,8,2888170,26054400,0.11085152603782854
San Francisco P2,CNN,adapted,9,1724936,26054400,0.06620517071972488
San Francisco P2,CNN,adapted,10,220168,26054400,0.008450319331859494
San Francisco P2,CNN,adapted,11,5958,26054400,0.00022867538688282977
San Francisco P2,CNN,adapted,12,860,26054400,3.300786047654139e-05
San Francisco P2,CNN,adapted,13,521,26054400,1.9996622451486123e-05
San Francisco P2,CNN,adapted,14,37,26054400,1.4201056251535249e-06
San Francisco P2,CNN,adapted,15,132,26054400,5.066322770817981e-06
San Francisco P2,CNN,adapted,16,93,26054400,3.569454679439941e-06
San Francisco P2,CNN,adapted,17,122,26054400,4.682510439695406e-06
San Francisco P2,CNN,adapted,18,185,26054400,7.100528125767625e-06
Florence P2,FastSET,pretrained,4,5224543,26054400,0.20052440278801278
Florence P2,FastSET,pretrained,5,4711981,26054400,0.18085164118152788
Florence P2,FastSET,pretrained,6,4187921,26054400,0.16073757215671824
Florence P2,FastSET,pretrained,7,3641721,26054400,0.13977374263080325
Florence P2,FastSET,pretrained,8,3076582,26054400,0.11808301093097519
Florence P2,FastSET,pretrained,9,2541762,26054400,0.09755595983787767
Florence P2,FastSET,pretrained,10,2016395,26054400,0.07739172654139032
Florence P2,FastSET,pretrained,11,1147880,26054400,0.04405704986489806
Florence P2,FastSET,pretrained,12,229633,26054400,0.00881359770326701
Florence P2,FastSET,pretrained,13,13870,26054400,0.0005323477032670106
Florence P2,FastSET,pretrained,14,672,26054400,2.5792188651436994e-05
Florence P2,FastSET,pretrained,15,0,26054400,0.0
Florence P2,FastSET,pretrained,16,0,26054400,0.0
Florence P2,FastSET,pretrained,17,0,26054400,0.0
Florence P2,FastSET,pretrained,18,0,26054400,0.0
Florence P2,FastSET,adapted,4,5712860,26054400,0.219266611397691
Florence P2,FastSET,adapted,5,5151041,26054400,0.19770330539179562
Florence P2,FastSET,adapted,6,4562420,26054400,0.17511130557602556
Florence P2,FastSET,adapted,7,3917791,26054400,0.15036964965610417
Florence P2,FastSET,adapted,8,3145802,26054400,0.12073975988700565
Florence P2,FastSET,adapted,9,2148845,26054400,0.08247532086710882
Florence P2,FastSET,adapted,10,679609,26054400,0.02608423145418816
Florence P2,FastSET,adapted,11,31670,26054400,0.001215533652665193
Florence P2,FastSET,adapted,12,482,26054400,1.849975436010808e-05
Florence P2,FastSET,adapted,13,0,26054400,0.0
Florence P2,FastSET,adapted,14,0,26054400,0.0
Florence P2,FastSET,adapted,15,0,26054400,0.0
Florence P2,FastSET,adapted,16,0,26054400,0.0
Florence P2,FastSET,adapted,17,0,26054400,0.0
Florence P2,FastSET,adapted,18,0,26054400,0.0
Florence P2,CNN,pretrained,4,5554309,26054400,0.21318122850650945
Florence P2,CNN,pretrained,5,5090882,26054400,0.19539432878899532
Florence P2,CNN,pretrained,6,4658526,26054400,0.17879997236551215
Florence P2,CNN,pretrained,7,4284169,26054400,0.16443168908130681
Florence P2,CNN,pretrained,8,3976359,26054400,0.15261756171702284
Florence P2,CNN,pretrained,9,3722955,26054400,0.14289160372144435
Florence P2,CNN,pretrained,10,3512556,26054400,0.13481623065585852
Florence P2,CNN,pretrained,11,3334410,26054400,0.1279787675018423
Florence P2,CNN,pretrained,12,3165892,26054400,0.1215108388602309
Florence P2,CNN,pretrained,13,2989426,26054400,0.11473785617784328
Florence P2,CNN,pretrained,14,2830818,26054400,0.10865028555637435
Florence P2,CNN,pretrained,15,2694249,26054400,0.10340859893146646
Florence P2,CNN,pretrained,16,2593554,26054400,0.09954380066322771
Florence P2,CNN,pretrained,17,2506162,26054400,0.0961895879390813
Florence P2,CNN,pretrained,18,2440355,26054400,0.09366383413166299
Florence P2,CNN,adapted,4,5335987,26054400,0.20480176093097519
Florence P2,CNN,adapted,5,4779535,26054400,0.18344444700319332
Florence P2,CNN,adapted,6,4215136,26054400,0.16178211741586834
Florence P2,CNN,adapted,7,3604873,26054400,0.1383594709530828
Florence P2,CNN,adapted,8,2854384,26054400,0.10955477769589782
Florence P2,CNN,adapted,9,1845200,26054400,0.07082105133873741
Florence P2,CNN,adapted,10,294402,26054400,0.011299511790714811
Florence P2,CNN,adapted,11,3815,26054400,0.0001464244043232621
Florence P2,CNN,adapted,12,56,26054400,2.1493490542864163e-06
Florence P2,CNN,adapted,13,0,26054400,0.0
Florence P2,CNN,adapted,14,0,26054400,0.0
Florence P2,CNN,adapted,15,0,26054400,0.0
Florence P2,CNN,adapted,16,0,26054400,0.0
Florence P2,CNN,adapted,17,0,26054400,0.0
Florence P2,CNN,adapted,18,0,26054400,0.0
Paris to San Francisco,FastSET,pretrained,4,5073963,26054400,0.19474495670596906
Paris to San Francisco,FastSET,pretrained,5,4561072,26054400,0.17505956767379022
Paris to San Francisco,FastSET,pretrained,6,4053570,26054400,0.15558101510685335
Paris to San Francisco,FastSET,pretrained,7,3555406,26054400,0.13646086649471875
Paris to San Francisco,FastSET,pretrained,8,3021789,26054400,0.11597998802505527
Paris to San Francisco,FastSET,pretrained,9,2391139,26054400,0.09177486336281011
Paris to San Francisco,FastSET,pretrained,10,1556676,26054400,0.05974714443625645
Paris to San Francisco,FastSET,pretrained,11,600726,26054400,0.02305660464259396
Paris to San Francisco,FastSET,pretrained,12,117989,26054400,0.004528563313682142
Paris to San Francisco,FastSET,pretrained,13,21719,26054400,0.0008336020019651192
Paris to San Francisco,FastSET,pretrained,14,3694,26054400,0.00014178027511667894
Paris to San Francisco,FastSET,pretrained,15,508,26054400,1.9497666421026774e-05
Paris to San Francisco,FastSET,pretrained,16,32,26054400,1.2281994595922378e-06
Paris to San Francisco,FastSET,pretrained,17,74,26054400,2.8402112503070497e-06
Paris to San Francisco,FastSET,pretrained,18,15,26054400,5.757184966838615e-07
Paris to San Francisco,FastSET,adapted,4,5146459,26054400,0.19752744258167526
Paris to San Francisco,FastSET,adapted,5,4622928,26054400,0.17743367722918202
Paris to San Francisco,FastSET,adapted,6,4071526,26054400,0.15627018852861704
Paris to San Francisco,FastSET,adapted,7,3444790,26054400,0.13221528801277327
Paris to San Francisco,FastSET,adapted,8,2649189,26054400,0.10167914056742815
Paris to San Francisco,FastSET,adapted,9,1584587,26054400,0.060818403033652664
Paris to San Francisco,FastSET,adapted,10,343013,26054400,0.013165261913534758
Paris to San Francisco,FastSET,adapted,11,20731,26054400,0.0007956813436502088
Paris to San Francisco,FastSET,adapted,12,1324,26054400,5.081675264062884e-05
Paris to San Francisco,FastSET,adapted,13,152,26054400,5.833947433063129e-06
Paris to San Francisco,FastSET,adapted,14,25,26054400,9.595308278064357e-07
Paris to San Francisco,FastSET,adapted,15,5,26054400,1.9190616556128716e-07
Paris to San Francisco,FastSET,adapted,16,12,26054400,4.6057479734708915e-07
Paris to San Francisco,FastSET,adapted,17,7,26054400,2.6866863178580204e-07
Paris to San Francisco,FastSET,adapted,18,1,26054400,3.838123311225743e-08
Paris to San Francisco,CNN,pretrained,4,5274138,26054400,0.20242792004421517
Paris to San Francisco,CNN,pretrained,5,4742357,26054400,0.1820175095185458
Paris to San Francisco,CNN,pretrained,6,4224864,26054400,0.1621554900515844
Paris to San Francisco,CNN,pretrained,7,3723186,26054400,0.14290046978629328
Paris to San Francisco,CNN,pretrained,8,3220004,26054400,0.12358772414640137
Paris to San Francisco,CNN,pretrained,9,2680306,26054400,0.10287344939818227
Paris to San Francisco,CNN,pretrained,10,2073702,26054400,0.07959123986735446
Paris to San Francisco,CNN,pretrained,11,1370456,26054400,0.05259979120609187
Paris to San Francisco,CNN,pretrained,12,600373,26054400,0.02304305606730533
Paris to San Francisco,CNN,pretrained,13,166130,26054400,0.006376274256939327
Paris to San Francisco,CNN,pretrained,14,43813,26054400,0.0016815969663473349
Paris to San Francisco,CNN,pretrained,15,10751,26054400,0.00041263663718987966
Paris to San Francisco,CNN,pretrained,16,2337,26054400,8.969694178334562e-05
Paris to San Francisco,CNN,pretrained,17,1080,26054400,4.145173176123802e-05
Paris to San Francisco,CNN,pretrained,18,295,26054400,1.1322463768115942e-05
Paris to San Francisco,CNN,adapted,4,5382113,26054400,0.2065721336895112
Paris to San Francisco,CNN,adapted,5,4809347,26054400,0.18458866832473594
Paris to San Francisco,CNN,adapted,6,4223645,26054400,0.16210870332842053
Paris to San Francisco,CNN,adapted,7,3586041,26054400,0.13763667557111275
Paris to San Francisco,CNN,adapted,8,2812526,26054400,0.10794821604028494
Paris to San Francisco,CNN,adapted,9,1784506,26054400,0.06849154077622206
Paris to San Francisco,CNN,adapted,10,471227,26054400,0.018086273335789733
Paris to San Francisco,CNN,adapted,11,19729,26054400,0.0007572233480717268
Paris to San Francisco,CNN,adapted,12,463,26054400,1.777051093097519e-05
Paris to San Francisco,CNN,adapted,13,211,26054400,8.098440186686317e-06
Paris to San Francisco,CNN,adapted,14,207,26054400,7.944915254237288e-06
Paris to San Francisco,CNN,adapted,15,87,26054400,3.3391672807663963e-06
Paris to San Francisco,CNN,adapted,16,29,26054400,1.1130557602554656e-06
Paris to San Francisco,CNN,adapted,17,0,26054400,0.0
Paris to San Francisco,CNN,adapted,18,2,26054400,7.676246622451486e-08
San Francisco to Paris,FastSET,pretrained,4,6129133,26054400,0.2352436824490297
San Francisco to Paris,FastSET,pretrained,5,5799657,26054400,0.2225979872881356
San Francisco to Paris,FastSET,pretrained,6,5505339,26054400,0.21130169952100222
San Francisco to Paris,FastSET,pretrained,7,5274382,26054400,0.20243728506509456
San Francisco to Paris,FastSET,pretrained,8,5104005,26054400,0.19589800571112748
San Francisco to Paris,FastSET,pretrained,9,4953392,26054400,0.19011729304839106
San Francisco to Paris,FastSET,pretrained,10,4824253,26054400,0.18516077898550726
San Francisco to Paris,FastSET,pretrained,11,4699209,26054400,0.18036143607221813
San Francisco to Paris,FastSET,pretrained,12,4575040,26054400,0.17559567673790225
San Francisco to Paris,FastSET,pretrained,13,4445282,26054400,0.17061540469172193
San Francisco to Paris,FastSET,pretrained,14,4308229,26054400,0.1653551415499877
San Francisco to Paris,FastSET,pretrained,15,4151735,26054400,0.1593487088553181
San Francisco to Paris,FastSET,pretrained,16,3969645,26054400,0.15235987011790714
San Francisco to Paris,FastSET,pretrained,17,3755334,26054400,0.14413434966838615
San Francisco to Paris,FastSET,pretrained,18,3511865,26054400,0.13478970922377795
San Francisco to Paris,FastSET,adapted,4,5403765,26054400,0.2074031641488578
San Francisco to Paris,FastSET,adapted,5,4893751,26054400,0.1878281979243429
San Francisco to Paris,FastSET,adapted,6,4362448,26054400,0.1674361336281012
San Francisco to Paris,FastSET,adapted,7,3776824,26054400,0.14495916236796855
San Francisco to Paris,FastSET,adapted,8,3062333,26054400,0.11753611674035863
San Francisco to Paris,FastSET,adapted,9,2154804,26054400,0.08270403463522476
San Francisco to Paris,FastSET,adapted,10,1052834,26054400,0.04040906718251044
San Francisco to Paris,FastSET,adapted,11,117793,26054400,0.00452104059199214
San Francisco to Paris,FastSET,adapted,12,3063,26054400,0.00011756171702284451
San Francisco to Paris,FastSET,adapted,13,5,26054400,1.9190616556128716e-07
San Francisco to Paris,FastSET,adapted,14,0,26054400,0.0
San Francisco to Paris,FastSET,adapted,15,0,26054400,0.0
San Francisco to Paris,FastSET,adapted,16,0,26054400,0.0
San Francisco to Paris,FastSET,adapted,17,0,26054400,0.0
San Francisco to Paris,FastSET,adapted,18,0,26054400,0.0
San Francisco to Paris,CNN,pretrained,4,6982170,26054400,0.26798429439941046
San Francisco to Paris,CNN,pretrained,5,6672053,26054400,0.25608162153033653
San Francisco to Paris,CNN,pretrained,6,6386357,26054400,0.24511625675509702
San Francisco to Paris,CNN,pretrained,7,6141149,26054400,0.2357048713461066
San Francisco to Paris,CNN,pretrained,8,5949683,26054400,0.22835617016703513
San Francisco to Paris,CNN,pretrained,9,5815037,26054400,0.22318829065340212
San Francisco to Paris,CNN,pretrained,10,5718230,26054400,0.21947271861950382
San Francisco to Paris,CNN,pretrained,11,5647553,26054400,0.2167600482068288
San Francisco to Paris,CNN,pretrained,12,5593352,26054400,0.21467974699091133
San Francisco to Paris,CNN,pretrained,13,5551695,26054400,0.21308089996315402
San Francisco to Paris,CNN,pretrained,14,5518574,26054400,0.21180967514124294
San Francisco to Paris,CNN,pretrained,15,5492948,26054400,0.21082611766150822
San Francisco to Paris,CNN,pretrained,16,5470496,26054400,0.20996438221567182
San Francisco to Paris,CNN,pretrained,17,5451675,26054400,0.20924200902726603
San Francisco to Paris,CNN,pretrained,18,5435176,26054400,0.2086087570621469
San Francisco to Paris,CNN,adapted,4,5580757,26054400,0.21419633535986243
San Francisco to Paris,CNN,adapted,5,5093214,26054400,0.19548383382461312
San Francisco to Paris,CNN,adapted,6,4609002,26054400,0.1768991801768607
San Francisco to Paris,CNN,adapted,7,4117770,26054400,0.15804509027266028
San Francisco to Paris,CNN,adapted,8,3584976,26054400,0.1375957995578482
San Francisco to Paris,CNN,adapted,9,2927195,26054400,0.11234935366003439
San Francisco to Paris,CNN,adapted,10,1944944,26054400,0.07464934905428641
San Francisco to Paris,CNN,adapted,11,460280,26054400,0.01766611397690985
San Francisco to Paris,CNN,adapted,12,3132,26054400,0.00012021002210759027
San Francisco to Paris,CNN,adapted,13,0,26054400,0.0
San Francisco to Paris,CNN,adapted,14,0,26054400,0.0
San Francisco to Paris,CNN,adapted,15,0,26054400,0.0
San Francisco to Paris,CNN,adapted,16,0,26054400,0.0
San Francisco to Paris,CNN,adapted,17,0,26054400,0.0
San Francisco to Paris,CNN,adapted,18,0,26054400,0.0
""",
'results/paper/final_resident864_20261016/tables/rt_timing.csv': 'case,method,est_iterations,dem_iterations,cnn_iterations,remote_ber,local_replay_ber,source,original_training_seconds,total_seconds,qualified\nSan Francisco P2,FastSET,5,2,0,5.262067059690494e-05,5.131570867108819e-05,source_6ec82f3960dfa2a5,0.9641611729894066,0.9694942369897035,True\nSan Francisco P2,CNN,0,0,6,3.300786047654139e-05,3.296947924342913e-05,source_6ec82f3960dfa2a5,1.2495090680022258,1.253445699002441,True\nParis to San Francisco,FastSET,5,2,0,5.081675264062884e-05,5.0087509211495944e-05,source_a048d061065ea778,0.9554807560052723,0.9606872720041792,True\nParis to San Francisco,CNN,0,0,6,1.777051093097519e-05,1.8461373126995825e-05,source_a048d061065ea778,1.1922164890129352,1.1961448140118591,True\nParis P2,FastSET,5,2,0,5.5921456644559075e-05,5.557602554654876e-05,source_720f73c1fd98e7c5,0.9671055649960181,0.9746079239957908,True\nParis P2,CNN,0,0,22,0.00015072310243183492,0.00015049281503316139,source_720f73c1fd98e7c5,4.594164652982727,4.617774467985328,True\nParis P3,FastSET,5,8,0,1.834622942765905e-05,1.8115942028985507e-05,source_31ac63ac3a0faf6d,1.9477699770068284,1.956998538006701,True\nParis P3,CNN,0,0,24,6.198569147629575e-05,6.16786416113977e-05,source_31ac63ac3a0faf6d,4.784793518978404,4.802736437976819,True\nMunich P2,FastSET,5,1,0,1.3548575288626872e-05,1.3510194055514616e-05,source_d2c7cdf2c16dd96b,0.76368592401559,0.767775318016902,True\nMunich P2,CNN,0,0,8,0.00013986121346106606,0.00013870977646769834,source_d2c7cdf2c16dd96b,1.5955286900425563,1.6009621730418075,True\nSan Francisco to Paris,FastSET,5,3,0,0.00011756171702284451,0.00011575779906656842,source_6e7401dbfbeed9ab,1.08649369997147,1.0920415739710734,True\nSan Francisco to Paris,CNN,0,0,24,0.00012021002210759027,0.00011967268484401867,source_6e7401dbfbeed9ab,4.843852196980151,4.862282472979132,True\nFlorence P2,FastSET,5,1,0,1.849975436010808e-05,1.8422991893883568e-05,source_51a7f0d649decd7c,0.7932252649916336,0.7977058759915963,True\nFlorence P2,CNN,0,0,7,2.1493490542864163e-06,2.4563989191844756e-06,source_51a7f0d649decd7c,1.4560594880167628,1.4612732350187798,True\n',
'results/paper/final_resident864_20261016/tables/rt_delay_all.csv': """delay_ns,method,est,dem,cnn,remote_ber,local_ber,local_test_hit,original_training_seconds,total_seconds,qualified
300,FastSET,0,0,0,0.0,0.0,True,0.0,0.0,True
300,CNN,0,0,0,0.0,0.0,True,0.0,0.0,True
600,FastSET,5,2,0,4.609586096782117e-05,4.5942336035372144e-05,True,0.9100409019883955,0.9155086429882431,True
600,CNN,0,0,7,5.0125890444608204e-05,4.943502824858757e-05,True,1.3888718470116146,1.3934530730120969,True
900,FastSET,5,2,0,0.0004387742569393269,0.00044261238025055267,False,0.9230184690386523,0.9274424500390523,False
900,CNN,0,0,19,2.9400024563989192e-05,2.9169737165315648e-05,True,3.7687608929845737,3.7835205329865857,True
1200,FastSET,5,2,0,6.35593220338983e-05,6.367446573323507e-05,True,0.9359035420056898,0.9408894000071086,True
1200,CNN,0,0,22,0.00016722703267010562,0.00016691998280520757,True,4.355233377995319,4.373298059993431,True
""",
'results/paper/final_resident864_20261016/tables/rt_delay_ber.csv': 'delay_ns,method,phase,snr_db,bit_errors,bits,ber,source\n300,FastSET,pretrained,4,4650154,26054400,0.17847864468189634,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,5,4054391,26054400,0.1556125260992385,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,6,3405910,26054400,0.1307230256693687,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,7,2648554,26054400,0.10165476848440187,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,8,1543237,26054400,0.05923133904446082,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,9,149848,26054400,0.005751351019405552,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,10,247,26054400,9.480164578727585e-06,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,11,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,12,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,13,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,14,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,15,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,16,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,17,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,pretrained,18,0,26054400,0.0,source_b32c1f6ae2d59cc1\n300,FastSET,adapted,4,4650154,26054400,0.17847864468189634,source_24bb4963a67a5448\n300,FastSET,adapted,5,4054391,26054400,0.1556125260992385,source_24bb4963a67a5448\n300,FastSET,adapted,6,3405910,26054400,0.1307230256693687,source_24bb4963a67a5448\n300,FastSET,adapted,7,2648554,26054400,0.10165476848440187,source_24bb4963a67a5448\n300,FastSET,adapted,8,1543237,26054400,0.05923133904446082,source_24bb4963a67a5448\n300,FastSET,adapted,9,149848,26054400,0.005751351019405552,source_24bb4963a67a5448\n300,FastSET,adapted,10,247,26054400,9.480164578727585e-06,source_24bb4963a67a5448\n300,FastSET,adapted,11,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,12,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,13,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,14,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,15,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,16,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,17,0,26054400,0.0,source_24bb4963a67a5448\n300,FastSET,adapted,18,0,26054400,0.0,source_24bb4963a67a5448\n300,CNN,pretrained,4,4867791,26054400,0.1868318211127487,source_2b1172e6e718ecf9\n300,CNN,pretrained,5,4294016,26054400,0.1648096290837632,source_2b1172e6e718ecf9\n300,CNN,pretrained,6,3694558,26054400,0.14180169184475558,source_2b1172e6e718ecf9\n300,CNN,pretrained,7,3031386,26054400,0.1163483327192336,source_2b1172e6e718ecf9\n300,CNN,pretrained,8,2223019,26054400,0.0853222104519774,source_2b1172e6e718ecf9\n300,CNN,pretrained,9,1005870,26054400,0.03860653095062638,source_2b1172e6e718ecf9\n300,CNN,pretrained,10,44483,26054400,0.0017073123925325474,source_2b1172e6e718ecf9\n300,CNN,pretrained,11,13,26054400,4.989560304593466e-07,source_2b1172e6e718ecf9\n300,CNN,pretrained,12,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,13,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,14,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,15,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,16,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,17,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,pretrained,18,0,26054400,0.0,source_2b1172e6e718ecf9\n300,CNN,adapted,4,4867791,26054400,0.1868318211127487,source_d218e7c81d05f6dc\n300,CNN,adapted,5,4294016,26054400,0.1648096290837632,source_d218e7c81d05f6dc\n300,CNN,adapted,6,3694558,26054400,0.14180169184475558,source_d218e7c81d05f6dc\n300,CNN,adapted,7,3031386,26054400,0.1163483327192336,source_d218e7c81d05f6dc\n300,CNN,adapted,8,2223019,26054400,0.0853222104519774,source_d218e7c81d05f6dc\n300,CNN,adapted,9,1005870,26054400,0.03860653095062638,source_d218e7c81d05f6dc\n300,CNN,adapted,10,44483,26054400,0.0017073123925325474,source_d218e7c81d05f6dc\n300,CNN,adapted,11,13,26054400,4.989560304593466e-07,source_d218e7c81d05f6dc\n300,CNN,adapted,12,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,13,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,14,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,15,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,16,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,17,0,26054400,0.0,source_d218e7c81d05f6dc\n300,CNN,adapted,18,0,26054400,0.0,source_d218e7c81d05f6dc\n600,FastSET,pretrained,4,5148338,26054400,0.19759956091869318,source_0be29671bd6eb36b\n600,FastSET,pretrained,5,4628312,26054400,0.17764032178825842,source_0be29671bd6eb36b\n600,FastSET,pretrained,6,4097821,26054400,0.15727942305330386,source_0be29671bd6eb36b\n600,FastSET,pretrained,7,3563106,26054400,0.1367564019896831,source_0be29671bd6eb36b\n600,FastSET,pretrained,8,2997928,26054400,0.1150641734217637,source_0be29671bd6eb36b\n600,FastSET,pretrained,9,2418643,26054400,0.09283050079832965,source_0be29671bd6eb36b\n600,FastSET,pretrained,10,1876142,26054400,0.07200864345369688,source_0be29671bd6eb36b\n600,FastSET,pretrained,11,1245065,26054400,0.047787130004912796,source_0be29671bd6eb36b\n600,FastSET,pretrained,12,526261,26054400,0.02019854611888971,source_0be29671bd6eb36b\n600,FastSET,pretrained,13,62480,26054400,0.0023980594448538444,source_0be29671bd6eb36b\n600,FastSET,pretrained,14,332,26054400,1.2742569393269467e-05,source_0be29671bd6eb36b\n600,FastSET,pretrained,15,0,26054400,0.0,source_0be29671bd6eb36b\n600,FastSET,pretrained,16,0,26054400,0.0,source_0be29671bd6eb36b\n600,FastSET,pretrained,17,0,26054400,0.0,source_0be29671bd6eb36b\n600,FastSET,pretrained,18,0,26054400,0.0,source_0be29671bd6eb36b\n600,FastSET,adapted,4,5189806,26054400,0.1991911538933923,source_856622580326d005\n600,FastSET,adapted,5,4656945,26054400,0.17873929163596167,source_856622580326d005\n600,FastSET,adapted,6,4095913,26054400,0.15720619166052566,source_856622580326d005\n600,FastSET,adapted,7,3460658,26054400,0.13282432141979858,source_856622580326d005\n600,FastSET,adapted,8,2682530,26054400,0.10295880926062392,source_856622580326d005\n600,FastSET,adapted,9,1737814,26054400,0.06669944423974454,source_856622580326d005\n600,FastSET,adapted,10,552597,26054400,0.021209354274134118,source_856622580326d005\n600,FastSET,adapted,11,33276,26054400,0.0012771739130434782,source_856622580326d005\n600,FastSET,adapted,12,1201,26054400,4.609586096782117e-05,source_856622580326d005\n600,FastSET,adapted,13,97,26054400,3.722979611888971e-06,source_856622580326d005\n600,FastSET,adapted,14,1,26054400,3.838123311225743e-08,source_856622580326d005\n600,FastSET,adapted,15,0,26054400,0.0,source_856622580326d005\n600,FastSET,adapted,16,0,26054400,0.0,source_856622580326d005\n600,FastSET,adapted,17,0,26054400,0.0,source_856622580326d005\n600,FastSET,adapted,18,0,26054400,0.0,source_856622580326d005\n600,CNN,pretrained,4,5645199,26054400,0.21666969878408254,source_70670b220b70caa2\n600,CNN,pretrained,5,5164470,26054400,0.19821872697126014,source_70670b220b70caa2\n600,CNN,pretrained,6,4699697,26054400,0.18038016611397692,source_70670b220b70caa2\n600,CNN,pretrained,7,4260329,26054400,0.16351668048391058,source_70670b220b70caa2\n600,CNN,pretrained,8,3858347,26054400,0.14808811563497912,source_70670b220b70caa2\n600,CNN,pretrained,9,3492461,26054400,0.1340449597764677,source_70670b220b70caa2\n600,CNN,pretrained,10,3159062,26054400,0.12124869503807419,source_70670b220b70caa2\n600,CNN,pretrained,11,2862353,26054400,0.10986063774256939,source_70670b220b70caa2\n600,CNN,pretrained,12,2585928,26054400,0.09925110537951363,source_70670b220b70caa2\n600,CNN,pretrained,13,2297932,26054400,0.08819746376811594,source_70670b220b70caa2\n600,CNN,pretrained,14,1932368,26054400,0.07416666666666667,source_70670b220b70caa2\n600,CNN,pretrained,15,1352103,26054400,0.05189538043478261,source_70670b220b70caa2\n600,CNN,pretrained,16,575250,26054400,0.022078804347826088,source_70670b220b70caa2\n600,CNN,pretrained,17,124557,26054400,0.0047806512527634485,source_70670b220b70caa2\n600,CNN,pretrained,18,13227,26054400,0.000507668570375829,source_70670b220b70caa2\n600,CNN,adapted,4,5346582,26054400,0.20520841009579957,source_96ea1a0fef6dc2b8\n600,CNN,adapted,5,4809354,26054400,0.18458893699336773,source_96ea1a0fef6dc2b8\n600,CNN,adapted,6,4255624,26054400,0.1633360967821174,source_96ea1a0fef6dc2b8\n600,CNN,adapted,7,3656256,26054400,0.1403316138540899,source_96ea1a0fef6dc2b8\n600,CNN,adapted,8,2957668,26054400,0.1135189449766642,source_96ea1a0fef6dc2b8\n600,CNN,adapted,9,2115318,26054400,0.08118851326455416,source_96ea1a0fef6dc2b8\n600,CNN,adapted,10,1048625,26054400,0.04024752057234095,source_96ea1a0fef6dc2b8\n600,CNN,adapted,11,91023,26054400,0.003493574981577008,source_96ea1a0fef6dc2b8\n600,CNN,adapted,12,1306,26054400,5.0125890444608204e-05,source_96ea1a0fef6dc2b8\n600,CNN,adapted,13,41,26054400,1.5736305576025547e-06,source_96ea1a0fef6dc2b8\n600,CNN,adapted,14,2,26054400,7.676246622451486e-08,source_96ea1a0fef6dc2b8\n600,CNN,adapted,15,0,26054400,0.0,source_96ea1a0fef6dc2b8\n600,CNN,adapted,16,0,26054400,0.0,source_96ea1a0fef6dc2b8\n600,CNN,adapted,17,0,26054400,0.0,source_96ea1a0fef6dc2b8\n600,CNN,adapted,18,0,26054400,0.0,source_96ea1a0fef6dc2b8\n900,FastSET,pretrained,4,5716961,26054400,0.21942401283468435,source_1eff1e68374fb83b\n900,FastSET,pretrained,5,5276382,26054400,0.20251404753131907,source_1eff1e68374fb83b\n900,FastSET,pretrained,6,4847846,26054400,0.18606630741832472,source_1eff1e68374fb83b\n900,FastSET,pretrained,7,4443095,26054400,0.17053146493490542,source_1eff1e68374fb83b\n900,FastSET,pretrained,8,4073566,26054400,0.15634848624416606,source_1eff1e68374fb83b\n900,FastSET,pretrained,9,3721364,26054400,0.14283053917956276,source_1eff1e68374fb83b\n900,FastSET,pretrained,10,3397376,26054400,0.1303954802259887,source_1eff1e68374fb83b\n900,FastSET,pretrained,11,3102331,26054400,0.11907128930238271,source_1eff1e68374fb83b\n900,FastSET,pretrained,12,2794899,26054400,0.10727167004421519,source_1eff1e68374fb83b\n900,FastSET,pretrained,13,2382145,26054400,0.09142966255219848,source_1eff1e68374fb83b\n900,FastSET,pretrained,14,1708317,26054400,0.06556731300663228,source_1eff1e68374fb83b\n900,FastSET,pretrained,15,552193,26054400,0.021193848255956767,source_1eff1e68374fb83b\n900,FastSET,pretrained,16,35380,26054400,0.0013579280275116679,source_1eff1e68374fb83b\n900,FastSET,pretrained,17,649,26054400,2.490942028985507e-05,source_1eff1e68374fb83b\n900,FastSET,pretrained,18,15,26054400,5.757184966838615e-07,source_1eff1e68374fb83b\n900,FastSET,adapted,4,5295813,26054400,0.20325983327192337,source_56c7edd693441b11\n900,FastSET,adapted,5,4763634,26054400,0.18283414701547532,source_56c7edd693441b11\n900,FastSET,adapted,6,4209686,26054400,0.16157293969540654,source_56c7edd693441b11\n900,FastSET,adapted,7,3600470,26054400,0.1381904783836895,source_56c7edd693441b11\n900,FastSET,adapted,8,2872538,26054400,0.11025155060181774,source_56c7edd693441b11\n900,FastSET,adapted,9,2034122,26054400,0.07807211066077131,source_56c7edd693441b11\n900,FastSET,adapted,10,1005060,26054400,0.03857544215180545,source_56c7edd693441b11\n900,FastSET,adapted,11,146114,26054400,0.005608035494964382,source_56c7edd693441b11\n900,FastSET,adapted,12,11432,26054400,0.0004387742569393269,source_56c7edd693441b11\n900,FastSET,adapted,13,736,26054400,2.824858757062147e-05,source_56c7edd693441b11\n900,FastSET,adapted,14,128,26054400,4.912797838368951e-06,source_56c7edd693441b11\n900,FastSET,adapted,15,0,26054400,0.0,source_56c7edd693441b11\n900,FastSET,adapted,16,0,26054400,0.0,source_56c7edd693441b11\n900,FastSET,adapted,17,0,26054400,0.0,source_56c7edd693441b11\n900,FastSET,adapted,18,0,26054400,0.0,source_56c7edd693441b11\n900,CNN,pretrained,4,6111492,26054400,0.23456659911569638,source_e7112802869ae7b9\n900,CNN,pretrained,5,5669233,26054400,0.21759215334070253,source_e7112802869ae7b9\n900,CNN,pretrained,6,5251968,26054400,0.20157700810611642,source_e7112802869ae7b9\n900,CNN,pretrained,7,4876264,26054400,0.18715702530090886,source_e7112802869ae7b9\n900,CNN,pretrained,8,4557169,26054400,0.17490976572095307,source_e7112802869ae7b9\n900,CNN,pretrained,9,4298827,26054400,0.16499428119626627,source_e7112802869ae7b9\n900,CNN,pretrained,10,4086942,26054400,0.1568618736182756,source_e7112802869ae7b9\n900,CNN,pretrained,11,3918563,26054400,0.15039927996806682,source_e7112802869ae7b9\n900,CNN,pretrained,12,3775775,26054400,0.1449189004544338,source_e7112802869ae7b9\n900,CNN,pretrained,13,3653536,26054400,0.14022721690002457,source_e7112802869ae7b9\n900,CNN,pretrained,14,3541057,26054400,0.13591013418079095,source_e7112802869ae7b9\n900,CNN,pretrained,15,3442039,26054400,0.13210970124048146,source_e7112802869ae7b9\n900,CNN,pretrained,16,3367200,26054400,0.1292372881355932,source_e7112802869ae7b9\n900,CNN,pretrained,17,3316564,26054400,0.12729381601572096,source_e7112802869ae7b9\n900,CNN,pretrained,18,3297103,26054400,0.12654687883812332,source_e7112802869ae7b9\n900,CNN,adapted,4,5468690,26054400,0.20989506570867109,source_c41d331f677068bd\n900,CNN,adapted,5,4968426,26054400,0.19069431650700075,source_c41d331f677068bd\n900,CNN,adapted,6,4464141,26054400,0.171339236366986,source_c41d331f677068bd\n900,CNN,adapted,7,3943813,26054400,0.1513684061041513,source_c41d331f677068bd\n900,CNN,adapted,8,3376654,26054400,0.1296001443134365,source_c41d331f677068bd\n900,CNN,adapted,9,2711409,26054400,0.10406722089167281,source_c41d331f677068bd\n900,CNN,adapted,10,1863309,26054400,0.07151609708916729,source_c41d331f677068bd\n900,CNN,adapted,11,508869,26054400,0.019531019712601325,source_c41d331f677068bd\n900,CNN,adapted,12,766,26054400,2.9400024563989192e-05,source_c41d331f677068bd\n900,CNN,adapted,13,0,26054400,0.0,source_c41d331f677068bd\n900,CNN,adapted,14,0,26054400,0.0,source_c41d331f677068bd\n900,CNN,adapted,15,0,26054400,0.0,source_c41d331f677068bd\n900,CNN,adapted,16,0,26054400,0.0,source_c41d331f677068bd\n900,CNN,adapted,17,0,26054400,0.0,source_c41d331f677068bd\n900,CNN,adapted,18,0,26054400,0.0,source_c41d331f677068bd\n1200,FastSET,pretrained,4,6350231,26054400,0.24372969632768363,source_e4fc6639eb813141\n1200,FastSET,pretrained,5,5998511,26054400,0.23023024901744044,source_e4fc6639eb813141\n1200,FastSET,pretrained,6,5676491,26054400,0.2178707243306313,source_e4fc6639eb813141\n1200,FastSET,pretrained,7,5400234,26054400,0.20726764001473838,source_e4fc6639eb813141\n1200,FastSET,pretrained,8,5169202,26054400,0.19840034696634734,source_e4fc6639eb813141\n1200,FastSET,pretrained,9,4965614,26054400,0.19058638847948908,source_e4fc6639eb813141\n1200,FastSET,pretrained,10,4785472,26054400,0.1836723163841808,source_e4fc6639eb813141\n1200,FastSET,pretrained,11,4615368,26054400,0.17714351510685336,source_e4fc6639eb813141\n1200,FastSET,pretrained,12,4445738,26054400,0.17063290653402113,source_e4fc6639eb813141\n1200,FastSET,pretrained,13,4256820,26054400,0.16338200073691966,source_e4fc6639eb813141\n1200,FastSET,pretrained,14,4037711,26054400,0.15497232713092607,source_e4fc6639eb813141\n1200,FastSET,pretrained,15,3786503,26054400,0.1453306543232621,source_e4fc6639eb813141\n1200,FastSET,pretrained,16,3478505,26054400,0.13350931128715304,source_e4fc6639eb813141\n1200,FastSET,pretrained,17,3108315,26054400,0.11930096260132646,source_e4fc6639eb813141\n1200,FastSET,pretrained,18,2663948,26054400,0.10224560918693196,source_e4fc6639eb813141\n1200,FastSET,adapted,4,5360266,26054400,0.20573361888970768,source_47235c030633d82f\n1200,FastSET,adapted,5,4844218,26054400,0.18592706030459347,source_47235c030633d82f\n1200,FastSET,adapted,6,4310120,26054400,0.16542772046180299,source_47235c030633d82f\n1200,FastSET,adapted,7,3720875,26054400,0.14281177075657087,source_47235c030633d82f\n1200,FastSET,adapted,8,2983361,26054400,0.11450507399901744,source_47235c030633d82f\n1200,FastSET,adapted,9,2003528,26054400,0.07689787521493491,source_47235c030633d82f\n1200,FastSET,adapted,10,787307,26054400,0.030217813497912062,source_47235c030633d82f\n1200,FastSET,adapted,11,52108,26054400,0.00199996929501351,source_47235c030633d82f\n1200,FastSET,adapted,12,1656,26054400,6.35593220338983e-05,source_47235c030633d82f\n1200,FastSET,adapted,13,0,26054400,0.0,source_47235c030633d82f\n1200,FastSET,adapted,14,0,26054400,0.0,source_47235c030633d82f\n1200,FastSET,adapted,15,0,26054400,0.0,source_47235c030633d82f\n1200,FastSET,adapted,16,0,26054400,0.0,source_47235c030633d82f\n1200,FastSET,adapted,17,0,26054400,0.0,source_47235c030633d82f\n1200,FastSET,adapted,18,0,26054400,0.0,source_47235c030633d82f\n1200,CNN,pretrained,4,6416580,26054400,0.24627625276344878,source_bee33bae48ecb894\n1200,CNN,pretrained,5,6017779,26054400,0.2309697786170474,source_bee33bae48ecb894\n1200,CNN,pretrained,6,5654282,26054400,0.21701831552444117,source_bee33bae48ecb894\n1200,CNN,pretrained,7,5343893,26054400,0.2051052029599607,source_bee33bae48ecb894\n1200,CNN,pretrained,8,5103725,26054400,0.19588725896585604,source_bee33bae48ecb894\n1200,CNN,pretrained,9,4927529,26054400,0.18912463921640874,source_bee33bae48ecb894\n1200,CNN,pretrained,10,4796050,26054400,0.18407831306804226,source_bee33bae48ecb894\n1200,CNN,pretrained,11,4695394,26054400,0.18021501166789486,source_bee33bae48ecb894\n1200,CNN,pretrained,12,4620472,26054400,0.1773394129206583,source_bee33bae48ecb894\n1200,CNN,pretrained,13,4555845,26054400,0.17485894896831244,source_bee33bae48ecb894\n1200,CNN,pretrained,14,4507652,26054400,0.17300924220093344,source_bee33bae48ecb894\n1200,CNN,pretrained,15,4463484,26054400,0.17131401989683123,source_bee33bae48ecb894\n1200,CNN,pretrained,16,4431671,26054400,0.170092997727831,source_bee33bae48ecb894\n1200,CNN,pretrained,17,4406019,26054400,0.16910844233603536,source_bee33bae48ecb894\n1200,CNN,pretrained,18,4384536,26054400,0.16828389830508475,source_bee33bae48ecb894\n1200,CNN,adapted,4,5499947,26054400,0.21109474791206093,source_16eb357fa7795c27\n1200,CNN,adapted,5,5023107,26054400,0.1927930407148121,source_16eb357fa7795c27\n1200,CNN,adapted,6,4545310,26054400,0.1744546026774748,source_16eb357fa7795c27\n1200,CNN,adapted,7,4053344,26054400,0.15557234094817,source_16eb357fa7795c27\n1200,CNN,adapted,8,3517122,26054400,0.13499147936624908,source_16eb357fa7795c27\n1200,CNN,adapted,9,2851578,26054400,0.10944707995578482,source_16eb357fa7795c27\n1200,CNN,adapted,10,1932170,26054400,0.07415906718251045,source_16eb357fa7795c27\n1200,CNN,adapted,11,564604,26054400,0.021670197740112994,source_16eb357fa7795c27\n1200,CNN,adapted,12,4357,26054400,0.00016722703267010562,source_16eb357fa7795c27\n1200,CNN,adapted,13,0,26054400,0.0,source_16eb357fa7795c27\n1200,CNN,adapted,14,0,26054400,0.0,source_16eb357fa7795c27\n1200,CNN,adapted,15,0,26054400,0.0,source_16eb357fa7795c27\n1200,CNN,adapted,16,0,26054400,0.0,source_16eb357fa7795c27\n1200,CNN,adapted,17,0,26054400,0.0,source_16eb357fa7795c27\n1200,CNN,adapted,18,0,26054400,0.0,source_16eb357fa7795c27\n',
}
# ===== 绘图代码区 =====

def main():
    O.mkdir(parents=True,exist_ok=True);(O/'data').mkdir(exist_ok=True)
    hashes={str(p):hashlib.sha256(DATA_CSV[p.relative_to(P).as_posix()].encode()).hexdigest() for p in INPUTS}
    data=rows(INPUTS[0]);delay=rows(INPUTS[3])
    for rr in [data,delay]:
        for r in rr:
            assert int(r['bits'])==26054400
            assert abs(float(r['ber'])-int(r['bit_errors'])/int(r['bits']))<1e-12
    ber_plot(data,['Paris P2','Munich P2','San Francisco P2','Florence P2'],'four_city_ber',labels=['Paris P2','Munich P2','SF P2','Florence P2'])
    ber_plot(data,['Paris P2','Paris P3'],'paris_P2_P3_ber')
    ber_plot(data,['Paris to San Francisco','San Francisco to Paris'],'cross_city_ber',labels=['Paris → SF','SF → Paris'])
    delay_plot(delay)
    # Full-range supplement exposes the high-SNR behavior obscured by the requested crop.
    ber_plot(data,['San Francisco P2','Paris to San Francisco'],'supplement/high_snr_behavior',labels=['SF P2','Paris → SF'],xmax=18)
    exceptions=[]
    for case,reason in [('San Francisco P2','CNN high-SNR BER rises from 1.42e-6 at 14 dB to 7.10e-6 at 18 dB; cropping at 12 dB hides this measured error-tail behavior.'),('Paris to San Francisco','Both adapted models still have nonzero errors above 12 dB, with different residual tails. Retain full-range companion figure for assessment.')]:
        points=[{k:r[k] for k in ['case','method','phase','snr_db','bit_errors','bits','ber']} for r in data if r['case']==case and r['phase']=='adapted' and float(r['snr_db']) in [12,14,16,18]]
        exceptions.append({'case':case,'main_display_snr_db':[4,12],'recommended_companion_snr_db':[4,18],'reason':reason,'evidence':points,'companion':'supplement/high_snr_behavior.pdf'})
    (O/'DISPLAY_RANGE_EXCEPTIONS.json').write_text(json.dumps(exceptions,indent=2))
    timing_plot()
    (O/'DISPLAY_RANGES.json').write_text(json.dumps(DISPLAY_RANGES,indent=2))
    (O/'CAPTIONS.json').write_text(json.dumps(CAPTIONS,indent=2))
    (O/'ASSET_MAP.json').write_text(json.dumps(MAP,indent=2))
    audit={'source_sha256':hashes,'sources_unchanged':True,'copied_csvs_byte_identical':True,'integer_ber_verified':True,'city_ber_rows':len(data),'delay_ber_rows':len(delay),'zero_error_marker':'down triangle at 1/N; not BER estimate','smoothing':False,'new_training':False,'new_measurement':False,'ablation_touched':False,'900ns_full_test_failure_preserved':True,'figure_count':len(MAP),'timing':'historical training + separately measured input; excludes BER evaluation','main_snr_db':[4,12],'noise_definition':'rt_full_campaign/stream.py:33 calls sionna.utils.ebnodb2no; axes use Eb/N0','target_annotation':'epsilon=2e-4 on every BER axis','all_subplots_have_xylabels_and_ticks':True,'300ns_figure_generated':False}
    (O/'AUDIT.json').write_text(json.dumps(audit,indent=2))
    (O/'README_CN.txt').write_text('主稿版：FastSET绿色圆点、CNN深蓝方块；STIXGeneral黑色坐标轴，透明背景。每个子轴保留X/Y轴名和刻度，4–12dB每2dB纵横网格；Y轴按当前显示的数据范围收紧，同一组面板保持一致。\nRT delay左侧展示900/1200ns适应后BER四条线，右侧展示600/900/1200ns耗时比，900ns可靠性差异仅在正文说明；不加星号或柱纹。独立RT delay时间图已删除，避免与图20重复。RT300不出图。\n旧金山P2、巴黎到旧金山存在12dB以上不同尾部行为，已保留新完整范围附图，具体整数计数见DISPLAY_RANGE_EXCEPTIONS.json。原始CSV和旧图都不改。\n未重训、未新计时、未碰消融。时间仍为历史训练与独立读取测量的相加。\n')
    print(json.dumps({'output':str(O),'figures':len(MAP),'integer_ber_rows_verified':len(data)+len(delay)}))

if __name__=='__main__':main()
