"""Editable speed figure; all measured data are embedded below by report.py."""
import csv
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np
from ieee_style import apply

BER_CSV = 'speed_mps,phase,snr,bit_errors,bits,ber\n0,pretrained,4,5519610,26054400,0.21184943809874723\n0,pretrained,5,5039013,26054400,0.19340353260869564\n0,pretrained,6,4526813,26054400,0.1737446650085974\n0,pretrained,7,3937005,26054400,0.15110710666912305\n0,pretrained,8,3312096,26054400,0.1271223286661754\n0,pretrained,9,2685667,26054400,0.10307921118889708\n0,pretrained,10,2130153,26054400,0.0817578988577745\n0,pretrained,11,1674189,26054400,0.06425743828297716\n0,pretrained,12,1322785,26054400,0.05077011944239745\n0,pretrained,13,1033419,26054400,0.03966389554163596\n0,pretrained,14,795030,26054400,0.030514231761238027\n0,pretrained,15,606196,26054400,0.023266549987718006\n0,pretrained,16,464476,26054400,0.01782716163104888\n0,pretrained,17,354909,26054400,0.013621845062638173\n0,pretrained,18,272111,26054400,0.010443955723409481\n0,adapted,4,4973117,26054400,0.19087436287153034\n0,adapted,5,4307822,26054400,0.16533952038811103\n0,adapted,6,3430344,26054400,0.1316608327192336\n0,adapted,7,2017021,26054400,0.0774157531933186\n0,adapted,8,761574,26054400,0.02923014922623434\n0,adapted,9,234059,26054400,0.008983473041021862\n0,adapted,10,66077,26054400,0.0025361167403586344\n0,adapted,11,22169,26054400,0.000850873556865635\n0,adapted,12,4781,26054400,0.00018350067550970277\n0,adapted,13,1589,26054400,6.098777941537706e-05\n0,adapted,14,32,26054400,1.2281994595922378e-06\n0,adapted,15,0,26054400,0.0\n0,adapted,16,0,26054400,0.0\n0,adapted,17,0,26054400,0.0\n0,adapted,18,0,26054400,0.0\n5,pretrained,4,5530447,26054400,0.21226537552198477\n5,pretrained,5,5051858,26054400,0.1938965395480226\n5,pretrained,6,4543275,26054400,0.17437649686809137\n5,pretrained,7,3958659,26054400,0.1519382138909359\n5,pretrained,8,3338842,26054400,0.12814887312699583\n5,pretrained,9,2723983,26054400,0.10454982651682633\n5,pretrained,10,2167887,26054400,0.08320617630803243\n5,pretrained,11,1700672,26054400,0.06527388847948908\n5,pretrained,12,1336725,26054400,0.051305153831982316\n5,pretrained,13,1034603,26054400,0.03970933892164087\n5,pretrained,14,791926,26054400,0.030395096413657577\n5,pretrained,15,588550,26054400,0.02258927474821911\n5,pretrained,16,435476,26054400,0.016714105870793416\n5,pretrained,17,338388,26054400,0.012987748710390568\n5,pretrained,18,267800,26054400,0.01027849422746254\n5,adapted,4,4914767,26054400,0.18863481791943013\n5,adapted,5,4255378,26054400,0.1633266549987718\n5,adapted,6,3371685,26054400,0.1294094279661017\n5,adapted,7,2024500,26054400,0.07770280643576517\n5,adapted,8,833939,26054400,0.03200760716040285\n5,adapted,9,267698,26054400,0.010274579341685089\n5,adapted,10,75619,26054400,0.0029023504667157946\n5,adapted,11,17685,26054400,0.0006787721075902726\n5,adapted,12,3406,26054400,0.0001307264799803488\n5,adapted,13,12,26054400,4.6057479734708915e-07\n5,adapted,14,0,26054400,0.0\n5,adapted,15,0,26054400,0.0\n5,adapted,16,0,26054400,0.0\n5,adapted,17,0,26054400,0.0\n5,adapted,18,0,26054400,0.0\n10,pretrained,4,5555763,26054400,0.2132370348194547\n10,pretrained,5,5080442,26054400,0.19499362871530337\n10,pretrained,6,4579882,26054400,0.1757815186686318\n10,pretrained,7,4010551,26054400,0.15392989283959715\n10,pretrained,8,3404932,26054400,0.1306854888233849\n10,pretrained,9,2793864,26054400,0.107231945467944\n10,pretrained,10,2234258,26054400,0.08575357713092606\n10,pretrained,11,1762443,26054400,0.06764473563006632\n10,pretrained,12,1372804,26054400,0.05268991034143945\n10,pretrained,13,1048180,26054400,0.040230440923605996\n10,pretrained,14,778548,26054400,0.029881632277081797\n10,pretrained,15,568634,26054400,0.02182487410955539\n10,pretrained,16,429646,26054400,0.016490343281748955\n10,pretrained,17,325418,26054400,0.01248994411692459\n10,pretrained,18,248953,26054400,0.009555123126995824\n10,adapted,4,4943984,26054400,0.18975620240727095\n10,adapted,5,4295817,26054400,0.16487875368459837\n10,adapted,6,3437240,26054400,0.13192550970277572\n10,adapted,7,2130717,26054400,0.08177954587324981\n10,adapted,8,890708,26054400,0.03418647138295259\n10,adapted,9,268708,26054400,0.01031334438712847\n10,adapted,10,63912,26054400,0.0024530213706705968\n10,adapted,11,8541,26054400,0.00032781411201179074\n10,adapted,12,1341,26054400,5.1469233603537215e-05\n10,adapted,13,0,26054400,0.0\n10,adapted,14,0,26054400,0.0\n10,adapted,15,0,26054400,0.0\n10,adapted,16,0,26054400,0.0\n10,adapted,17,0,26054400,0.0\n10,adapted,18,0,26054400,0.0\n20,pretrained,4,5645195,26054400,0.21666954525915008\n20,pretrained,5,5182848,26054400,0.1989240972733972\n20,pretrained,6,4705806,26054400,0.18061463706705969\n20,pretrained,7,4189706,26054400,0.16080608265782362\n20,pretrained,8,3634103,26054400,0.13948135439695405\n20,pretrained,9,3049336,26054400,0.11703727585359862\n20,pretrained,10,2491349,26054400,0.09562104673298943\n20,pretrained,11,1998957,26054400,0.07672243459837878\n20,pretrained,12,1565844,26054400,0.06009902358142962\n20,pretrained,13,1183849,26054400,0.04543758443871285\n20,pretrained,14,884548,26054400,0.03395004298698109\n20,pretrained,15,660519,26054400,0.025351533714075166\n20,pretrained,16,478937,26054400,0.018382192643085236\n20,pretrained,17,348983,26054400,0.013394397875214934\n20,pretrained,18,250705,26054400,0.009622367047408499\n20,adapted,4,5100113,26054400,0.19574862595185458\n20,adapted,5,4478324,26054400,0.17188359739621714\n20,adapted,6,3706685,26054400,0.14226714105870794\n20,adapted,7,2557774,26054400,0.09817052014247114\n20,adapted,8,1186787,26054400,0.04555034850159666\n20,adapted,9,370352,26054400,0.014214566445590764\n20,adapted,10,78036,26054400,0.0029951179071481208\n20,adapted,11,13980,26054400,0.0005365696389093589\n20,adapted,12,4486,26054400,0.00017217821174158685\n20,adapted,13,2265,26054400,8.693349299926308e-05\n20,adapted,14,1996,26054400,7.660894129206583e-05\n20,adapted,15,1592,26054400,6.110292311471383e-05\n20,adapted,16,942,26054400,3.61551215917465e-05\n20,adapted,17,713,26054400,2.736581920903955e-05\n20,adapted,18,17,26054400,6.524809629083763e-07\n'
TIMING_CSV = 'speed_mps,method,target_reached,est,dem,cnn,bit_errors,bits,ber,input_seconds,training_seconds,other_seconds,total_seconds,selected_repeat\n5,FastSET,True,1,0,0,3406,26054400,0.0001307264799803488,0.005060251001850702,0.12919131900707725,3.1759991543367505e-06,0.1342547460080823,3\n5,CNN,True,0,0,51,4451,26054400,0.00017083486858265784,0.07101603400951717,10.4620422609878,0.00024233599833678454,10.533300630995655,2\n10,FastSET,True,1,0,0,1341,26054400,5.1469233603537215e-05,0.0018009680061368272,0.12589426699560136,1.2520031305029988e-06,0.1276964870048687,3\n10,CNN,True,0,0,44,4731,26054400,0.0001815816138540899,0.06507604703074321,9.118729060952319,0.00026496901409700513,9.184070076997159,3\n20,FastSET,True,4,0,0,4486,26054400,0.00017217821174158685,0.007707114011282101,0.5090496109914966,1.8332008039578795e-05,0.5167750570108183,3\n20,CNN,True,0,0,64,5142,26054400,0.0001973563006632277,0.08971695596119389,13.153031034045853,0.0003334529901621863,13.243081442997209,3\n'
OUT = Path(__file__).resolve().parent.parent / 'figs/current'


def main():
    apply()
    OUT.mkdir(parents=True, exist_ok=True)
    ber = list(csv.DictReader(io.StringIO(BER_CSV)))
    times = list(csv.DictReader(io.StringIO(TIMING_CSV)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.5, 3.0),
                                     gridspec_kw={'width_ratios': [1, 1.5]})
    colors = ['#EDB120', '#7E2F8E', '#77AC30']
    markers = ['o', '^', 's']
    visible = [2e-4]
    for phase, linestyle in [('pretrained', '--'), ('adapted', '-')]:
        for speed, color, marker in zip([5, 10, 20], colors, markers):
            rows = sorted([r for r in ber if int(r['speed_mps']) == speed
                           and r['phase'] == phase and float(r['snr']) <= 12],
                          key=lambda r: float(r['snr']))
            ys = [int(r['bit_errors']) / int(r['bits']) if int(r['bit_errors']) else 1/int(r['bits']) for r in rows]
            visible.extend(ys)
            left.semilogy([float(r['snr']) for r in rows], ys, color=color,
                          marker=marker, mfc='white', ls=linestyle, lw=1.85, ms=4.5,
                          markevery=2, label=f'{speed} m/s '+('before' if phase=='pretrained' else 'after'))
    left.axhline(2e-4, color='#5D6D7E', ls=(0,(3,3)), lw=.9)
    left.set(xlabel=r'$E_b/N_0$ (dB)', ylabel='BER', xticks=[4,6,8,10,12],
             xlim=(3.7,12.3), ylim=(min(visible)/1.65,min(1,max(visible)*1.6)))
    left.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    left.yaxis.set_minor_locator(NullLocator())
    vals=[]; qualified=[]
    for speed in [5,10,20]:
        pair=[next(r for r in times if int(r['speed_mps'])==speed and r['method']==m)
              for m in ['FastSET','CNN']]
        vals.append([float(r['total_seconds']) for r in pair])
        qualified.append(all(r['target_reached']=='True' for r in pair))
    vals=np.asarray(vals);top=float(vals.max());x=np.arange(3)*1.45
    for j,color in enumerate(['#0072BD','#D95319']):
        xx=x+(-.28 if j==0 else .28)
        right.bar(xx,vals[:,j],width=.52,color=color,zorder=3)
        for k,(xxi,value) in enumerate(zip(xx,vals[:,j])):
            right.annotate(f'{value:.2f}',(xxi,value+top*(.025 if j==0 else .065)),
                           xytext=(-3 if j==0 else 0,0),textcoords='offset points',
                           rotation=0,ha='center',va='bottom',zorder=5)
    for k,(f,c) in enumerate(vals):
        label=f'{c/f:.1f}×' if qualified[k] and f>0 else 'N/R'
        y=max(f,c)+top*.21
        right.plot([x[k]-.28,x[k]-.28,x[k]+.28,x[k]+.28],
                   [y-top*.02,y,y,y-top*.02],color='#5D6D7E',lw=.7)
        right.text(x[k],y+top*.025,label,ha='center',va='bottom')
    right.set(xlabel='Speed (m/s)',ylabel='Time (s)',xticks=x,xticklabels=['5','10','20'],
              ylim=(0,top*1.45),xlim=(-.85,3.75))
    for ax in [left,right]:
        ax.grid(axis='both',which='major',color='#D5D8DC',lw=.5)
        ax.set_axisbelow(True);ax.tick_params(direction='in',pad=2)
    handles=[Line2D([],[],color=c,marker=m,mfc='white',lw=1.85,label=f'{s} m/s')
             for s,c,m in zip([5,10,20],colors,markers)]
    handles += [Line2D([],[],color='#5D6D7E',lw=1.85,ls='--',label='Before'),
                Line2D([],[],color='#5D6D7E',lw=1.85,ls='-',label='After'),
                Patch(facecolor='#0072BD',label='FastSET'),Patch(facecolor='#D95319',label='CNN')]
    # Three compact rows; the threshold is stated in the caption.
    fig.legend(handles=handles,ncol=3,frameon=False,loc='upper center',
               bbox_to_anchor=(.52,1.0),columnspacing=.7,handlelength=1.5,labelspacing=.3)
    fig.subplots_adjust(left=.15,right=.98,bottom=.20,top=.72,wspace=.58)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'speed_ber.{ext}',dpi=300,transparent=True)
    plt.close(fig)


if __name__=='__main__':
    main()
