"""Publication layout of completed data-volume results; no model evaluation."""
import csv,io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator,MaxNLocator
from ieee_style import apply,_save
from data_volume_results import SUMMARY_CSV,PEBS_CSV
apply()
O=Path(__file__).resolve().parent.parent/'figs/data_volume'
rows=list(csv.DictReader(io.StringIO(SUMMARY_CSV)))
pebs=list(csv.DictReader(io.StringIO(PEBS_CSV)))
frames=[544,2720,5440,10880,54400]
labels=['0.1m','0.5m','m','2m','10m']
styles=[('d8_w64','FastSET 8–64','#0072BD','o','-'),('d9_w32','FastSET 9–32','#77AC30','^','-.'),('CNN','E2E','#D95319','s','--')]
for case in ['A2B','A2C']:
 fig,axs=plt.subplots(1,3,figsize=(7.16,2.4))
 for c,label,color,marker,ls in styles:
  data=sorted([r for r in rows if r['scenario']==case and r['candidate']==c],key=lambda r:int(r['frames']))
  for ax,key,ylabel in zip(axs,['first_hit_test_ber','first_hit_iteration','first_hit_seconds'],['Evaluation BER','Iterations to target','Adaptation time (s)']):
   ax.plot([int(r['frames']) for r in data],[float(r[key]) for r in data],color=color,marker=marker,mfc='white',ms=4.5,ls=ls,label=label,lw=1.8)
   ax.set(xscale='log',xlabel='Training-set size',ylabel=ylabel,xlim=(430,70000),xticks=frames,xticklabels=labels)
   ax.xaxis.set_minor_locator(NullLocator());ax.tick_params(direction='in',pad=2,labelsize=8.5)
   ax.grid(which='major',axis='both',color='#D5D8DC',lw=.5);ax.set_axisbelow(True)
 axs[0].set_yscale('log');axs[0].set_ylim((3.8e-5,3e-4) if case=='A2B' else (2.9e-5,3.5e-4))
 axs[0].yaxis.set_major_locator(FixedLocator([5e-5,1e-4,2e-4]));axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v,p: {5e-5:r'$5\times10^{-5}$',1e-4:r'$10^{-4}$',2e-4:r'$2\times10^{-4}$'}[v]));axs[0].yaxis.set_minor_locator(NullLocator())
 axs[0].axhline(2e-4,ls=':',color='#666666',lw=.9)
 axs[1].yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));axs[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
 axs[1].set_ylim(bottom=0);axs[2].set_ylim(bottom=0)
 fig.legend(*axs[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.52,.99),frameon=False,fontsize=9)
 fig.subplots_adjust(left=.085,right=.995,bottom=.22,top=.81,wspace=.55)
 for ext in ['pdf','png','svg']:_save(fig,O/(case+'_data_volume.'+ext),dpi=250,transparent=True)
 plt.close(fig)
 values=[float(r['pebs']) for r in pebs if r['scenario']==case and r['candidate'] in ['d8_w64','d9_w32']];vmax=max(values)
 fig,axs=plt.subplots(1,2,figsize=(7.16,2.35))
 for ax,c,label in zip(axs,['d8_w64','d9_w32'],['8–64 split','9–32 split']):
  z=np.array([[float(next(r['pebs'] for r in pebs if r['scenario']==case and r['candidate']==c and float(r['alpha'])==a and int(r['frames'])==f)) for f in frames] for a in [.2,.4,.6,.8,.9]])
  im=ax.imshow(z,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
  ax.set(xticks=range(5),xticklabels=labels,yticks=range(5),yticklabels=['0.2','0.4','0.6','0.8','0.9'],xlabel='Training-set size',ylabel=r'$\alpha$',title=label)
  ax.tick_params(pad=2,labelsize=9)
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',color='white' if z[i,j]>.6*vmax else 'black',fontsize=9)
 fig.subplots_adjust(left=.07,right=.89,bottom=.23,top=.86,wspace=.25)
 cax=fig.add_axes([.915,.23,.017,.63]);fig.colorbar(im,cax=cax,label='PEBS');cax.tick_params(labelsize=9)
 for ext in ['pdf','png','svg']:_save(fig,O/(case+'_pebs_alpha.'+ext),dpi=250,transparent=True)
 plt.close(fig)
print('Replotted four figures from the 30 completed runs.')
