# -*- coding: utf-8 -*-
# 独立绘图文件：所需数据直接嵌入本文件，不读取仓库数据、CSV、模型或GPU。
# 修改配色/线宽/字号/坐标范围后直接运行；输出覆盖 ../figs/current/ 对应图片。
# 依赖：Python 3.10+，numpy，matplotlib。
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
O=Path(__file__).resolve().parent.parent/'figs/current';O.mkdir(parents=True,exist_ok=True)
COMPONENTS=['read_seconds','deserialize_check_seconds','tensor_device_seconds','training_seconds']
NAMES=['File read','Deserialize/check','Tensor/device prep.','Training update']
COLORS=['#0072BD','#D95319','#EDB120','#77AC30']

HATCHES=['///','...','\\\\','']
def component_colors(rows,k):
 from matplotlib.colors import to_rgb
 out=[]
 for r in rows:
  base=np.array(to_rgb('#0072BD' if r['method']=='FastSET' else '#D95319'))
  pale=np.array(to_rgb('#D9EAF4' if r['method']=='FastSET' else '#F7DDCF'))
  t=[0,.3,.6,1][COMPONENTS.index(k)]
  out.append(tuple(pale*(1-t)+base*t))
 return out

# ===== 实测数据区，单位秒 =====
summary = [{'id': 'A2B_FastSET', 'scenario': 'A2B', 'method': 'FastSET', 'selected_repeat': '1', 'read_seconds': 0.0013919909979449585, 'deserialize_check_seconds': 0.0011067289997299667, 'tensor_device_seconds': 0.0019161669988534413, 'training_seconds': 0.12418051200074842, 'total_seconds': 0.1285953989972768, 'historical_main_total_seconds': '0.1181350949964326', 'delta_seconds': '0.01046030400084419', 'delta_percent': '8.854527099808962', 'est_iterations': '1', 'dem_iterations': '0', 'cnn_iterations': '0'}, {'id': 'A2B_CNN', 'scenario': 'A2B', 'method': 'CNN', 'selected_repeat': '3', 'read_seconds': 0.03242732199942111, 'deserialize_check_seconds': 0.05316863000189187, 'tensor_device_seconds': 0.10345543498624465, 'training_seconds': 10.509601574001863, 'total_seconds': 10.69865296098942, 'historical_main_total_seconds': '10.196998199982772', 'delta_seconds': '0.5016547610066482', 'delta_percent': '4.919631750131082', 'est_iterations': '0', 'dem_iterations': '0', 'cnn_iterations': '50'}, {'id': 'A2C_FastSET', 'scenario': 'A2C', 'method': 'FastSET', 'selected_repeat': '3', 'read_seconds': 0.0049534329991729464, 'deserialize_check_seconds': 0.006226130000868579, 'tensor_device_seconds': 0.011670427000353811, 'training_seconds': 0.7957874649982841, 'total_seconds': 0.8186374549986795, 'historical_main_total_seconds': '0.8059559390194408', 'delta_seconds': '0.012681515979238611', 'delta_percent': '1.5734750952598464', 'est_iterations': '5', 'dem_iterations': '1', 'cnn_iterations': '0'}, {'id': 'A2C_CNN', 'scenario': 'A2C', 'method': 'CNN', 'selected_repeat': '2', 'read_seconds': 0.1097648340073647, 'deserialize_check_seconds': 0.09172592799222912, 'tensor_device_seconds': 0.17367209702933906, 'training_seconds': 17.651015044968517, 'total_seconds': 18.02617790399745, 'historical_main_total_seconds': '17.719502427980842', 'delta_seconds': '0.3066754760166077', 'delta_percent': '1.730722842038368', 'est_iterations': '0', 'dem_iterations': '0', 'cnn_iterations': '84'}]

# ===== 绘图代码区 =====

plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.labelsize':9,'axes.titlesize':8.5,'legend.fontsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

from ieee_style import apply as apply_ieee_style
apply_ieee_style()
fig=plt.figure(figsize=(7.1,3.85));gs=fig.add_gridspec(2,2,height_ratios=[2.1,1],hspace=.6)
for j,method in enumerate(['FastSET','CNN']):
 ax=fig.add_subplot(gs[0,j]);rr=[next(r for r in summary if r['scenario']==s and r['method']==method) for s in ['A2B','A2C']]
 bottom=np.zeros(2)
 for k,col in zip(COMPONENTS,COLORS):
  vals=np.array([r[k]*1000 for r in rr]);ax.bar([0,1],vals,bottom=bottom,color=component_colors(rr,k),hatch=HATCHES[COMPONENTS.index(k)],width=.48,edgecolor='white',linewidth=.2,zorder=3);bottom+=vals
 for i,r in enumerate(rr):
  ax.text(i,r['total_seconds']*1000+max(bottom)*.04,f"{r['total_seconds']*1000:.2f}",ha='center',va='bottom',fontsize=8.5)
  ax.text(i,(r['total_seconds']-r['training_seconds']/2)*1000,f"{r['training_seconds']*1000:.2f}",ha='center',va='center',fontsize=8.5,color='white')
 ax.set(xticks=[0,1],xticklabels=['A2B','A2C'],ylabel='Online adaptation time (ms)',ylim=(0,max(bottom)*1.22),xlim=(-.6,1.6));ax.grid(axis='y',color='#e2e2e2',lw=.5);ax.set_axisbelow(True)
fig.legend(handles=[Patch(facecolor='#D5D8DC',edgecolor='#5D6D7E',hatch=h,label=n) for h,n in zip(HATCHES,NAMES)],loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,columnspacing=1.2,handlelength=2.4)
ax=fig.add_subplot(gs[1,:]);ax.axis('off')
cell=[[f"{r[k]*1000:.2f}" for k in COMPONENTS]+[f"{r['total_seconds']*1000:.2f}"] for r in summary]
table=ax.table(cellText=cell,rowLabels=[r['scenario']+' '+r['method'] for r in summary],colLabels=['Read','Deserialize/check','Tensor/device','Training','Total'],loc='center',cellLoc='right',colLoc='center',colWidths=[.13,.22,.19,.17,.17]);table.auto_set_font_size(False);table.set_fontsize(7.2);table.scale(1,1.1)
for (i,j),cell in table.get_celld().items():
 cell.set_edgecolor('#dddddd');cell.set_linewidth(.4)
 if i==0:cell.set_facecolor('#eeeeee')
fig.text(.54,.29,'Measured components (ms); panels use different y-axis scales',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.105,right=.98,top=.84,bottom=.035,wspace=.36)
(O/'supplement').mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(O/'supplement'/('adaptation_time_breakdown_detailed.'+ext),dpi=300)
plt.close(fig)
# Four bars on one linear seconds axis; the inset repeats their actual bottom 400 ms.
fig,ax=plt.subplots(figsize=(7.1,2.7));xs=np.array([0.,1.,3.,4.]);bottom=np.zeros(4)
for k,col in zip(COMPONENTS,COLORS):
 values=np.array([r[k] for r in summary]);ax.bar(xs,values,bottom=bottom,width=.56,color=component_colors(summary,k),hatch=HATCHES[COMPONENTS.index(k)],edgecolor='#5D6D7E',linewidth=.3,zorder=3);bottom+=values
for x,r in zip(xs,summary):
 ax.text(x,r['total_seconds']+.45,f"{r['total_seconds']:.3f}",ha='center',va='bottom',fontsize=8.5)
 if r['method']=='CNN':ax.text(x,r['total_seconds']-r['training_seconds']/2,f"{r['training_seconds']:.2f}",ha='center',va='center',fontsize=8.5,color='white')
ax.set(xticks=xs,xticklabels=['(1) A2B\nFastSET','(2) A2B\nCNN','(3) A2C\nFastSET','(4) A2C\nCNN'],ylabel='Online adaptation time (s)',xlim=(-.55,4.55),ylim=(0,21),yticks=[0,5,10,15,20]);ax.grid(axis='y',color='#e2e2e2',lw=.5);ax.set_axisbelow(True)
inset=fig.add_axes([.47,.41,.225,.34]);b=np.zeros(4)
for k,col in zip(COMPONENTS,COLORS):
 values=np.array([r[k]*1000 for r in summary]);inset.bar(range(4),values,bottom=b,width=.58,color=component_colors(summary,k),hatch=HATCHES[COMPONENTS.index(k)],edgecolor='#5D6D7E',linewidth=.3,zorder=3);b+=values
inset.set(xticks=range(4),xticklabels=['1','2','3','4'],ylim=(0,400),yticks=[0,200,400],ylabel='Time (ms)',xlabel='Same bar number')
inset.tick_params(labelsize=8.5,pad=1);inset.xaxis.label.set_size(8.5);inset.yaxis.label.set_size(8.5);inset.title.set_size(8.5);inset.xaxis.labelpad=1;inset.yaxis.labelpad=1
inset.grid(axis='y',color='#dddddd',lw=.4);inset.set_axisbelow(True)
fig.legend(handles=[Patch(facecolor='#D5D8DC',edgecolor='#5D6D7E',hatch=h,label=n) for h,n in zip(HATCHES,NAMES)],loc='upper center',bbox_to_anchor=(.5,1.01),ncol=4,frameon=False,columnspacing=1.2,handlelength=2.4)
fig.subplots_adjust(left=.085,right=.99,top=.84,bottom=.22)
for ext in ['png','pdf','svg']:fig.savefig(O/('adaptation_time_breakdown.'+ext),dpi=300)
plt.close(fig)

# Editable manuscript table generated from the same embedded measurements.
import csv
_counts = [r['est_iterations']+'/'+r['dem_iterations'] if r['method']=='FastSET' else r['cnn_iterations'] for r in summary]
_table_rows = ['Retraining iterations & '+' & '.join(_counts)+r' \\']
_table_rows.append(r'\midrule')
for _name,_key in [('File reading','read_seconds'),('Deserialization/checks','deserialize_check_seconds'),('Tensor preparation/transfer','tensor_device_seconds'),('Training','training_seconds'),('Total (ms)','total_seconds')]:
 _table_rows.append(_name+' & '+' & '.join(f"{r[_key]*1000:.2f}" for r in summary)+r' \\')
_table = [r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{A2B} & \multicolumn{2}{c}{A2C} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r' & FastSET & E2E & FastSET & E2E \\',r'\midrule']+_table_rows+[r'\bottomrule',r'\end{tabular}']
(O/'adaptation_time_breakdown_table.tex').write_text('\n'.join(_table)+'\n')
_table_data = [dict(scenario=r['scenario'],method=r['method'],est_iterations=r['est_iterations'],dem_iterations=r['dem_iterations'],cnn_iterations=r['cnn_iterations'],**{k.replace('_seconds','_ms'):r[k]*1000 for k in COMPONENTS+['total_seconds']}) for r in summary]
with (O/'adaptation_time_breakdown_table.csv').open('w',newline='') as _f:
 _w=csv.DictWriter(_f,fieldnames=list(_table_data[0]));_w.writeheader();_w.writerows(_table_data)
