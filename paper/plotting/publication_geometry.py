"""Physical plotting frames for the two-column manuscript; measured data untouched."""
import hashlib
import numpy as np
from matplotlib.transforms import Bbox
SINGLE={'learning_rate_iterations','pebs_pareto_40','pebs_alpha_40','adaptation_time_comparison','backbone_time','A2B_ablation_3seeds_log10','A2C_ablation_3seeds_log10'}
MULTI={'main_ber','four_city_ber','paris_P2_P3_ber','cross_city_ber','environment_transition_performance','multipath_delay_ber','speed_ber','rt_delay_ber'}

def signature(fig):
 h=hashlib.sha256()
 for ax in fig.axes:
  for line in ax.lines:h.update(np.asarray(line.get_xydata(),dtype=float).tobytes())
  for coll in ax.collections:
   h.update(np.asarray(coll.get_offsets(),dtype=float).tobytes())
   for path in coll.get_paths():h.update(np.asarray(path.vertices,dtype=float).tobytes())
 return h.hexdigest()

def normalize(fig,name):
 if name not in SINGLE|MULTI|{'pebs_heatmap_40'}:return None
 before=signature(fig)
 if not getattr(fig,'_physical_layout_done',False):
  fig._physical_layout_done=True
  width=3.5
  axes=list(fig.axes)
  if name=='pebs_heatmap_40':
   # Match the adjacent Pareto panel's physical frame, leaving room for the colorbar.
   height=1.84;fig.set_size_inches(width,height,forward=True)
   axes[0].set_position([.47/width,.40/height,2.36/width,1.25/height])
   axes[1].set_position([2.89/width,.40/height,.06/width,1.25/height])
   axes[1].yaxis.labelpad=2
   axes[1].tick_params(pad=2)
  elif name in SINGLE or name=='main_ber':
   old=getattr(fig,'_compact_legend',None)
   if old is not None:
    handles=getattr(old,'legend_handles',getattr(old,'legendHandles',[]));labels=[t.get_text() for t in old.get_texts()]
    old.remove()
    ablation=name in {'A2B_ablation_3seeds_log10','A2C_ablation_3seeds_log10'}
    pareto=name=='pebs_pareto_40'
    column_options=[4] if ablation else ([3] if pareto else ([3,2,1] if len(labels)>4 else [2,1]))
    for cols in column_options:
     leg=fig.legend(handles,labels,ncols=cols,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False,fontsize=10,handlelength=1.65 if ablation else (1.1 if pareto else 2.8),handletextpad=.2 if ablation or pareto else .25,columnspacing=.3 if ablation or pareto else .5,labelspacing=.15,borderpad=0,borderaxespad=0)
     fig.canvas.draw(); box=leg.get_window_extent(fig.canvas.get_renderer())
     if box.width/fig.dpi<=width-.05:break
     leg.remove()
    legend_h=box.height/fig.dpi+.08
    fig._compact_legend=leg
    fig._compact_report.update(placement='outside',columns=cols,rows=(len(labels)+cols-1)//cols)
   else:legend_h=.06
   bottom=.40;frame_h=1.25;gap=.51
   height=bottom+frame_h+(frame_h+gap if name=='main_ber' else 0)+legend_h
   fig.set_size_inches(width,height,forward=True)
   for i,ax in enumerate(axes):
    y=bottom+((1-i)*(frame_h+gap) if name=='main_ber' else 0)
    ax.set_position([.50/width,y/height,2.90/width,frame_h/height])
    if name=='learning_rate_iterations':ax.yaxis.labelpad=1.5
  elif name=='four_city_ber':
   height=3.22;fig.set_size_inches(width,height,forward=True)
   for i,ax in enumerate(axes):
    row,col=divmod(i,2)
    ax.set_position([(.53+col*1.69)/width,(.40+(1-row)*1.48)/height,1.17/width,1.00/height])
   leg=getattr(fig,'_compact_legend',None)
   if leg is not None:
    hh=getattr(leg,'legend_handles',getattr(leg,'legendHandles',[]));ll=[t.get_text() for t in leg.get_texts()];leg.remove()
    leg=fig.legend(hh,ll,ncols=2,loc='upper center',bbox_to_anchor=(.53,.995),frameon=False,fontsize=10,handlelength=2.8,handletextpad=.25,columnspacing=.5,labelspacing=.15,borderpad=0,borderaxespad=0)
    fig._compact_legend=leg;fig._compact_report.update(placement='outside',columns=2,rows=2)
  else:
   # Preserve horizontal spacing and the established two-row semantic legends.
   height=1.82 if name in {'four_city_ber','paris_P2_P3_ber','cross_city_ber'} else 1.98
   frame_h=1.10
   positions=[a.get_position().frozen() for a in axes]
   fig.set_size_inches(width,height,forward=True)
   for j,(ax,pos) in enumerate(zip(axes,positions)):
    if name in {'paris_P2_P3_ber','cross_city_ber'}:
     ax.set_position([(.53+j*1.69)/width,.40/height,1.17/width,frame_h/height])
    else:
     ax.set_position([pos.x0,.40/height,pos.width,frame_h/height])
   leg=getattr(fig,'_compact_legend',None)
   if leg is not None:
    leg.set_bbox_to_anchor((.5,.995),transform=fig.transFigure)
    leg._loc=9
    fig._compact_report['placement']='outside'
  fig.canvas.draw()
 assert before==signature(fig),name
 frames=[dict(width_in=round(a.get_position().width*fig.get_figwidth(),4),height_in=round(a.get_position().height*fig.get_figheight(),4)) for a in fig.axes]
 fig._physical_geometry=dict(axes_frames=frames,data_sha256=before,data_unchanged=True)
 return Bbox.from_bounds(0,0,fig.get_figwidth(),fig.get_figheight())

def finish_spacing(fig,name):
 """Keep upper legends two points above the frames and separate bar annotations."""
 if not getattr(fig,'_bar_headroom_done',False) and name in {'environment_transition_performance','multipath_delay_ber','speed_ber','rt_delay_ber'}:
  ax=fig.axes[-1];heights=[p.get_height() for p in ax.patches if hasattr(p,'get_height')]
  if heights:
   top=max(heights);ax.set_ylim(0,top*1.85)
   for line in ax.lines:
    y=np.asarray(line.get_ydata(),dtype=float)
    if len(y)==4 and min(y)>top*.2:line.set_ydata(y+top*.27)
   for text in ax.texts:
    if '×' in text.get_text() or text.get_text()=='N/R':
     x,y=text.get_position();text.set_position((x,y+top*.27))
   fig._bar_headroom_done=True
 if getattr(fig,'_upper_legend_packed',False):return
 leg=getattr(fig,'_compact_legend',None)
 if leg is None or fig._compact_report.get('placement')!='outside':return
 fig.canvas.draw();oldheight=fig.get_figheight();width=fig.get_figwidth()
 renderer=fig.canvas.get_renderer();legend_h=leg.get_window_extent(renderer).height/fig.dpi
 positions=[ax.get_position().frozen() for ax in fig.axes]
 top=max(pos.y1*oldheight for pos in positions)
 if name=='paris_san_francisco_tx_rx':
  top=max(top,max(ax.title.get_window_extent(renderer).y1/fig.dpi for ax in fig.axes))
 lower=top+2/72;newheight=lower+legend_h+1.5/72
 fig.set_size_inches(width,newheight,forward=True)
 for ax,pos in zip(fig.axes,positions):ax.set_position([pos.x0,pos.y0*oldheight/newheight,pos.width,pos.height*oldheight/newheight])
 leg.set_bbox_to_anchor((.5,lower/newheight),transform=fig.transFigure);leg._loc=8
 fig._upper_legend_packed=True
 fig._physical_geometry=getattr(fig,'_physical_geometry',{})
 fig._physical_geometry['legend_to_frame_gap_pt']=2
