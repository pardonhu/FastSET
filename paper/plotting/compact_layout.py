"""Compact paper legends without changing measured points or axis limits."""
import math,json
from pathlib import Path
import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
HEIGHTS={'main_ber':3.75,'adaptation_time_comparison':2.05,'environment_transition_performance':2.25,'multipath_delay_ber':2.25,'speed_ber':2.25,'backbone_time':2.4,'A2B_ablation_3seeds_log10':2.40,'A2C_ablation_3seeds_log10':2.40,'learning_rate_iterations':2.0,'width_depth_combined':2.1,'four_city_ber':1.95,'paris_P2_P3_ber':1.95,'cross_city_ber':1.95,'urban_adaptation_time':2.35,'rt_delay_ber':2.25,'paris_san_francisco_tx_rx':1.95,'pebs_pareto_40':2.55,'pebs_alpha_40':2.43}
PAIRED={'environment_transition_performance':('Before (B2C)','A2C','B2C'),'multipath_delay_ber':('Before','600 ns','1200 ns'),'rt_delay_ber':('Before','900 ns','1200 ns')}
def collision(fig,box,renderer):
 hits=0
 for ax in fig.axes:
  for line in ax.lines:
   if not line.get_visible():continue
   path=line.get_transform().transform_path(line.get_path())
   if path.intersects_bbox(box,filled=False):hits+=1
  for p in ax.patches:
   if not p.get_visible() or p.get_alpha()==0:continue
   if p.get_transform().transform_path(p.get_path()).intersects_bbox(box,filled=True):hits+=1
  for coll in ax.collections:
   if not coll.get_visible():continue
   offsets=coll.get_offsets()
   if offsets is not None and len(offsets) and (len(offsets)>1 or not np.allclose(offsets,0)):
    pts=coll.get_offset_transform().transform(offsets)
    if box.count_contains(pts):hits+=1
   else:
    for path in coll.get_paths():
     if coll.get_transform().transform_path(path).intersects_bbox(box,filled=True):hits+=1;break
  for t in ax.texts:
   if t.get_visible() and t.get_text() and box.overlaps(t.get_window_extent(renderer)):hits+=1
  for child in ax.child_axes:
   if box.overlaps(child.get_window_extent(renderer)):hits+=1
 return hits


def paired_curve_legend(fig,name):
 """Six explicit curve keys, grouped into two rows; preserve all measured points."""
 from matplotlib.collections import PathCollection
 import hashlib
 ax=fig.axes[0]
 lines=[line for line in ax.lines if not line.get_label().startswith('_')]
 assert len(lines)==6,(name,[line.get_label() for line in lines])
 def digest():
  h=hashlib.sha256()
  for line in ax.lines:h.update(np.asarray(line.get_xydata(),dtype=np.float64).tobytes())
  for coll in ax.collections:h.update(np.asarray(coll.get_offsets(),dtype=np.float64).tobytes())
  return h.hexdigest()
 before=digest()
 if name=='speed_ber':
  row_names=['Before','After'];case_names=['5 m/s','10 m/s','20 m/s']
  groups=[lines[:3],lines[3:]]
 else:
  row_names=['FastSET','E2E'];case_names=list(PAIRED[name])
  groups=[lines[::2],lines[1::2]]
 for r,group in enumerate(groups):
  for c,line in enumerate(group):
   label=line.get_label()
   if name=='speed_ber':
    assert case_names[c] in label and row_names[r].lower() in label,(name,label)
   else:
    assert ('FastSET' if r==0 else 'CNN') in label,(name,label)
    assert ('before' in label.lower()) if c==0 else (case_names[c].split()[0] in label),(name,label)
 # Each measured curve has its own MATLAB color AND marker.
 colors=[['#4DBEEE','#0072BD','#77AC30'],['#A2142F','#D95319','#7E2F8E']]
 markers=[['^','o','D'],['v','s','P']]
 if name=='speed_ber':
  colors=[['#0072BD','#D95319','#7E2F8E']]*2
  markers=[['o','^','s']]*2
 styles={};report=[]
 for r,group in enumerate(groups):
  for c,line in enumerate(group):
   line.set_color(colors[r][c]);line.set_marker(markers[r][c])
   line.set_markerfacecolor('white');line.set_markeredgecolor(colors[r][c]);line.set_markersize(4.5);line.set_linewidth(2)
   ls=('--' if r==0 else '-') if name=='speed_ber' else (':' if c==0 else ('-' if r==0 else '--'))
   line.set_linestyle(ls);styles[line]=colors[r][c]
   if name=='speed_ber':
    line.set_markevery([i for i,x in enumerate(line.get_xdata()) if (int(round(x))-4-c)%3==0])
   elif name in PAIRED:
    offset,period=(r,3) if c==0 else (2*(c-1)+r,4)
    line.set_markevery([i for i,x in enumerate(line.get_xdata()) if (int(round(x))-4-offset)%period==0 and line.get_ydata()[i]>1/26054400])
   report.append(dict(method_or_phase=row_names[r],condition=case_names[c],color=colors[r][c],marker=markers[r][c],linestyle=ls,label=line.get_label()))
 # Zero-error scatter markers follow their curve in the original drawing order.
 current=None
 for artist in ax.get_children():
  if isinstance(artist,Line2D):current=styles.get(artist)
  elif isinstance(artist,PathCollection) and current:
   artist.set_edgecolor(current);artist.set_facecolor('white')
 assert before==digest(),name
 # Matplotlib fills columns first: two row headings followed by three curve-key columns.
 handles=[Line2D([],[],ls='none',label=row_names[0]),Line2D([],[],ls='none',label=row_names[1])]
 labels=row_names.copy()
 for c,case in enumerate(case_names):
  for r in range(2):
   line=groups[r][c]
   handles.append(Line2D([],[],color=line.get_color(),marker=line.get_marker(),ls=line.get_linestyle(),lw=2,ms=4.5,mfc='white',mec=line.get_color()))
   labels.append(case)
 leg=fig.legend(handles,labels,ncols=4,frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,.995),handlelength=2.0,handletextpad=.22,columnspacing=.28,labelspacing=.25,borderpad=0,borderaxespad=0)
 # Hide unused handle space in the method/phase-heading column.
 heading_column=leg._legend_handle_box.get_children()[0]
 for row in heading_column.get_children():
  row.get_children()[0].set_width(0)
  row.sep=0
 fig.canvas.draw();renderer=fig.canvas.get_renderer();box=leg.get_window_extent(renderer)
 assert box.width<=fig.bbox.width*.99,(name,'legend too wide',box.width,fig.bbox.width)
 top=box.y0/fig.bbox.height-.025
 roots=[a for a in fig.axes if a.get_label()!='<colorbar>']
 oldtop=max(a.get_position().y1 for a in roots);bottom=min(a.get_position().y0 for a in roots)
 for a in roots:
  pos=a.get_position();scale=(top-bottom)/(oldtop-bottom)
  a.set_position([pos.x0,bottom+(pos.y0-bottom)*scale,pos.width,pos.height*scale])
 fig._compact_legend=leg
 fig._compact_report=dict(placement='outside',columns=4,rows=2,labels=labels,curve_styles=report,curve_data_sha256=before,curve_data_unchanged=True,old_height=3.0,new_height=fig.get_figheight())

def place(fig,name):
 if name not in HEIGHTS or getattr(fig,'_compact_applied',False):return
 fig._compact_applied=True
 oldheight=fig.get_figheight();fig.set_size_inches(fig.get_figwidth(),HEIGHTS[name],forward=True)
 originals=list(fig.legends)+[a.get_legend() for a in fig.axes if a.get_legend() is not None]
 if not originals:return
 # The paper uses one common legend per figure. Retain every displayed semantic key.
 handles=[];labels=[]
 for legend in originals:
  hh=getattr(legend,'legend_handles',getattr(legend,'legendHandles',[]))
  for h,t in zip(hh,legend.get_texts()):
   if t.get_text() not in labels:handles.append(h);labels.append(t.get_text())
  legend.remove()
 if name in PAIRED or name=='speed_ber':
  paired_curve_legend(fig,name)
  return
 if name=='backbone_time':
  families=[('CNN','#0072BD','o'),('Transformer','#D95319','s'),('MLP','#7E2F8E','D')]
  for ax in fig.axes:
   for line in ax.lines:
    for family,color,marker in families:
     if line.get_label() in [family+' FastSET',family+' E2E']:
      line.set_color(color);line.set_marker(marker)
      line.set_linestyle('-' if line.get_label().endswith(' FastSET') else '--')
  families=[families[i] for i in [0,2,1]]  # CNN, MLP, Transformer; paired method columns.
  handles=[Line2D([],[],color=color,ls=ls,marker=marker,mfc='white',label=family+' '+method) for method,ls in [('FastSET','-'),('E2E','--')] for family,color,marker in families]
  labels=[h.get_label() for h in handles]
 if name=='paris_san_francisco_tx_rx':
  labels=['TX','Source RX','P1','P2']
  for h in handles:
   if hasattr(h,'set_markersize'):h.set_markersize(6)
  leg=fig.legend(handles,labels,ncols=4,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False,fontsize=8,handlelength=1,handletextpad=.2,columnspacing=.6,borderpad=0)
  fig._compact_legend=leg
  fig._compact_report=dict(placement='outside',columns=4,rows=1,labels=labels,old_height=oldheight,new_height=fig.get_figheight())
  return
 # Presentation-only capitalization; data IDs remain unchanged.
 labels=[s.replace('est. only','Est. only').replace('dem. only','Dem. only').replace('est 5','Est. 5').replace('est 1','Est. 1').replace('Joint (joint PT)','Joint') for s in labels]
 n=len(labels);font=10 if name!='paris_san_francisco_tx_rx' else 8
 base=dict(handles=handles,labels=labels,frameon=False,fontsize=font,handlelength=2.8,handletextpad=.32,columnspacing=.6,labelspacing=.18,borderpad=.22,borderaxespad=.25)
 if name=='backbone_time':base['handlelength']=2.6
 # Try clear space within each actual panel, never covering data or annotations.
 # Inset-heavy PEBS figures keep a compact outside legend to preserve the zoom.
 fig.canvas.draw();original_positions=[a.get_position().frozen() for a in fig.axes]
 if name not in ['pebs_pareto_40','pebs_alpha_40','paris_san_francisco_tx_rx']:
  try:fig.subplots_adjust(top=.96)
  except ValueError:pass
  fig.canvas.draw();renderer=fig.canvas.get_renderer()
  cols=[2] if name=='backbone_time' else list(dict.fromkeys([math.ceil(n/2),math.ceil(n/3),1]))
  for ax in fig.axes:
   if ax.get_label()=='<colorbar>':continue
   for col in cols:
    if math.ceil(n/col)>4:continue
    for loc,anchor in [(q,None) for q in ['upper right','lower left','upper left','lower right','center right','center left']]+[('upper right',(x,y)) for x,y in [(0.98,.82),(.98,.68),(.98,.54),(.8,.90),(.8,.72),(.65,.90),(.65,.72)]]:
     extra={} if anchor is None else dict(bbox_to_anchor=anchor,bbox_transform=ax.transAxes)
     leg=Legend(ax,loc=loc,ncols=col,**extra,**base);ax.add_artist(leg);box=leg.get_window_extent(renderer);bounds=ax.get_window_extent(renderer)
     fits=box.x0>=bounds.x0+1 and box.x1<=bounds.x1-1 and box.y0>=bounds.y0+1 and box.y1<=bounds.y1-1
     clear=fits and collision(fig,box.padded(3),renderer)==0
     if clear:
      fig._compact_legend=leg;fig._compact_report=dict(placement='inside',columns=col,rows=math.ceil(n/col),labels=labels,data_overlaps=0,old_height=oldheight,new_height=fig.get_figheight());return
     leg.remove()
  for ax,pos in zip(fig.axes,original_positions):ax.set_position(pos)
 # Two rows at most outside. Tight, short handles keep labels at text size.
 ncol=2 if name=='backbone_time' else n
 leg=fig.legend(loc='upper center',bbox_to_anchor=(.54,.995),ncols=ncol,**base)
 fig.canvas.draw()
 if leg.get_window_extent(fig.canvas.get_renderer()).width>fig.bbox.width*.96:
  leg.remove();ncol=2 if name=='backbone_time' else math.ceil(n/2)
  leg=fig.legend(loc='upper center',bbox_to_anchor=(.54,.995),ncols=ncol,**base)
 # Reserve the exact legend height plus a small gap above the topmost axes.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bbox=leg.get_window_extent(renderer)
 top=bbox.y0/fig.bbox.height-(.05 if name=='width_depth_combined' else .025)
 roots=[a for a in fig.axes if a.get_label()!='<colorbar>']
 oldtop=max(a.get_position().y1 for a in roots)
 bottom=min(a.get_position().y0 for a in roots)
 # Snapshot positions before resizing: twinx axes share positions and would
 # otherwise apply the same vertical scale twice.
 positions=[ax.get_position().frozen() for ax in roots]
 for ax,p in zip(roots,positions):
  scale=(top-bottom)/(oldtop-bottom)
  ax.set_position([p.x0,bottom+(p.y0-bottom)*scale,p.width,p.height*scale])
 fig._compact_legend=leg;fig._compact_report=dict(placement='outside',columns=ncol,rows=math.ceil(n/ncol),labels=labels,old_height=oldheight,new_height=fig.get_figheight())

def audit(fig,path):
 if not hasattr(fig,'_compact_report'):return
 fig.canvas.draw();renderer=fig.canvas.get_renderer();report=dict(fig._compact_report)
 report['legend_bbox_px']=list(fig._compact_legend.get_window_extent(renderer).bounds)
 if report['placement']=='inside':
  report['data_overlaps_after_font_scaling']=collision(fig,fig._compact_legend.get_window_extent(renderer).padded(2),renderer)
 Path(path).with_suffix('.legend.json').write_text(json.dumps(report,indent=2)+'\n')
