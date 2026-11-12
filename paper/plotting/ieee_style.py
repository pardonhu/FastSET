"""MATLAB classic default colors, transparent exports, and STIXGeneral labels."""
from pathlib import Path
import json,re
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text
from cycler import cycler
IEEE=dict(green='#0072BD',navy='#D95319',red='#A2142F',light_blue='#F7DDCF',light_green='#D9EAF4',slate='#5D6D7E',grid='#D5D8DC',bg='none',text='#000000',amber='#7E2F8E',purple='#EDB120',cyan='#77AC30')
MATLAB_COLORS=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
NORMAL=MATLAB_COLORS
# IEEE keys remain for compatibility; data colors follow the MATLAB palette.
_save=Figure.savefig
WIDE={'paris_rt_scene_and_positions','adaptation_time_breakdown'}

def apply():
 font_manager.findfont(font_manager.FontProperties(family='STIXGeneral'),fallback_to_default=False)
 mpl.rcParams.update({'font.family':'STIXGeneral','font.serif':['STIXGeneral'],'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.titlesize':10,
 'mathtext.fontset':'custom','mathtext.rm':'STIXGeneral','mathtext.it':'STIXGeneral:italic','mathtext.bf':'STIXGeneral:bold','mathtext.sf':'STIXGeneral','mathtext.cal':'STIXGeneral','mathtext.tt':'STIXGeneral','mathtext.fallback':None,
 'axes.prop_cycle':cycler(color=NORMAL),'axes.facecolor':'none','figure.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,'axes.edgecolor':'black','grid.color':IEEE['grid'],'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','lines.linewidth':2,'lines.markersize':4.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.edgecolor':IEEE['slate']})
 Figure.savefig=savefig

def publication_label(label, figure_name=''):
 # CNN denotes architecture in the backbone comparison and CNN est. baseline.
 if 'pebs' in figure_name:
  label=label.replace('Test BER','BER')
 if 'backbone' in figure_name:
  return label
 return re.sub(r'\bCNN\b(?!\s+(?:est\.|estimator))', 'E2E', label)

def savefig(fig,filename,*args,**kwargs):
 path=Path(filename);name=path.stem
 from curve_markers import prepare_overlap
 prepare_overlap(fig,name)
 from compact_layout import place, audit
 place(fig,name)
 axes=list(fig.axes)
 for ax in list(axes):axes.extend(ax.child_axes)
 target_width=7.16 if name in WIDE else 3.5
 if 'fullrange' in name or '3seeds' in name or 'individual_seeds' in name or name.startswith('A2'):target_width=float(fig.get_figwidth())
 if name=='high_snr_behavior':target_width=7.16
 # Native physical size and font scale match the LaTeX include width.
 font_pt=8 if name=='paris_san_francisco_tx_rx' else 10
 scale=fig.get_figwidth()/target_width
 for text in fig.findobj(Text):
  text.set_text(publication_label(text.get_text(),name))
  text.set_fontfamily('STIXGeneral');text.set_fontsize(font_pt*scale)
 for ax in axes:
  for axis in (ax.xaxis,ax.yaxis):
   ticklabels=axis.get_ticklabels()
   old=[t.get_text() for t in ticklabels]
   new=[publication_label(t,name) for t in old]
   if new!=old:
    axis.set_major_locator(mpl.ticker.FixedLocator(axis.get_ticklocs()))
    axis.set_major_formatter(mpl.ticker.FixedFormatter(new))
  ax.set_facecolor('none')
  for spine in ax.spines.values():spine.set_edgecolor('black')
  for grid in ax.get_xgridlines()+ax.get_ygridlines():grid.set_color(IEEE['grid'])
  for t in [ax.xaxis.label,ax.yaxis.label,ax.xaxis.get_offset_text(),ax.yaxis.get_offset_text(),*ax.get_xticklabels(),*ax.get_yticklabels()]:t.set_color('black')
  ax.tick_params(colors='black',labelsize=font_pt*scale)
  for line in ax.lines:
   if line.get_marker() not in ['',None,'None','none']:
    line.set_markerfacecolor('white');line.set_markeredgecolor(line.get_color())
   if line.get_marker()=='*':line.set_marker('^')
   if line.get_marker() not in ['',None,'None','none'] and line.get_markersize()<8:line.set_markersize(4.5)
 for legend in [*fig.legends,*[ax.get_legend() for ax in axes if ax.get_legend() is not None]]:
  for line in legend.get_lines():
   if line.get_marker() not in ['',None,'None','none']:
    line.set_markerfacecolor('white');line.set_markeredgecolor(line.get_color())
 fig.set_facecolor('none')
 kwargs.update(facecolor='none',edgecolor='none',transparent=True)
 # Tight bounding boxes slightly change the scaling in LaTeX. Correct fonts
 # against the actual crop width; report the effective size for verification.
 if kwargs.get('bbox_inches')=='tight':
  for _ in range(3):
   fig.canvas.draw();width=fig.get_tightbbox(fig.canvas.get_renderer()).width+2*kwargs.get('pad_inches',.1)
   factor=width/target_width
   for text in fig.findobj(Text):text.set_fontsize(font_pt*factor)
 else:width=fig.get_figwidth()
 from publication_geometry import normalize
 physical_bbox=normalize(fig,name)
 if physical_bbox is not None:
  kwargs.update(bbox_inches=physical_bbox,pad_inches=0)
  width=fig.get_figwidth()
  for text in fig.findobj(Text):text.set_fontsize(10)
 if name in {'pebs_pareto_40','pebs_alpha_40'}:
  for parent in fig.axes:
   for inset in parent.child_axes:
    inset.tick_params(labelsize=8,pad=1)
    inset.xaxis.label.set_fontsize(8);inset.yaxis.label.set_fontsize(8)
    inset.xaxis.labelpad=0;inset.yaxis.labelpad=1
    if name=='pebs_pareto_40':inset.set_ylabel('BER',fontsize=8,labelpad=1)
 from publication_geometry import finish_spacing
 finish_spacing(fig,name)
 # Keep several visible dash gaps in each printed legend key. Apply the same
 # pattern to the plotted curves and legend samples after layout rebuilds.
 legends=[*fig.legends,*[ax.get_legend() for ax in axes if ax.get_legend() is not None]]
 compact=getattr(fig,'_compact_legend',None)
 if compact is not None and compact not in legends:legends.append(compact)
 for line in [*[line for ax in axes for line in ax.lines],*[line for leg in legends for line in leg.get_lines()]]:
  if line.get_linestyle()=='--':
   line.set_dashes([2.0,1.8]);line.set_dash_capstyle('butt')
 if physical_bbox is not None:
  from matplotlib.transforms import Bbox
  kwargs.update(bbox_inches=Bbox.from_bounds(0,0,fig.get_figwidth(),fig.get_figheight()),pad_inches=0)
 # Cell values are annotations, kept at 8.5pt so adjacent heatmap cells remain distinct.
 if name=='pebs_heatmap_40':
  for ax in axes:
   if ax.images:
    for text in ax.texts:text.set_fontsize(8.5*width/target_width)
 if path.suffix=='.pdf':
  fig.canvas.draw();width=(fig.get_tightbbox(fig.canvas.get_renderer()).width+2*kwargs.get('pad_inches',.1)) if kwargs.get('bbox_inches')=='tight' else fig.get_figwidth()
  labels=[t for ax in axes for t in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()] if t.get_text() and t.get_visible()]
  metrics={'name':name,'latex_width_inches':target_width,'pdf_width_inches':width,'axis_fonts_pt_at_latex_size':[round(t.get_fontsize()*target_width/width,3) for t in labels], 'axis_font_family':'STIXGeneral','axis_text_black':all(t.get_color()=='black' for t in labels),'background':'transparent','palette':'MATLAB classic R2014b-R2024b','method_colors':{'FastSET':'#0072BD','E2E':'#D95319'}}
  if hasattr(fig,'_physical_geometry'):metrics.update(fig._physical_geometry)
  if name=='backbone_time':
   metrics.pop('method_colors',None)
   metrics['backbone_colors']={'CNN':'#0072BD','Transformer':'#D95319','MLP':'#7E2F8E'}
   metrics['method_linestyles']={'FastSET':'solid','E2E':'dashed'}
  if hasattr(fig,'_compact_report') and 'curve_styles' in fig._compact_report:
   metrics['bar_method_colors']=metrics.pop('method_colors')
   metrics['ber_curve_styles']=fig._compact_report['curve_styles']
  path.with_suffix('.style.json').write_text(json.dumps(metrics,indent=2)+'\n')
 if path.suffix=='.pdf':audit(fig,path)
 from curve_markers import lift_markers
 lift_markers(fig,name)
 if path.suffix=='.pdf' and hasattr(fig,'_marker_report'):
  path.with_suffix('.markers.json').write_text(json.dumps(fig._marker_report,indent=2)+'\n')
 # Portable fonts can have different glyph extents from the author workstation.
 # Preserve labels and legends at the export boundary without altering data.
 kwargs.update(bbox_inches="tight",pad_inches=.04)
 return _save(fig,filename,*args,**kwargs)
