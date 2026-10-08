"""Matched metric intervals and ranked dot comparisons from frozen source CSVs."""
import csv
from matplotlib import pyplot as plt
from panel_common import Panel,DATA,BLUE,ORANGE,GREEN,INK,PALE,clean_axes,interval

def rows(name):
 with (DATA/'source_data'/name).open(newline='') as f:return list(csv.DictReader(f))
def label(r):
 s=r['model'].replace('gpt-','GPT-').replace('-astra',' Astra').replace('-sol',' Sol').replace('-luna',' Luna');e=r['reasoning_effort']
 return s+(f' ({e})' if e not in ('not_requested','none','') else '')
def mean_label(ax,x,y,side='right',color=INK):
 # Labels sit outside the paired endpoints with a fixed 1.5 mm gap.
 gap=1.5*72/25.4
 return ax.annotate(f'{x:.1f}',(x,y),xytext=(-gap if side=='left' else gap,0),
  textcoords='offset points',ha='right' if side=='left' else 'left',va='center',
  fontsize=6.3,color=color,annotation_clip=False,zorder=6)

def interval_side_label(ax,mean,y,low,high,side):
 # Put each mean outside its interval, leaving the connecting line unobstructed.
 gap=1.3*72/25.4
 return ax.annotate(f'{mean:.1f}',(low if side=='left' else high,y),
  xytext=(-gap if side=='left' else gap,0),textcoords='offset points',
  ha='right' if side=='left' else 'left',va='center',fontsize=6.3,
  color=INK,annotation_clip=False,zorder=6)

def check_mean_labels(panel,artists):
 panel.fig.canvas.draw();renderer=panel.fig.canvas.get_renderer()
 bounds=panel.fig.bbox; boxes=[a.get_window_extent(renderer) for a in artists]
 # Allow no out-of-panel labels and no overlap between any two numeric labels.
 for box in boxes:
  assert bounds.x0<=box.x0 and box.x1<=bounds.x1 and bounds.y0<=box.y0 and box.y1<=bounds.y1
 for i,a in enumerate(boxes):
  for b in boxes[i+1:]:assert not a.overlaps(b),'Numeric labels overlap'

def main():
 metrics=rows('metric_summary.csv');ranked=sorted([r for r in metrics if r['metric']=='benchmark_index' and r['collection']=='all'],key=lambda r:-float(r['mean']))
 lo=next(r for r in ranked if r['model']=='gpt-4o');hi=ranked[0]
 assert hi['model']=='gpt-6-astra' and hi['reasoning_effort']=='max'
 lookup={(r['run_id'],r['metric']):r for r in metrics}
 p=Panel('d','Performance by rubric dimension',72,33)
 handles=[plt.Line2D([],[],marker='o',ms=3.4,mfc=face,mec=color,lw=0)
          for color,face in [(BLUE,PALE),(GREEN,GREEN)]]
 p.fig.legend(handles,['GPT-4o','GPT-6 Astra (max)'],loc='center',
  bbox_to_anchor=(.5,1-6.5/p.h),bbox_transform=p.fig.transFigure,
  ncol=2,frameon=False,fontsize=6,handlelength=.6,handletextpad=.65,columnspacing=1.7)
 # Balance the plot within its panel and reserve right space for interval labels.
 ax=p.ax(23,11,41,15);clean_axes(ax);ax.set_ylim(3.55,-.55)
 numeric_labels=[]
 keys=['factual_precision','required_fact_recall','structure_score','instruction_score']
 for i,key in enumerate(keys):
  a,b=[lookup[r['run_id'],key] for r in (lo,hi)]
  ax.plot([float(a['mean']),float(b['mean'])],[i,i],color='#C7CDCC',lw=.6,zorder=1)
  for row,c,face,side in [(a,BLUE,PALE,'left'),(b,GREEN,GREEN,'right')]:
   x,l,h=[float(row[k]) for k in ('mean','ci95_low','ci95_high')]
   interval(ax,x,i,l,h,c,face)
   numeric_labels.append(interval_side_label(ax,x,i,l,h,side))
 ax.set_yticks(range(4),['Factual correctness','Coverage','Structure','Instructions']);ax.tick_params(axis='y',labelsize=6)
 ax.set_xlabel('Rubric score (0–100)',labelpad=2);check_mean_labels(p,numeric_labels);p.save()
 p=Panel('e','Overall model comparison',98,73)
 p.text(32,6,'Mean and 95% cluster-bootstrap CI',size=6)
 ax=p.ax(32,12,56,53);clean_axes(ax,(45,95),(45,55,65,75,85,95));ax.set_ylim(13.65,-.65)
 for i,r in enumerate(ranked):
  col=BLUE if r['run_id']==lo['run_id'] else GREEN if r['run_id']==hi['run_id'] else INK
  face=PALE if r['run_id']==lo['run_id'] else col
  x,l,h=[float(r[k]) for k in ['mean','ci95_low','ci95_high']];assert 45<l<=x<=h<95
  interval(ax,x,i,l,h,col,face);ax.text(h+2*(50/56),i,f'{x:.1f}',ha='left',va='center',fontsize=6.5,clip_on=False)
 ax.set_yticks(range(14),[label(r) for r in ranked]);ax.tick_params(axis='y',labelsize=6.3,pad=3)
 ax.set_xlabel('MouseGPT Benchmark Index',labelpad=2);p.save()
 p=Panel('f','Challenging frames: lowest 10%',72,73)
 for x,name,c,face in [(1,'All frames (435)',INK,INK),(31,'Lowest-scoring 10% (44)',ORANGE,'white')]:
  p.fig.add_artist(plt.Line2D([(x+1)/p.w],[1-7/p.h],marker='o',ms=3.2,mfc=face,mec=c,mew=.7,lw=0,transform=p.fig.transFigure))
  p.text(x+4,7,name,size=6,va='center')
 subset={r['run_id']:r for r in rows('panel_f_dot_means.csv')};numeric_labels=[]
 ax=p.ax(4,12,66,53);clean_axes(ax,(30,95),(30,50,70,95));ax.set_ylim(13.65,-.65)
 for i,r in enumerate(ranked):
  s=subset[r['run_id']];a=float(s['all435_benchmark_index']);b=float(s['bottom44_benchmark_index']);assert 30<a<95 and 30<b<95
  ax.plot([a,b],[i,i],color='#999999',lw=.6,zorder=1)
  ax.plot(a,i,'o',ms=3.2,mfc=INK,mec=INK,mew=.6,zorder=4)
  ax.plot(b,i,'o',ms=3.5,mfc='white',mec=ORANGE,mew=.8,zorder=5)
  numeric_labels.extend([mean_label(ax,b,i,side='left',color='#A04B11'),mean_label(ax,a,i,side='right')])
 ax.set_yticks([]);ax.set_xlabel('MouseGPT Benchmark Index',labelpad=2);check_mean_labels(p,numeric_labels);p.save()
if __name__=='__main__':main()
