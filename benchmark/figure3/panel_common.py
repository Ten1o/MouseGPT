"""Shared print styles and vector panel export for Extended Data Figure3."""
from pathlib import Path
import base64,io,xml.etree.ElementTree as ET,re,logging,os,warnings
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import FancyArrowPatch,Rectangle
from fontTools import subset
CODE=Path(__file__).resolve().parent
REPOSITORY=CODE.parents[1]
ROOT=Path(os.environ.get('MOUSEGPT_FIGURE_OUTPUT', REPOSITORY/'outputs'/'benchmark-figure3')).resolve()
DATA=Path(os.environ.get('MOUSEGPT_FIGURE_DATA', REPOSITORY/'data'/'benchmark'/'figure3')).resolve()
BLUE='#3160AC';ORANGE='#ED7222';GREEN='#648B69';INK='#252525';PALE='#A6CEE3'
def resolve_font(weight, variable):
 configured=os.environ.get(variable)
 if configured:
  path=Path(configured).expanduser().resolve()
  if not path.is_file():raise FileNotFoundError(f'{variable} does not name a font file: {path}')
  font_manager.fontManager.addfont(str(path))
  return path
 try:return Path(font_manager.findfont(font_manager.FontProperties(family='Arial',weight=weight),fallback_to_default=False))
 except ValueError:
  warnings.warn('Arial unavailable; using DejaVu Sans. Set MOUSEGPT_FONT_REGULAR and MOUSEGPT_FONT_BOLD for paper typography.')
  return Path(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight)))
FONT=resolve_font('normal','MOUSEGPT_FONT_REGULAR');BOLD=resolve_font('bold','MOUSEGPT_FONT_BOLD')
FONT_FAMILY=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':6.3,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6.1,
 'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':0,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix','mathtext.default':'it',
 'text.color':'black','axes.labelcolor':'black','savefig.facecolor':'white'})
class Panel:
 def __init__(self,letter,title,width,height):
  self.letter=letter;self.w=width;self.h=height
  self.fig=plt.figure(figsize=(width/25.4,height/25.4),facecolor='white')
  self.text(0,0,letter,size=8,weight='bold');self.text(4.5,.2,title,size=7.3,weight='bold')
 def text(self,x,y,t,size=6.3,weight='normal',ha='left',va='top',**kw):
  return self.fig.text(x/self.w,1-y/self.h,t,fontsize=size,fontweight=weight,ha=ha,va=va,**kw)
 def ax(self,x,y,w,h):return self.fig.add_axes([x/self.w,(self.h-y-h)/self.h,w/self.w,h/self.h])
 def line(self,x1,y1,x2,y2,color='black',lw=.5,**kw):
  self.fig.add_artist(plt.Line2D([x1/self.w,x2/self.w],[1-y1/self.h,1-y2/self.h],transform=self.fig.transFigure,color=color,lw=lw,**kw))
 def arrow(self,x1,y1,x2,y2,color='black',lw=.6):
  self.fig.add_artist(FancyArrowPatch((x1/self.w,1-y1/self.h),(x2/self.w,1-y2/self.h),transform=self.fig.transFigure,arrowstyle='-|>',mutation_scale=5,linewidth=lw,color=color))
 def rect(self,x,y,w,h,fill='white',edge='#B6B6B6',lw=.5):
  self.fig.add_artist(Rectangle((x/self.w,1-(y+h)/self.h),w/self.w,h/self.h,transform=self.fig.transFigure,facecolor=fill,edgecolor=edge,lw=lw))
 def save(self):
  out=ROOT/'panels';out.mkdir(parents=True,exist_ok=True)
  self.fig.savefig(out/f'panel_{self.letter}.pdf',dpi=600)
  self.fig.savefig(out/f'panel_{self.letter}.svg',dpi=600)
  self.fig.savefig(out/f'panel_{self.letter}.png',dpi=220)
  embed_svg_fonts(out/f'panel_{self.letter}.svg')
  plt.close(self.fig)
def embed_svg_fonts(path):
 svg=path.read_text();tree=ET.fromstring(svg);ns={'svg':'http://www.w3.org/2000/svg'}
 chars=''.join(''.join(e.itertext()) for e in tree.findall('.//svg:text',ns))
 specs=[(FONT,FONT_FAMILY,'400','normal'),(BOLD,FONT_FAMILY,'700','normal')]
 # Mathtext uses STIXGeneral for italic variables and STIXSize for operators.
 for f in font_manager.fontManager.ttflist:
  if f.name.startswith('STIX'):
   specs.append((Path(f.fname),f.name,str(f.weight if isinstance(f.weight,int) else (700 if f.weight=='bold' else 400)),f.style))
 css=[];seen=set()
 for fp,name,weight,style in specs:
  if (name,weight,style) in seen:continue
  seen.add((name,weight,style));opts=subset.Options();font=subset.load_font(str(fp),opts)
  sub=subset.Subsetter(options=opts);sub.populate(text=chars);sub.subset(font);buf=io.BytesIO();font.save(buf)
  css.append("@font-face{font-family:'"+name+"';font-weight:"+weight+";font-style:"+style+";src:url(data:font/ttf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('truetype');}")
 svg=svg.replace('<defs>','<defs>\n<style type="text/css"><![CDATA['+'\n'.join(css)+']]></style>',1);path.write_text(svg)
def clean_axes(ax,xlim=(0,100),ticks=(0,50,100)):
 ax.spines[['top','right']].set_visible(False);ax.set_xlim(*xlim);ax.set_xticks(ticks);ax.grid(False);ax.tick_params(pad=2)
def interval(ax,x,y,lo,hi,color=INK,face=None):
 return ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',markersize=3.3,mfc=face or color,mec=color,mew=.6,ecolor=color,elinewidth=.7,capsize=1.8,capthick=.7,zorder=5)
