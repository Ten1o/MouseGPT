#!/usr/bin/env python3
"""Build individual vector panels, compose exact-print HTML/SVG/PDF, and raster exports.
No inference, judging or database writes. SVG/PDF panels remain independently editable.
"""
from pathlib import Path
import argparse,copy,hashlib,json,re,subprocess,sys,xml.etree.ElementTree as ET,logging
from pypdf import PdfReader,PdfWriter,Transformation
from PIL import Image
from panel_common import ROOT,CODE,DATA,FONT_FAMILY,embed_svg_fonts
from verify_inputs import verify
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
W,H=180,156;PT=72/25.4
# One layout manifest controls HTML/CSS and vector assembly; all units millimetres.
LAYOUT={'a':[3,3,98,31],'b':[105,3,72,31],'c':[3,40,98,33],
        'd':[105,40,72,33],'e':[3,80,98,73],'f':[105,80,72,73]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fonts_of(page):
 found=[]
 for ref in page['/Resources'].get('/Font',{}).values():
  f=ref.get_object();desc=(f.get('/DescendantFonts') or [f])[0].get_object();fd=desc.get('/FontDescriptor');fd=fd.get_object() if fd else {}
  found.append({'font':str(f.get('/BaseFont')),'subtype':str(f.get('/Subtype')),'embedded':bool(fd.get('/FontFile2') or fd.get('/FontFile3'))})
 return found

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--compose-only',action='store_true');args=parser.parse_args()
 ROOT.mkdir(parents=True,exist_ok=True)
 if not args.compose_only:
  verify(DATA)
  for source in ['panel_a.py','panel_b.py','panel_c.py','panels_results.py']:
   subprocess.run([sys.executable,str(CODE/source)],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
 (ROOT/'layout.json').write_text(json.dumps({'width_mm':W,'height_mm':H,'panels_mm':LAYOUT},indent=2)+'\n')
 writer=PdfWriter();page=writer.add_blank_page(width=W*PT,height=H*PT)
 SVG='http://www.w3.org/2000/svg';ET.register_namespace('',SVG);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
 merged=ET.Element('{'+SVG+'}svg',{'width':f'{W}mm','height':f'{H}mm','viewBox':f'0 0 {W*PT} {H*PT}','version':'1.1'})
 ET.SubElement(merged,'{'+SVG+'}defs')
 ET.SubElement(merged,'{'+SVG+'}rect',{'width':'100%','height':'100%','fill':'white'})
 htmlpanels=[];panelmeta={}
 for key,(x,y,w,h) in LAYOUT.items():
  panelpath=ROOT/'panels'/f'panel_{key}.pdf';rp=PdfReader(panelpath).pages[0]
  assert abs(float(rp.mediabox.width)/PT-w)<.02 and abs(float(rp.mediabox.height)/PT-h)<.02
  assert x>=0 and y>=0 and x+w<=W and y+h<=H
  fs=fonts_of(rp);assert fs and all(q['embedded'] and q['subtype']!='/Type3' for q in fs)
  page.merge_transformed_page(rp,Transformation().translate(x*PT,(H-y-h)*PT),over=True)
  tree=ET.fromstring((ROOT/'panels'/f'panel_{key}.svg').read_text())
  for par in tree.iter():
   for child in list(par):
    if child.tag.endswith('metadata') or (child.tag.endswith('style') and '@font-face' in (child.text or '')):par.remove(child)
  mapping={e.attrib['id']:key+'_'+e.attrib['id'] for e in tree.iter() if 'id' in e.attrib}
  for e in tree.iter():
   for attr,val in list(e.attrib.items()):
    if attr=='id':val=mapping[val]
    else:
     for old,new in mapping.items():val=val.replace('url(#'+old+')','url(#'+new+')');val=('#'+new) if val=='#'+old else val
    e.set(attr,val)
  group=ET.SubElement(merged,'{'+SVG+'}g',{'id':'panel_'+key,'transform':f'translate({x*PT} {y*PT})'})
  for child in list(tree):group.append(copy.deepcopy(child))
  htmlpanels.append(f'<object class="panel" aria-label="Panel {key}" data="panels/panel_{key}.svg" type="image/svg+xml" style="left:{x}mm;top:{y}mm;width:{w}mm;height:{h}mm"></object>')
  panelmeta[key]={'position_size_mm':[x,y,w,h],'pdf_sha256':sha(panelpath),'svg_sha256':sha(ROOT/'panels'/f'panel_{key}.svg'),'fonts':fs}
 writer.add_metadata({'/Title':'MouseGPT Extended Data Figure 3','/Subject':'Unified 435-frame benchmark; aligned rubric dimensions'})
 with (ROOT/'ExtendedFigure3.pdf').open('wb') as f:writer.write(f)
 ET.ElementTree(merged).write(ROOT/'ExtendedFigure3.svg',encoding='unicode',xml_declaration=True)
 embed_svg_fonts(ROOT/'ExtendedFigure3.svg')
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MouseGPT · Extended Data Figure 3</title>
<style>*{box-sizing:border-box}body{margin:0;background:#ebedee;font:14px Arial,sans-serif;color:#222}header{padding:18px 24px;display:flex;gap:20px;align-items:center}header strong{margin-right:auto}header a{color:#234d7b}main{margin:0 auto 30px;width:180mm;height:156mm}.plate{position:relative;width:180mm;height:156mm;background:white;box-shadow:0 2px 14px #0002;transform-origin:top left}.panel{position:absolute;border:0}.outline .panel{outline:1px dashed #aaa}select,button{font:inherit;padding:4px 8px} @page{size:180mm 156mm;margin:0}@media print{header{display:none}body{background:white}main{margin:0;width:180mm!important;height:156mm!important}.plate{box-shadow:none;transform:none!important}.outline .panel{outline:0}}</style>
<header><strong>Extended Data Figure 3 · editable vector composition</strong><a href="ExtendedFigure3.pdf">PDF</a><a href="ExtendedFigure3.svg">SVG</a><label>Scale <select id="scale"><option value="1">100%</option><option value="1.5">150%</option><option value="2">200%</option></select></label><button id="bounds">Panel bounds</button></header><main id="canvas"><div class="plate" id="plate">'''+''.join(htmlpanels)+'''</div></main><script>document.getElementById('scale').onchange=e=>{let s=Number(e.target.value);document.getElementById('plate').style.transform=`scale(${s})`;let c=document.getElementById('canvas');c.style.width=180*s+'mm';c.style.height=156*s+'mm'};document.getElementById('bounds').onclick=()=>document.getElementById('plate').classList.toggle('outline');</script></html>'''
 (ROOT/'composition.html').write_text(html)
 # Raster derivatives come from the vector PDF, not from a browser screenshot.
 for dpi,stem in [(300,'ExtendedFigure3'),(600,'.raster600')]:
  subprocess.run(['pdftoppm','-r',str(dpi),'-png','-singlefile',str(ROOT/'ExtendedFigure3.pdf'),str(ROOT/stem)],check=True)
 im=Image.open(ROOT/'.raster600.png').convert('RGB');im.save(ROOT/'ExtendedFigure3.tiff',compression='tiff_lzw',dpi=(600,600));(ROOT/'.raster600.png').unlink()
 reader=PdfReader(ROOT/'ExtendedFigure3.pdf');fs=fonts_of(reader.pages[0]);assert len(reader.pages)==1 and all(f['embedded'] and f['subtype']!='/Type3' for f in fs)
 report={'font_family':FONT_FAMILY,'data_directory':str(DATA),'revision':'v8-aligned-rubric-dimensions','dimensions_mm':[W,H],'pdf_pages':1,'vector_assembly':'pypdf transforms of independently editable panel PDFs; no screenshot flattening','html_layout':'composition.html uses same layout.json millimetre positions','fonts':fs,'panels':panelmeta,'svg_editable_text_nodes':len(ET.parse(ROOT/'ExtendedFigure3.svg').findall('.//{'+SVG+'}text')),'outputs':{},'tiff':{'size_px':list(im.size),'dpi':[600,600],'compression':'tiff_lzw','bytes':(ROOT/'ExtendedFigure3.tiff').stat().st_size},'validation_scope':'Panel dimensions, placement, embedded fonts, output checksums, and export formats.'}
 assert report['tiff']['bytes']<10_000_000
 for name in ['ExtendedFigure3.pdf','ExtendedFigure3.svg','ExtendedFigure3.png','ExtendedFigure3.tiff','composition.html','layout.json']:report['outputs'][name]={'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size}
 (ROOT/'export_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'panels':len(LAYOUT),'tiff_bytes':report['tiff']['bytes'],'pdf_bytes':report['outputs']['ExtendedFigure3.pdf']['bytes']}))
if __name__=='__main__':main()
