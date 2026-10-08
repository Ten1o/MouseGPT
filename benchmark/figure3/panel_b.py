"""Compact horizontal strip of full native paired inputs, without cropping."""
import json
from PIL import Image
from panel_common import Panel,DATA

def main():
 p=Panel('b','Behavioral examples',72,31)
 cases=json.loads((DATA/'example_manifest.json').read_text())['selection']
 for i,case in enumerate(cases):
  left=0 if i==0 else 37
  for j,im in enumerate(case['images']):
   ax=p.ax(left+17.5*j,6.5,16.5,16.5);ax.imshow(Image.open(DATA/im['export_path']),interpolation='none');ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_linewidth(.3);sp.set_color('#686868')
  p.text(left+17,25.2,'Lying belly-down' if i==0 else 'Hopping posture',size=6.3,ha='center')
 p.save()
if __name__=='__main__':main()
