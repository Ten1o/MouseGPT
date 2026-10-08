import numpy as np

from glob import glob

import xlwt

for npy in sorted(glob("/home/vrlab/Downloads/*.npy")):
    a=np.load(npy)
    
    print(a)
    
    wt=xlwt.Workbook()
    ws=wt.add_sheet("Sheet1")
    for i in range(len(a)):
        ws.write(i,0,a[i])
        
    wt.save(f'/home/vrlab/Downloads/{npy.split("/")[-1]}.xls')