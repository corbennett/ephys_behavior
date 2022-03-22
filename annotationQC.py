# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:45:46 2022

@author: svc_ccg
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42


annotationInfo = pd.read_excel(r'C:\Users\svc_ccg\Desktop\VBN_annotation.xlsx')

annotationPath = r'\\allen\programs\braintv\workgroups\neuralcoding\corbettb\VBN_production'

for d in (r'\\allen\programs\mindscope\workgroups\np-behavior\processed_ALL\524760',):#annotationInfo['OPT directory']:
    pdf = PdfPages(os.path.join(d,'annotation_qc_plots.pdf'))
    visModFiles = glob.glob(os.path.join(d,'channel_visual_modulation*.npy'))
    for f in visModFiles:
        match = re.search('channel_visual_modulation_(\d*)_(\d{6,6})_(\d{8,8})',os.path.basename(f))
        limsId,mouseId,expDate = [match.group(i) for i in (1,2,3)]
        
        visModDict = np.load(f, allow_pickle=True)[()]
        
        fig = plt.figure(figsize=(18,9))
        fig.text(0.5,0.99,mouseId+' '+expDate,ha='center',va='top')
        for probe in visModDict:
            visMod = visModDict[probe][::-1]
            visMod /= visMod.max()
            
            unitDensity = np.load(os.path.join(d,'unitdensity_probe'+probe+'_'+limsId+'_'+mouseId+'_'+expDate+'.npy'))
            unitDensity = unitDensity[::-1]
            unitDensity /= unitDensity.max()
            
            ccf = pd.read_csv(os.path.join(annotationPath,limsId+'_'+mouseId+'_'+expDate+'_probe'+probe+'_sorted','ccf_regions.csv'))
            
            depth = np.array(ccf['D/V'])[::-1]
            depth /= depth.max()
            corticalDepth = np.array(ccf['cortical_depth'])[::-1]
            cortexStart,cortexEnd = np.where(corticalDepth>=0)[0][[0,-1]]
            
            ax = fig.add_subplot(6,1,'ABCDEF'.find(probe)+1)
            ax.plot([cortexStart]*2,[0,1],color='0.5')
            ax.plot([cortexEnd+1]*2,[0,1],color='0.5')
            ax.plot(depth,color='0.5')
            ax.plot(unitDensity,'k')
            ax.plot(visMod,'b')
            regions = np.array(ccf['structure_acronym'])[::-1]
            r = regions[0]
            start = 0
            regionsPlotted = 0
            for i in range(depth.size):
                if i == depth.size-1 or regions[i+1] != r:
                    clr = plt.cm.Set2((regionsPlotted % 8)/8)
                    ax.add_patch(matplotlib.patches.Rectangle([start,-0.2],width=i+1-start,height=0.2,facecolor=clr,edgecolor=clr))
                    ax.text(start+(i+1-start)/2,-0.1,r,ha='center',va='center',fontsize=6)
                    regionsPlotted += 1
                    if i < depth.size-1:
                        start = i+1
                        r = regions[i+1]
            for side in ('right','top','left','bottom'):
                ax.spines[side].set_visible(False)
            ax.tick_params(right=False,top=False,left=False,bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0,depth.size])
            ax.set_ylim([-0.2,1])
            ax.set_ylabel(probe,rotation=0)
        plt.tight_layout()
        
        fig.savefig(pdf,format='pdf')
    pdf.close()
    plt.close('all')
            
