import os
from time import time

import numpy as np

from utils.GenerateData import GenerateNoiseTV, Semblance, SyntheticGather
from utils.LoadVelModel import GetVelModel
from utils.PreProcess import interpolation
from utils.VisualTools import PlotSpec, PlotVelField, W_Plot


def VisualFlow(VelDict):
    """VelDict = GetVelModel(root)"""
    # visual the velocity field 
    for line in sorted(list(VelDict['VelModel'].keys())):
        
        LineVel = VelDict['VelModel'][line]
        # get the interpolated velocity curve
        VelField = []
        for cdp in sorted(list(LineVel.keys())):
            label_point = LineVel[cdp]['RMSVel']
            t0Vec = VelDict['t0Vec']
            vVec = LineVel[cdp]['vVec']
            VelField.append(interpolation(label_point, t0Vec, vVec)[:, 1])
        VelField = np.array(VelField).T
        # plot the velocity field 
        PlotVelField(VelField, list(LineVel.keys()), t0Vec, vVec, line, 
                     os.path.join('result', 'hade', 'Line-%s.png' % line))
        print(line, 'done!')


def GenerateFlow(VelDict):
    
    for line in sorted(list(VelDict['VelModel'].keys())):
        for cdp in sorted(list(VelDict['VelModel'][line].keys())):
            label_point = VelDict['VelModel'][line][cdp]['RMSVel']
            t0Vec = VelDict['t0Vec']
            tVec = VelDict['tVec']
            vVec = VelDict['VelModel'][line][cdp]['vVec']
            OffsetVec = VelDict['VelModel'][line][cdp]['offVec']
            start = time()

            ################################################################
            # Synthetic gather
            ################################################################
            # generate t-v noise
            TvSet = GenerateNoiseTV(label_point, t0Vec, vVec, vRange=400, nsr=1.0, LowNoise=5)
            # generate supergather
            GatherAll, _ = SyntheticGather(TvSet, tVec, OffsetVec, 50)
            print('Line %s\tCDP %s\tGenerate a gather cost time:\t%.3fs' % (str(line), str(cdp), time()-start))
            # visual gather
            W_Plot(GatherAll, OffsetVec, tVec, xlab='Trace Index', ylab='Time (ms)', title='Synthetic All Gather',
                   SavePath=os.path.join('result', 'SyntheticData', 'line-%s-cdp-%s-Gather.png') % (str(line), str(cdp)))

            ################################################################
            # Calculate the specturm using the semblance method
            ################################################################
            # generate spectrum
            start = time()
            Spectrum = Semblance(GatherAll, tVec, OffsetVec, t0Vec, vVec)
            print('Line %s\tCDP %s\tGenerate a spectrum cost time:\t%.3fs' % (str(line), str(cdp), time()-start))
            # visual spec
            PlotSpec(Spectrum, t0Vec, vVec, 'Synthetic Spectrum',
                     SavePath=os.path.join('result', 'SyntheticData', 'line-%s-cdp-%s-Spectrum.png') % (str(line), str(cdp)))



if __name__ == '__main__':
    root = '/home/colin/data/Spectrum/hade'
    # get the info of the rms velocity model 
    VelDict = GetVelModel(root)
    # generate synthetic data
    GenerateFlow(VelDict)
