import argparse
import os
from time import time
import numpy as np

from utils.GenerateData import GenerateNoiseTV, Semblance, SyntheticGather
from utils.LoadVelModel import GetVelModel
from utils.PreProcess import interpolation
from utils.VisualTools import PlotVelField


#########################################
# setting
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('--DataRoot', type=str, default='/mnt/data/spectrum/hade')
parser.add_argument('--OutRoot', type=str, default='/mnt/data/spectrum/hade-S1', help='path of Output')
parser.add_argument('--SNR', type=float, default=1, help='signal noise ratio')
args = parser.parse_args()


#########################################
# get RMS velocity model
#########################################
# init folder
for folder in ['gth', 'pwr']:
    if not os.path.exists(os.path.join(args.OutRoot, folder)):
        os.makedirs(os.path.join(args.OutRoot, folder))

# get the info of the rms velocity model 
if os.path.exists(os.path.join('model_info', 'RMS_info.npy')):
    VelDict = np.load('model_info/RMS_info.npy', allow_pickle=True)
else:
    VelDict = GetVelModel(args.DataRoot)
    if not os.path.exists('model_info'): os.makedirs('model_info')
    np.save(os.path.join('model_info', 'RMS_info.npy'), VelDict)

#########################################
# synthetic data
#########################################
# get index
IndexList = []
for line in sorted(list(VelDict['VelModel'].keys())):
    for cdp in sorted(list(VelDict['VelModel'][line].keys())):
        IndexList.append(str(line) + '-' + str(cdp)) 
for index in IndexList:
    start = time()
    # load information
    line, cdp = index.split('-')
    LabelPoint = VelDict['VelModel'][line][cdp]['RMSVel']
    t0Vec = VelDict['t0Vec']
    tVec = VelDict['tVec']
    vVec = VelDict['VelModel'][line][cdp]['vVec']
    OffsetVec = VelDict['VelModel'][line][cdp]['offVec']
    # generate t-v noise
    TvSet = GenerateNoiseTV(LabelPoint, t0Vec, vVec, vRange=400, nsr=1/args.SNR, LowNoise=2)
    # generate supergather
    GatherAll, _ = SyntheticGather(TvSet, tVec, OffsetVec, 50)
    np.save(os.path.join(args.OutRoot, 'gth', 'gth-%s-%s.npy' % (str(line), str(cdp))), GatherAll)
    # Calculate the specturm using the semblance method
    Spectrum = Semblance(GatherAll, tVec, OffsetVec, t0Vec, vVec)
    np.save(os.path.join(args.OutRoot, 'pwr', 'pwr-%s-%s.npy' % (str(line), str(cdp))), Spectrum)
    print('Line %s\tCDP %s\tcost time:\t%.3fs' % (str(line), str(cdp), time()-start))

########################################################
# visualize the generated RMS velocity field
########################################################
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
    if not os.path.exists(os.path.join('model_info', 'fig')): 
        os.makedirs(os.path.join('model_info', 'fig'))
    # plot the velocity field 
    PlotVelField(VelField, list(LineVel.keys()), t0Vec, vVec, line, 
                    os.path.join('model_info', 'fig', 'Line-%s.png' % line))
    print(line, 'done!')