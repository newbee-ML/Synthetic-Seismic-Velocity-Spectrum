import os
from multiprocessing import Pool
from time import time, sleep

import numpy as np

from utils.GenerateData import GenerateNoiseTV, Semblance, SyntheticGather
from utils.LoadVelModel import GetVelModel


def GetIndex():
    IndexList = []
    for line in sorted(list(VelDict['VelModel'].keys())):
        for cdp in sorted(list(VelDict['VelModel'][line].keys())):
           IndexList.append(str(line) + '-' + str(cdp)) 
    return IndexList
    

def Synthetic(index):
    line, cdp = index.split('-')
    LabelPoint = VelDict['VelModel'][line][cdp]['RMSVel']
    t0Vec = VelDict['t0Vec']
    tVec = VelDict['tVec']
    vVec = VelDict['VelModel'][line][cdp]['vVec']
    OffsetVec = VelDict['VelModel'][line][cdp]['offVec']

    start = time()

    ################################################################
    # Synthetic gather
    ################################################################
    # generate t-v noise
    TvSet = GenerateNoiseTV(LabelPoint, t0Vec, vVec, vRange=400, nsr=0.7, LowNoise=6)
    # generate supergather
    GatherAll, _ = SyntheticGather(TvSet, tVec, OffsetVec, 50)
    np.save(os.path.join(SaveRoot, 'gth', 'gth-%s-%s.npy' % (str(line), str(cdp))), GatherAll)

    ################################################################
    # Calculate the specturm using the semblance method
    ################################################################
    Spectrum = Semblance(GatherAll, tVec, OffsetVec, t0Vec, vVec)
    np.save(os.path.join(SaveRoot, 'pwr', 'pwr-%s-%s.npy' % (str(line), str(cdp))), Spectrum)

    print('Line %s\tCDP %s\tcost time:\t%.3fs' % (str(line), str(cdp), time()-start))


if __name__ == '__main__':
    # base root path
    root = '/home/colin/data/Spectrum/1011'
    SaveRoot = '/home/colin/data/Spectrum/syn-1011'

    if not os.path.exists(SaveRoot):
        os.makedirs(SaveRoot)
        for folder in ['gth', 'pwr']:
            os.makedirs(os.path.join(SaveRoot, folder))

    # get the info of the rms velocity model 
    VelDict = GetVelModel(root)
    Index = GetIndex()

    # save velocity model information
    np.save(os.path.join(SaveRoot, 'ModelInfo.npy'), VelDict)  # np.load('dict.npy', allow_pickle=True)

    # # test single process function
    # Synthetic(Index[0])

    # generate the synthetic data
    p = Pool(8)
    for index in Index:
        p.apply_async(Synthetic, args=(index,))
        sleep(1)
    p.close()
    p.join()


