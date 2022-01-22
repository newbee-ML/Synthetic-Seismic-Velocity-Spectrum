"""
Get the velocity model of field data

Author: Hongtao Wang | stolzpi@163.com

---
contain: 
1. gather data: 1) t index 2) offset index 
2. velocity: 1) t0 index 2) velocity index 3) rms velocity model
"""

import numpy as np
import segyio
import os
import h5py
from tqdm import tqdm


# load all info of segy file
def LoadFile(RootPath):
    """
    Load Segy, h5, label file from the root directory
    return:
        SegyDict, H5Dict, LabelDict
        ---------------------------
        SegyDict: include pwr, stk, gth three segy information
        H5Dict: include pwr, stk, gth three index information
        LabelDict: include all labels of Spectra 
    """
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(RootPath, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(RootPath, 'h5File', path), 'r'))

    # load label.npy
    LabelDict = np.load(os.path.join(RootPath, 't_v_labels.npy'), allow_pickle=True).item()

    return SegyDict, H5Dict, LabelDict


# get the continue index
def GetIndex(OriVec):
    diff = OriVec[1:] - OriVec[:-1]
    dList = list(set(diff))
    dNum = [len(np.where(diff == d)[0]) for d in dList]
    dSelected = dList[np.argmax(dNum)]
    VecMax = OriVec[0] + (OriVec[-1] - OriVec[0]) // dSelected * dSelected
    GenerateNum = int((VecMax - OriVec[0]) / dSelected) + 1
    NewVec = np.linspace(OriVec[0], VecMax, num=GenerateNum)
    return NewVec 


# select the info we need
def GetVelModel(root):
    SegyDict, H5Dict, LabelDict = LoadFile(root)

    # get index
    LabIndex = []
    for line in list(LabelDict.keys()):
        LabIndex += ['%s_%s' % (str(line), str(cdp)) for cdp in list(LabelDict[line].keys())]
    GthIndex = list(H5Dict['gth'].keys())
    PwrIndex = list(H5Dict['pwr'].keys())
    UseIndex = sorted(list(set(LabIndex) & set(GthIndex) & set(PwrIndex)))

    # get the velocity model info
    VelInfoDict = {'tVec': np.array(SegyDict['gth'].samples),
                   't0Vec': np.array(SegyDict['pwr'].samples),
                   'VelModel': {}}

    for index in tqdm(UseIndex):
        line, cdp = index.split('_')
        VelInfoDict['VelModel'].setdefault(line, {})
        OffsetIndex = np.array(H5Dict['gth'][index]['GatherIndex'])
        PwrIndex = np.array(H5Dict['pwr'][index]['SpecIndex'])
        vVec = np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]])
        offVec = np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[OffsetIndex[0]: OffsetIndex[1]])
        VelMod = {
            'vVec': GetIndex(vVec),
            'offVec': GetIndex(offVec),
            'RMSVel': LabelDict[int(line)][int(cdp)]
        }
        VelInfoDict['VelModel'][line][cdp] = VelMod

    return VelInfoDict


    