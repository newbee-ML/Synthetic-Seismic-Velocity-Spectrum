import numpy as np
import random
import math
from scipy import interpolate


def interpolation(label_point, t_interval, v_interval=None):
    # sort the label points
    label_point = np.array(sorted(label_point, key=lambda t_v: t_v[0]))

    # ensure the input is int
    t0_vec = np.array(t_interval).astype(int)

    # get the ground truth curve using interpolation
    peaks_selected = np.array(label_point)
    func = interpolate.interp1d(peaks_selected[:, 0], peaks_selected[:, 1], kind='linear', fill_value="extrapolate")
    y = func(t0_vec)
    if v_interval is not None:
        v_vec = np.array(v_interval).astype(int) 
        y = np.clip(y, v_vec[0], v_vec[-1])

    return np.hstack((t0_vec.reshape((-1, 1)), y.reshape((-1, 1))))


def GenerateNoiseTV(TVSet, t0Vec, vVec, vRange=200, nsr=2.0, LowNoise=5):
    """
    generate t-v set of the noise
    including a few noise point
    """
    TVSet = np.array(TVSet)
    vel_curve = interpolation(TVSet, t0Vec, vVec)
    TVSet = np.array(TVSet)

    noise = []
    # uniform noise (deep layer)
    if int(TVSet.shape[0]*nsr) > 0:
        for i in range(int(TVSet.shape[0]*nsr)):
            t_random = random.randint(np.ptp(t0Vec)*0.3+np.min(t0Vec), np.max(t0Vec))
            v_center = vel_curve[np.argmin(np.abs(vel_curve[:, 0] - t_random)), 1]
            v_random = random.randint(int(max((v_center-vRange, 0))),
                                      int(min((v_center+vRange, vVec[-1]))))
            noise.append([t_random, v_random, 2])

    # low velocity noise (shallow layer)
    for i in range(LowNoise):
        t_random = random.randint(np.min(t0Vec), np.ptp(t0Vec)*0.3+np.min(t0Vec))
        v_center = vel_curve[np.argmin(np.abs(vel_curve[:, 0] - t_random)), 1]
        v_random = random.randint(int(max((v_center-vRange/2, 0))),
                                    int(min((v_center+vRange/2, vVec[-1]))))
        noise.append([t_random, v_random, 1])

    AllTvSet = np.vstack((np.hstack((TVSet, np.zeros((TVSet.shape[0], 1)))), np.array(noise)))
    return AllTvSet


# Calculate ricker wavelet value
def RickerWavelet(t, dr, g=15):
    exp = (np.pi * g * (t-dr) / 1000) ** 2
    return (1 - 2 * exp) * np.exp(-exp)


# Calculate travel time
def TravelTime(t0, x, vNMO):
    return np.sqrt(t0 ** 2 + (x / vNMO * 1000) ** 2)
        

# Generate the gather data according the t-v set
def SyntheticGather(TvSet, tVec, OffsetVec, g=40):
    GatherReal = np.zeros((len(tVec), len(OffsetVec)))
    GatherNoise = np.zeros((len(tVec), len(OffsetVec)))

    # calculate the travel curve of each t-v points
    TravelCurveReal, TravelCurveNoise = [], []
    for t0, vNMO, PType in TvSet:
        if PType == 0:
            TravelCurveReal.append(TravelTime(t0, OffsetVec, vNMO))
        else:
            TravelCurveNoise.append(TravelTime(t0, OffsetVec, vNMO))
    TravelCurveReal, TravelCurveNoise = np.array(TravelCurveReal), np.array(TravelCurveNoise)

    # calculate the amplitude of the each trace
    for i, t in enumerate(tVec):
        GatherReal[i, :] += np.sum(RickerWavelet(t, TravelCurveReal, g), axis=0)
        GatherNoise[i, :] += np.sum(RickerWavelet(t, TravelCurveNoise, g), axis=0)
    GatherNoise += np.random.normal(0, 1e-2, GatherNoise.shape)
    return GatherReal+GatherNoise, GatherNoise  


# Calculate the spectrum
def Semblance(CMPGather, tVec, OffsetVec, t0Vec, vVec):

    # NMO correction using single NMO velocity 
    def NMOCor(VNMO):
        nmo = np.zeros_like(CMPGather)
        for i, t0 in enumerate(tVec):
            TravelT = TravelTime(t0, OffsetVec, VNMO)
            # invert to the t0 index
            t0Index = ((TravelT-tMin)/dt).astype(np.int32)
            offIndex = np.where(t0Index < len(tVec))[0]
            nmo[i, offIndex] = CMPGather[t0Index[offIndex], offIndex]
        return nmo

    # base info
    spectrum = np.zeros((len(t0Vec), len(vVec)))
    dt0, tMin, dt = t0Vec[1] - t0Vec[0], tVec[0], tVec[1] - tVec[0]

    # iteration the velocity list
    for j, v in enumerate(vVec):
        NMOCMP = NMOCor(v)
        SumSquareRow = np.sum(NMOCMP, axis=1) ** 2
        SumSquareAll = np.sum(NMOCMP ** 2, axis=1)
        for i, t0 in enumerate(t0Vec):
            tIndex = range(int((t0 - tMin) / dt - 0.3 * dt0 / dt),
                            int((t0 - tMin) / dt + 0.3 * dt0 / dt) + 1)
            if min(tIndex) < 0 or max(tIndex) > len(tVec) - 1:
                continue
            denominator_square = np.sum(SumSquareAll[tIndex])
            if denominator_square == 0:
                continue
            else:
                spectrum[i, j] = np.sum(SumSquareRow[tIndex]) / (denominator_square * NMOCMP.shape[1])
    return spectrum