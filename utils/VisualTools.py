"""
A few tools to visual the velocity field
"""

import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def PlotVelField(VelField, cdpList, tInd, vInd, LineName, save_path):
    plt.figure(figsize=(20, 10), dpi=300)
    tshow = [''] * len(tInd)
    tIndex = np.linspace(0, len(tInd)-1, num=20).astype(np.int32)
    for i in tIndex:
        tshow[i] = tInd[i]
    # heatmap
    h = sns.heatmap(data=VelField, cmap='jet', linewidths=0, annot=False, cbar=False,
                    vmax=vInd[-1], vmin=vInd[0], xticklabels=cdpList, yticklabels=tshow,
                    cbar_kws={'label': 'Velocity (m/s)'})

    # color bar
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=15)

    plt.title('Line %s Velocity Field' % LineName, fontsize=25)
    plt.xlabel('CDP', fontsize=20)
    plt.ylabel('t0', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close('all')

def VisualTV(OriTv, Noise, t0Vec, vVec):
    plt.figure(figsize=(20, 10), dpi=300)
    plt.scatter(OriTv[:, 0], OriTv[:, 1], c='black', label='Label')
    plt.scatter(Noise[:, 0], Noise[:, 1], c=Noise[:, 2], label='Noise')
    plt.legend()
    plt.xlim((vVec[0], vVec[-1]))
    plt.ylim((t0Vec[0], t0Vec[-1]))
    

# Plot CMP gather and NMO CMP gather
def W_Plot(traces, xVec, yVec, xlab='Trace Index', 
           ylab='Time (ms)', title='W-Plot', color='black', norm='All', SavePath=None):

    ################################################################
    # Basic Index and Interval Setting
    ################################################################    
    if xVec is None:
        xVec = np.range(traces.shape[1])
    if yVec is None:
        yVec = np.range(traces.shape[0])

    xIndex = np.linspace(np.min(xVec), np.max(xVec), traces.shape[1])
    xInt = (xIndex[1] - xIndex[0]) * 0.55
    TracesCp = copy.deepcopy(traces)

    ################################################################
    # Scale each traces
    ################################################################

    # Split the positive and negative part of the traces
    TracesCpPos = np.zeros_like(TracesCp)
    TracesCpPos[TracesCp > 0] = TracesCp[TracesCp > 0]
    TracesCpNeg = np.zeros_like(TracesCp)
    TracesCpNeg[TracesCp < 0] = TracesCp[TracesCp < 0]

    # Scale the positive and negative parts
    if norm == 'All':
        TracesCpPos /= (TracesCpPos.max() / xInt)
        TracesCpNeg /= (-TracesCpNeg.min() / xInt)
    else:
        TracesCpPos /= (np.max(TracesCpPos, axis=0) / xInt)
        TracesCpNeg /= (-np.min(TracesCpNeg, axis=0) / xInt)
    
    ################################################################
    # Plot the wiggle figure of the traces
    ################################################################

    _, ax = plt.subplots(figsize=(3, 10), dpi=90)
    for i in range(TracesCp.shape[1]):
        ax.fill_betweenx(yVec, xIndex[i], xIndex[i] + TracesCpPos[:, i], facecolor=color, interpolate=True, alpha=0.99)
        ax.plot(xIndex[i] + TracesCpNeg[:, i], yVec, c=color, linewidth=0.2)
        # ax.fill_betweenx(yVec, xIndex[i] + TracesCpNeg[:, i], xIndex[i], facecolor='white', interpolate=True)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(min(xVec)-xInt, max(xVec)+xInt)
    ax.set_ylim(min(yVec), max(yVec))
    ax.set_title(title)
    ax.invert_yaxis()
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close()



def set_axis(x_dim, y_dim, interval=50):
    x_dim, y_dim = x_dim.astype(int), y_dim.astype(int)
    show_x = range(0, len(x_dim) - 1, interval)
    show_y = range(0, len(y_dim) - 1, interval)
    ori_x = range(x_dim[0], x_dim[-1], (x_dim[1] - x_dim[0]) * interval)
    ori_y = range(y_dim[0], y_dim[-1], (y_dim[1] - y_dim[0]) * interval)
    plt.xticks(show_x, ori_x)
    plt.yticks(show_y, ori_y)


def PlotSpec(spectrum, t0Vec, vVec, title=None, SavePath=None):
    if len(t0Vec) != spectrum.shape[0]:
        t0Vec = np.linspace(t0Vec[0], t0Vec[-1], spectrum.shape[0])
    if len(vVec) != spectrum.shape[1]:
        vVec = np.linspace(vVec[0], vVec[-1], spectrum.shape[1])
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    plt.imshow(data_plot_hot, aspect='auto')
    plt.xlabel('Velocity (m/s)')
    set_axis(vVec, t0Vec)
    plt.ylabel('Time (ms)')
    if title is not None:
        plt.title(title)
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()
