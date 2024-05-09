import matplotlib.pyplot as plt
import numpy as np
from constants import *

def get_center(ax):
    low, high = ax.get_xlim()
    x = 0.5 * (low + high)
    low, high = ax.get_ylim()
    y = 0.5 * (low + high)
    return x,y        

def add_birds_eye_view(ax, x, y, lookup):
    ''' 
    Adds a plot of x, y to ax with a marker that tracks the x coordinate of the mouse in other axes
    '''
    ax.plot(x, y, c='gray')
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    marker, = ax.plot([], [], marker='o', c='violet')
    
    def on_move(event):
        if event.inaxes == ax:
            return
        position = event.xdata
        idx = np.searchsorted(lookup, position)
        marker.set_data(x[idx], y[idx])
    return plt.connect('motion_notify_event', on_move)

def add_crosshairs(*axs):
    '''
    Adds crosshairs that follow the mouse to subplots in axs
    '''
    active = [False] * len(axs)
    hlines = [None] * len(axs)
    vlines = [None] * len(axs)
    for idx, ax in enumerate(axs):
        x,y = get_center(ax)
        vlines[idx] = ax.axvline(x=x,ls='--', color='gray', lw=0.8)
        hlines[idx] = ax.axhline(y=y,ls='--', color='gray', lw=0.8)
        vlines[idx].set_visible(False)
        hlines[idx].set_visible(False)
        
    def on_enter(event):
        target = event.inaxes
        for idx, ax in enumerate(axs):
            vlines[idx].set_visible(True)
            active[idx] = ax == target
            hlines[idx].set_visible(active[idx])
    
    def on_move(event):
        if not any(active):
            return
        x = event.xdata
        y = event.ydata
        for vline in vlines:
            vline.set_xdata([x,x])
        for idx, hline in enumerate(hlines):
            if active[idx]:
                hline.set_ydata([y,y])
    plt.connect('axes_enter_event', on_enter)
    return plt.connect('motion_notify_event', on_move)