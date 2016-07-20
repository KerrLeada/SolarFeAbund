# -*- coding: utf8 -*-

"""
This module contains functions for plotting observed and synthetic spectra.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.interpolate as si
import astropy.units

def equivalent_width(wav, inten):
    """
    Calculates the equivalent width for a line obtained with the given wavelength and intensity.
    """
    
    # Technically, the method chosen here is not always optimal since it is sensitive towards blends
    # and such things. As such it should not be used when there are a lot of blended lines. However,
    # most lines are "nice" so this might be sufficient. There is also a slight error caused by the
    # use of the trapezoidal rule, which can be problematic if the resolution of the lines is too low.
    # In this particular case, the resolution is high enough that this should not be a problem.
    
    # The continuum level should be the maximum intensity
    cont = inten.max()

    # Calculate the area of the line
    area = np.trapz(cont - inten, x = wav)
    
    # If ew is the equivalent width, we have that: cont*ew = area
    # As such the equivalent width is given by ew = area/cont
    return area / cont

def equivalent_width_cc(region, offsets):
    """
    """
    
    wav = region.wav
    inten = region.inten
    inten_max = inten.max()
    
    ew = np.zeros(len(offsets)+1, dtype = np.float64)
    ew[0] = equivalent_width(wav, inten)
    
    if not isinstance(offsets, np.ndarray):
        offsets = np.array(offsets)
    cont = inten_max + offsets
    
    for i, c in enumerate(cont):
        area = np.trapz(c - inten, x = wav)
        ew[i+1] = area / c
    
    return np.concatenate(([0], offsets)), ew

def equivalent_width_ci(region, offsets):
    """
    """
    
    wav = region.wav
    inten0 = region.inten
    cont = inten0.max()
    
    ew = np.zeros(len(offsets)+1, dtype = np.float64)
    ew[0] = equivalent_width(wav, inten0)
    
    if not isinstance(offsets, np.ndarray):
        offsets = np.array(offsets)
    inten = [inten0 + off for off in offsets]
    
    for i, curr_inten in enumerate(inten):
        area = np.trapz(cont - curr_inten, x = wav)
        ew[i+1] = area / cont
    
    return np.concatenate(([0], offsets)), ew

#class FitResult(object):
#    def __init__(self, eq_width, obs_eq_width):
#        

def fit(region, offsets, synth_wav, synth_inten, eq_width_unit = astropy.units.pm):
    conv_factor = (1 * astropy.units.AA).to(eq_width_unit).value
    best_ew = np.zeros(len(offsets), dtype = np.float64)
    best_index = np.zeros(len(offsets), dtype = np.int64)
    ew = np.zeros(len(synth_inten), dtype = np.float64)
    obs_wav = region.wav
    obs_inten = region.inten
    inten_max = region.inten.max()
    for i, off in enumerate(offsets):
        cont = inten_max + off
        area = np.trapz(cont - obs_inten, x = obs_wav)
        obs_ew = conv_factor * area / cont
        for ii, si in enumerate(synth_inten):
            area = np.trapz(cont - si, x = synth_wav)
            ew[ii] = conv_factor * area / cont
        best = np.argmin(np.abs(ew - obs_ew))
        best_ew[i] = ew[best]
        best_index[i] = best
    return best_index, best_ew

def _fit_with(obs_ew, synth_wav, synth_inten, cont, conv_factor):
    ew = np.zeros(len(synth_inten), dtype = np.float64)
    for i, si in enumerate(synth_inten):
        area = np.trapz(cont - si, x = synth_wav)
        ew[i] = conv_factor * area / cont
    best = np.argmin(np.abs(ew - obs_ew))
    return best, ew

def equivalent_width_for(region, points, times = 100):
    """
    """
    
    wav = region.wav
    inten = region.inten
    points = [p for p in points if p < len(wav)]
    ew = np.zeros(len(points), dtype = np.float64)
    ew_std = np.zeros(len(points), dtype = np.float64)
    
    for i, pts in enumerate(points):
        selpts = np.zeros((times+1, pts), dtype = np.int64)
        selpts[0] = np.array([int(w) for w in np.arange(0, len(wav)-1, step = len(wav) / pts)], dtype = np.int64)
        selpts[1:] = np.random.randint(0, len(wav), size = (times, pts))
        selpts.sort()
        ew_curr = [equivalent_width(wav[p], inten[p]) for p in selpts]
        ew[i] = np.mean(ew_curr)
        ew_std[i] = np.std(ew_curr)
    
    return points, ew, ew_std

def equivalent_width_with(region, skips):
    """
    """
    
    wav = region.wav
    inten = region.inten
    ew = np.zeros(len(skips), dtype = np.float64)
    
    for i, s in enumerate(skips):
        ew[i] = equivalent_width(wav[::s], inten[::s])
    
    return ew
        

def interp_equivalent_width_for(region, points):
    """
    """
    
    tck = si.splrep(region.wav, region.inten)
    ew_lin = np.zeros(len(points), dtype = np.float64)
    ew_quad = np.zeros(len(points), dtype = np.float64)
    for i, pts in enumerate(points):
        wavpts = np.linspace(region.wav[0], region.wav[-1], num = pts)
        ew_lin[i] = equivalent_width(wavpts, np.interp(wavpts, region.wav, region.inten))
        ew_quad[i] = equivalent_width(wavpts, si.splev(wavpts, tck))
    return ew_lin, ew_quad

def approx_equivalent_width_for(region, points, rand_factor):
    """
    """
    
    if not hasattr(randfactor, "__call__"):
        randfactor = lambda dwav, rand_num: dwav*rand_factor*(rand_num - 1.0/2.0)
    
    point_count = len(points)
    tck = si.splrep(region.wav, region.inten)
    ew_lin = np.zeros((point_count, 2), dtype = np.float64)
    ew_quad = np.zeros((point_count, 2), dtype = np.float64)
    
    for i, pts in enumerate(points):
        wavpts = np.linspace(region.wav[0], region.wav[-1], num = pts)
        ew_lin[i] = equivalent_width(wavpts, np.interp(wavpts, region.wav, region.inten))
        ew_quad[i] = equivalent_width(wavpts, si.splev(wavpts, tck))
    return ew_lin, ew_quad
