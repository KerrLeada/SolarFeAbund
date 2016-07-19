# -*- coding: utf8 -*-

"""
This module contains functions for plotting observed and synthetic spectra.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.interpolate as si

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

def calc_ew_for(region, points):
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

