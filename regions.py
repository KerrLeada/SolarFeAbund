"""
This module contains helper functions to handle regions. A region is a tuple with 4 elements
with information about regions. The functions here are not strictly needed but can make the
code clearer.
"""

from __future__ import print_function

import numpy as np

def get_nlambda(region):
    """
    Returns the amount of steps in a region.
    """
    return region[2]

def get_dlambda(region):
    """
    Returns the difference between each step.
    """
    return region[1]

def region(lambda_0, dlambda, nlambda, scale_factor, enhancement = 1.0):
    """
    Creates a region. The parameters are:
        lambda_0     = starting wavelength
        dlambda      = the length of each step
        nlambda      = the number of steps
        scale_factor = the scale factor (used for normalization)
        enhancement  = can be used to increase the resolution of the region
    The end point can be found with: lambda_0 + dlambda*nlambda
    The return value is a tuple of the form:
        (lambda_0, dlambda/E, E*nlambda, scale_factor)
    where E = enhancement.
    """
    return (lambda_0, dlambda/enhancement, enhancement*nlambda, scale_factor)

def region_in(lambda_0, lambda_end, wl, scale_factor, dlambda_fn = None):
    wav = wl[(lambda_0 <= wl) & (wl <= lambda_end)]
    nlambda = len(wav)
    dlambda = wav[1] - wav[0] if not dlambda_fn else dlambda_fn(wav, nlambda)
    return region(lambda_0, dlambda, nlambda, scale_factor)

def region_start(region):
    """
    Returns the starting point of the region. Since the starting point of the region
    is the first element, this is equivalent to: region[0]
    """
    return region[0]

def region_end(region):
    """
    Calculates the end point of the given region. Specifically, the given region
    is a tuple where the elements are
        (lambda_0, dlambda, nlambda, scale_factor)
    The end point wavelength is calculated as: lambda_0 + dlambda*nlambda
    """
    lambda_0, dlambda, nlambda = region[0], region[1], region[2]
    return lambda_0 + dlambda*nlambda

def region_length(region):
    """
    Calculates the length of the given region. If the region is
        (lambda_0, dlambda, nlambda, scale_factor)
    then the region length is: dlambda*nlambda
    """
    dlambda, nlambda = region[1], region[2]
    return dlambda*nlambda

def get_region(reg, wav, inten, left_padding = 0.0, right_padding = 0.0):
    """
    Returns the wav and inten values contained in the given region reg. Specifically
    two arrays are returned. The first is the wavelengths wav contained in the region
    and the second is the corresponding intensities inten.
    
    The padding paramters, left_padding and right_padding, allows pushing the bounderies of
    the region on the left (lower then minimum) side and right (higher then maximum) respectively.
    Specifically, for a given value of padding, the start of the region is given by
        region_start(region) - padding_left
    while the end is given by
        ragion_end(region) + padding_right
    The default value of both the padding parameters is 0.
    """
    rs = region_start(reg) - left_padding
    re = region_end(reg) + right_padding
    interval = (rs <= wav) & (wav <= re)
    return wav[interval], inten[interval]

def get_region_N(reg, wav, intensities):
    rs = region_start(reg)
    re = region_end(reg)
    interval = (rs <= wav) & (wav <= re)
    intens = [inten[interval] for inten in intensities]
    return wav[interval], intens

def get_interval(reg, wav, padding = 0.0):
    rs = region_start(reg) - padding
    re = region_end(reg) + padding
    return (rs <= wav) & (wav <= re)

def _best_dlambda(wav0, length, nlambda, obs_wav):
    dlambda = []
    chisq = []
    dl = float(length)/float(nlambda)
    dl_limit = float(length)/float(nlambda + 1)
    step = (dl - dl_limit) / 10.0
    while dl > dl_limit:
        dlambda.append(dl)
        wav = wav0 + dl*np.arange(0.0, nlambda, step = 1.0)
        chisq.append(((obs_wav - wav)**2).sum())
        dl -= step
    chisq = np.array(chisq)
    return dlambda[np.argmin(chisq)]

def create_regions_in1(wav, length, obs_wav, scale_factor, fit_best_spacing = False):
    if isinstance(scale_factor, (int, float, np.float64, np.float32, np.int32, np.int64)):
        scale_factor = [float(scale_factor)]*len(wav)
    elif len(scale_factor) != len(wav):
        raise Exception("Dimension mismatch: scale_factor and wav must have the same length. Scale_factor had the length " + str(len(scale_factor)) + " while wav had the length " + str(len(wav)) + ".")
    regs = []
    for w, s in zip(wav, scale_factor):
        interval = (w <= obs_wav) & (obs_wav <= w + length)
        robs_wav = obs_wav[interval]
        nlambda = len(robs_wav)
        if fit_best_spacing:
            dlambda = _best_dlambda(w, length, nlambda, obs_wav[interval])
        else:
            dlambda = float(length) / float(nlambda)
#        dlambda = _best_dlambda(w, length, nlambda, obs_wav[interval])
        regs.append(region(w, dlambda, nlambda, s))
    return regs

def create_regions_in2(wav, length, obs_wav, scale_factor, fit_best_spacing = False):
    if isinstance(scale_factor, (int, float, np.float64, np.float32, np.int32, np.int64)):
        scale_factor = [float(scale_factor)]*len(wav)
    elif len(scale_factor) != len(wav):
        raise Exception("Dimension mismatch: scale_factor and wav must have the same length. Scale_factor had the length " + str(len(scale_factor)) + " while wav had the length " + str(len(wav)) + ".")
    regs = []
    for w, s in zip(wav, scale_factor):
        interval = (w <= obs_wav) & (obs_wav <= w + length)
        robs_wav = obs_wav[interval]
        nlambda = len(robs_wav)
        w0 = robs_wav[0]
        length1 = length - (w0 - w)
        if fit_best_spacing:
            dlambda = _best_dlambda(w, length1, nlambda, obs_wav[interval])
        else:
            dlambda = float(length1) / float(nlambda)
#        dlambda = float(length1) / float(nlambda)
#        dlambda = _best_dlambda(w, length1, nlambda, obs_wav[interval])
        regs.append(region(w0, dlambda, nlambda, s))
    return regs

def create_regions_in3(wav, length, obs_wav, scale_factor, fit_best_spacing = False):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!******************!!!!!!!!!!!!!!!!!!!!!!!!******************************!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if isinstance(scale_factor, (int, float, np.float64, np.float32, np.int32, np.int64)):
        scale_factor = [float(scale_factor)]*len(wav)
    elif len(scale_factor) != len(wav):
        raise Exception("Dimension mismatch: scale_factor and wav must have the same length. Scale_factor had the length " + str(len(scale_factor)) + " while wav had the length " + str(len(wav)) + ".")
    regs = []
    for w, s in zip(wav, scale_factor):
        interval = (w <= obs_wav) & (obs_wav <= w + length)
        robs_wav = obs_wav[interval]
        nlambda = len(robs_wav)
        w0 = robs_wav[0]
        we = robs_wav[-1]
        length1 = we - w0
        if fit_best_spacing:
            dlambda = _best_dlambda(w, length1, nlambda, robs_wav)
        else:
            dlambda = float(length1) / float(nlambda)
#        dlambda = float(length1) / float(nlambda)
#        dlambda = _best_dlambda(w, length1, nlambda, obs_wav[interval])
        regs.append(region(w0, dlambda, nlambda, s))
    return regs

def create_regions(wav, dlambda, nlambda, scale_factor, enhancement = 1.0):
    """
    Creats a list of regions from the given list of wavelengths. Specifically, given a list of
    wavelengths wav of length N, a list of N regions will be returned where the i:th region
    is given by:
        (wav[i], dlambda/E, E*nlambda, scale_factor)
    where E = 2**enhancement.

    If scale_factor is an int or a float, it is assumed all regions has the given scale factor.
    Otherwise it is assumed that
    """
    if isinstance(scale_factor, (int, float, np.float64, np.float32, np.int32, np.int64)):
        scale_factor = [float(scale_factor)]*len(wav)
    elif len(scale_factor) != len(wav):
        raise Exception("Dimension mismatch: scale_factor and wav must have the same length. Scale_factor had the length " + str(len(scale_factor)) + " while wav had the length " + str(len(wav)) + ".")
    return [region(wav[i], dlambda, nlambda, scale_factor[i], enhancement = enhancement) for i in range(len(wav))]
