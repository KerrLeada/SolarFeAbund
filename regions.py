"""
This module contains helper functions to handle regions. A region is a tuple with 4 elements
with information about regions. The functions here are not strictly needed but can make the
code clearer.
"""

from __future__ import print_function

import numpy as np
import satlas as sa

def _require_positive_int(obj, name):
    """
    Checks so obj is a positive integer, and throws an exception if it's not. The name of the object can
    be set with name.
    """
    if not isinstance(obj, int):
        raise Exception(name +  " must be an int greater then 0, but it was a " + type(obj).__name__)
    elif obj < 1:
        raise Exception(name + " must be an int greater then 0, but it had the value " + str(obj))

class Region(object):
#    def __init__(self, wav_start, wav_end, dlambda, nlambda = None, scale_factor = 1.0, cgs = True):
    def __init__(self, lambda0, wav, inten, dlambda, nlambda, scale_factor = 1.0):
        """
        Constructor for Region objects. It is not recommeneded to use this constructor directly. The parameters are:
            lambda0 = starting wavelength
            wav     = wavelengths of observed data points
            inten   = intensity of the observed data points
            dlambda = step size for synthezising a spectra
            nlambda = amount of steps for synthezising a spectra
        Optional parameters:
            scale factor       = the scale factor for the synthesizing spectra
        """
        self.wav = wav
        self.inten_scale_factor = inten.max()
        self.inten = inten / self.inten_scale_factor
        self.lambda0 = lambda0
        self.length = wav.size
        
        # Get dlambda
        if hasattr(dlambda, "__call__"):
            dlambda = dlambda(wav)
        elif not np.isscalar(dlambda):
            raise Exception("dlambda must be a scalar or a callable")
        
        # Get nlambda
        if hasattr(nlambda, "__call__"):
            nlambda = nlambda(wav)
        elif not np.isscalar(nlambda):
            raise Exception("nlambda must be a scalar or a callable")
        
        # Get and validate the scale factor
        if hasattr(scale_factor, "__call__"):
            scale_factor = scale_factor(wav, self.inten)
        elif not np.isscalar(scale_factor):
            raise Exception("scale_factor must be a scalar or a callable")

        self.dlambda = dlambda
        self.nlambda = nlambda
        self.scale_factor = scale_factor
        self.lambda_end = lambda0 + dlambda*nlambda
    
    def to_tuple(self):
        """
        Returns the tuple representation of the resion
        """
        return region(self.lambda0, self.dlambda, self.nlambda, self.scale_factor)
    
    def get_contained(self, wav, inten, left_padding = 0.0, right_padding = 0.0):
        """
        Returns the parts of wav and inten contained within this region.
        """
        return get_within(self.lambda0, self.lambda_end, wav, inten, left_padding = left_padding, right_padding = right_padding)

    def copy(self):
        reg = Region(self.lambda0, self.wav, self.inten, self.dlambda, self.nlambda, scale_factor = self.scale_factor)
        reg.inten_scale_factor = self.inten_scale_factor
        return reg
   
    def refine(self, left, right, dlambda = None, nlambda = None):
        reg = self.copy()
        lambda0 = reg.lambda0 + left
        lambda_end = reg.lambda_end - right
        wav, inten = get_within(lambda0, lambda_end, reg.wav, reg.inten)
        inten_max = inten.max()
        
        if nlambda == None:
            nlambda = wav.size
        elif hasattr(dlambda, "__call__"):
            nlambda = nlambda(reg, wav)
        else:
            raise Exception("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
        
        if dlambda == None:
            dlambda = reg.dlambda
        elif hasattr(dlambda, "__call__"):
            dlambda = dlambda(reg, wav)
        
        if lambda0 + dlambda*nlambda > lambda_end + dlambda/100:
            raise Exception("jiaduiduWUHDUQIWNiawsu")
        
        reg.lambda0 = lambda0
        reg.lambda_end = lambda_end
        reg.inten_scale_factor = inten_max * reg.inten_scale_factor
        reg.wav = wav
        reg.inten = inten
        reg.nlambda = nlambda
        reg.dlambda = dlambda
        reg.length = wav.size
        
        return reg
    
    def __str__(self):
        return ("Region(lambda0 = " + str(self.lambda0) +
                ", lambda_mid = " + str((self.lambda0 + self.lambda_end) / 2) +
                ", lambda_end = " + str(self.lambda_end) +
                ", dlambda = " + str(self.dlambda) +
                ", nlambda = " + str(self.nlambda) +
                ", scale_factor = " + str(self.scale_factor) +
                ", length = " + str(self.length) + ")")

    def __eq__(self, other):
        return (self.inten_scale_factor == other.inten_scale_factor and
                self.lambda0 == other.lambda0 and
                self.length == other.length and
                self.dlambda == other.dlambda and
                self.nlambda == other.nlambda and
                self.scale_factor == other.scale_factor and
                self.lambda_end == other.lambda_end and
                np.array_equal(self.wav, other.wav) and
                np.array_equal(self.inten, other.inten))

    def __ne__(self, other):
        return not (self == other)

def calc_dwav(region):
    """
    Calculates the wavelength changes between the wavelengths in the observed data of a region.
    """
    return region.wav[1:] - region.wav[:-1]

def new_region_in(atlas, lambda0, lambda_end, scale_factor = 1.0, dlambda = None, nlambda = None, cgs = True):
    """
    Creates a new region in the form of a Region object between lambda0 and lambda_end.
    """
    
    # Get the wavelengths and their corresponding intensities and continuum intensities
    wav, inten, cont = atlas.getatlas(lambda0, lambda_end, cgs = cgs)
    lambda0 = wav[0]
    
    # Handle the default value for dlambda
    if dlambda == None:
        dlambda = wav[1] - wav[0]
    
    # Handle the default value for nlambda
    if nlambda == None:
        nlambda = wav.size
    
    # Return the new region
    return Region(lambda0, wav, inten, dlambda, nlambda, scale_factor = scale_factor)

def new_region(atlas, lambda0, dlambda, nlambda, scale_factor = 1.0, cgs = True):
    """
    Creates a region in Region form (use "region" to create in tuple form instead). The parameters are:
        atlas   = the atlas object
        lambda0 = starting wavelength
        dlambda = the length of each step
        nlambda = the number of steps
    Optional parameters:
        scale_factor = the scale factor (used for normalization)
    The end point can be found with: lambda_0 + dlambda*nlambda
    The return value is a tuple of the form:
        (lambda_0, dlambda, nlambda, scale_factor)
    """
    obs_wav, obs_inten, _ = atlas.getatlas(lambda0, lambda0 + dlambda*nlambda, cgs = cgs)
    if len(obs_wav) != nlambda:
        raise Exception("The observed data had a length " + str(len(obs_wav)) + " while nlambda was " + str(nlambda) + ". They must match.")
    return Region(lambda0, obs_wav, obs_inten, dlambda, nlambda, scale_factor = scale_factor)

def new_region_from(atlas, obj):
    if not isinstance(obj, Region):
        obj = new_region(atlas, *obj)
    return obj

def get_nlambda(region):
    """
    Returns the amount of steps in a region. The region can be a Region object or a tuple of the form:
        (lambda0, dlambda, nlambda, scale_factor)
    """
    if isinstance(region, Region):
        return region.nlambda
    return region[2]

def get_dlambda(region):
    """
    Returns the difference between each step. The region can be a Region object or a tuple of the form:
        (lambda0, dlambda, nlambda, scale_factor)
    """
    if isinstance(region, Region):
        return region.dlambda
    return region[1]

def region(lambda0, dlambda, nlambda, scale_factor = 1.0):
    """
    Creates a region in tuple form (use new_region and new_regon_in to create Region objects instead).
    The parameters are:
        lambda0      = starting wavelength
        dlambda      = the length of each step
        nlambda      = the number of steps
        scale_factor = the scale factor (used for normalization)
    The end point can be found with: lambda_0 + dlambda*nlambda
    The return value is a tuple of the form:
        (lambda_0, dlambda, nlambda, scale_factor)
    """
    return (lambda0, dlambda, nlambda, scale_factor)

def get_within(lambda0, lambda_end, wav, inten, left_padding = 0.0, right_padding = 0.0):
    """
    Returns the elements in wav what are between lambda0 and lambda_end. The corresponding inten elements
    are returned as well.
    """
    interval = (lambda0 - left_padding <= wav) & (wav <= lambda_end + right_padding)
    return wav[interval], inten[interval]

def shrink(left, right, wav, inten):
    return get_within(wav[0] + left, wav[-1] - right, wav, inten)

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
