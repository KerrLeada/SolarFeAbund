"""
This module contains helper functions to handle regions. A region is a tuple with 4 elements
with information about regions. The functions here are not strictly needed but can make the
code clearer.
"""

from __future__ import print_function

import numpy as np
import satlas as sa

"""
IDEAS FOR REGION:
Region(select(start_wav, end_wav, wav, inten), lambda w: w[1] - w[0])
Region(select(start_wav, end_wav, wav, inten), lambda w: w[1] - w[0])
Region(atlas(start_wav, end_wav), lambda w: w[1] - w[0])
"""

def _require_positive_int(obj, name):
    if not isinstance(obj, int):
        raise Exception(name +  " must be an int greater then 0, but it was a " + type(obj).__name__)
    elif obj < 1:
        raise Exception(name + " must be an int greater then 0, but it had the value " + str(obj))

class Region(object):
#    def __init__(self, wav_start, wav_end, dlambda, nlambda = None, scale_factor = 1.0, cgs = True):
    def __init__(self, lambda0, wav, inten, dlambda, nlambda, scale_factor = 1.0, interp_obs = False, inten_scale_factor = 1.0, nshift = None, nmul = None):
        """
        Constructor for Region objects. It is not recommeneded to use this constructor directly. The parameters are:
            lambda0 = starting wavelength
            wav     = wavelengths of observed data points
            inten   = intensity of the observed data points
            dlambda = step size for synthezising a spectra
            nlambda = amount of steps for synthezising a spectra
        Optional parameters:
            scale factor       = the scale factor for the synthesizing spectra
            interp_obs         = determines if the observed (True) or synthetic (False) data should be interpolated when
                                 fitting the synthetic spectra to the observed spectra
            inten_scale_factor = used to scale the intensity inten (the "real" intensity is: inten * inten_scale_factor)
            nshift             = a positive integer that determines how many shifts are used when trying to determine how shifted
                                 a line is (default is 101)
            nmul               = a positive integer that multiplies the nshift quantity (default is 1)
        """
        self.wav = wav
        self.inten = inten
        self.inten_scale_factor = inten_scale_factor
        self.lambda0 = lambda0
        self.length = wav.size
        self.dlambda = dlambda
        self.nlambda = nlambda
        self.scale_factor = scale_factor
        self.lambda_end = lambda0 + dlambda*nlambda
        self.interp_obs = interp_obs
        
        # Set nshift
        if nshift != None:
            _require_positive_int(nshift, "nshift")
            self.nshift = nshift
        else:
            self.nshift = 101
        
        # Set nmul
        if nmul != None:
            _require_positive_int(nmul, "nmul")
            self.nmul = nmul
        else:
            self.nmul = 1
            
        
#        if self.lambda0 + self.dlambda*self.nlambda > self.lambda_end:
#            print("?????? ", self.lambda0 + self.dlambda*self.nlambda, " ?? ", self.lambda_end)
    
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
    
    def __str__(self):
        return ("Region(lambda0 = " + str(self.lambda0) +
                ", dlambda = " + str(self.dlambda) +
                ", nlambda = " + str(self.nlambda) +
                ", scale_factor = " + str(self.scale_factor) +
                ", length = " + str(self.length) +
                ", lambda_end = " + str(self.lambda_end) + ")")

def new_region_in(atlas, lambda0, lambda_end, scale_factor = 1.0, dlambda = None, nlambda = None, cgs = True, interp_obs = False, nshift = None, nmul = None):
    """
    Creates a new region in the form of a Region object between lambda0 and lambda_end.
    """
    
    # Get the wavelengths and their corresponding intensities and continuum intensities
    wav, inten, cont = atlas.getatlas(lambda0, lambda_end, cgs = cgs)
    lambda0 = wav[0]
    inten_max = inten.max()
    inten /= inten_max
    
    # Get and validate dlambda
    if dlambda == None:
        dlambda = wav[1] - wav[0]
    elif hasattr(dlambda, "__call__"):
        dlambda = dlambda(wav)
    elif not np.isscalar(dlambda):
        raise Exception("dlambda must be a scalar or a callable")
    
    # Get and validate nlambda
    if nlambda == None:
        nlambda = wav.size
    elif hasattr(nlambda, "__call__"):
        nlambda = nlambda(wav)
    else:
        raise Exception("nlambda must be None or a callable")
    
    # Get and validate the scale factor
    if hasattr(scale_factor, "__call__"):
        scale_factor = scale_factor(wav, inten, cont)
    elif not np.isscalar(scale_factor):
        raise Exception("scale_factor must be a scalar or a callable")
    
    # Return the new region    
    return Region(lambda0, wav, inten, dlambda, nlambda, scale_factor = scale_factor, interp_obs = interp_obs, inten_scale_factor = inten_max, nshift = nmul, nmul = nmul)

def new_region(obs_wav, obs_inten, lambda0, dlambda, nlambda, scale_factor = 1.0, interp_obs = False, nshift = None, nmul = None):
    """
    Creates a region in Region form (use "region" to create in tuple form instead). The parameters are:
        obs_wav      = observed wavelengths
        obs_inten    = obserbed intensities
        lambda0      = starting wavelength
        dlambda      = the length of each step
        nlambda      = the number of steps
    Optional parameters:
        scale_factor = the scale factor (used for normalization)
        interp_obs   = determines if the observed (True) or synthetic (False) data should be interpolated when
                       fitting the synthetic spectra to the observed spectra
        nshift             = a positive integer that determines how many shifts are used when trying to determine how shifted
                             a line is (default is 101)
        nmul               = a positive integer that multiplies the nshift quantity (default is 1)
    The end point can be found with: lambda_0 + dlambda*nlambda
    The return value is a tuple of the form:
        (lambda_0, dlambda, nlambda, scale_factor)
    """
    obs_wav, obs_inten = get_within(lambda0, lambda0 + dlambda*nlambda, obs_wav, obs_inten)
    if len(obs_wav) != nlambda:
        raise Exception("The observed data had a length " + str(len(obs_wav)) + " while nlambda was " + str(nlambda) + ". They must match.")
    return Region(lambda0, obs_wav, obs_inten, dlambda, nlambda, scale_factor = scale_factor, interp_obs = interp_obs, nshift = nmul, nmul = nmul)

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

#def region_in(lambda_0, lambda_end, wl, scale_factor, dlambda_fn = None):
#    wav = wl[(lambda_0 <= wl) & (wl <= lambda_end)]
#    nlambda = len(wav)
#    dlambda = wav[1] - wav[0] if not dlambda_fn else dlambda_fn(wav, nlambda)
#    return region(lambda_0, dlambda, nlambda, scale_factor)

#def region_start(region):
#    """
#    Returns the starting point of the region. Since the starting point of the region
#    is the first element, this is equivalent to: region[0]
#    """
#    if isinstance(region, Region):
#        return region.lambda0
#    return region[0]

#def region_end(region):
#    """
#    Calculates the end point of the given region. Specifically, the given region
#    is a tuple where the elements are
#        (lambda0, dlambda, nlambda, scale_factor)
#    The end point wavelength is calculated as: lambda_0 + dlambda*nlambda
#    """
#    if not isinstance(region, Region):
#        lambda0, dlambda, nlambda = region[0], region[1], region[2]
#        return lambda0 + dlambda*nlambda
#    return region.lambda_end

#def region_length(region):
#    """
#    Calculates the length of the given region. If the region is
#        (lambda0, dlambda, nlambda, scale_factor)
#    then the region length is: dlambda*nlambda
#    """
#    if not isinstance(region, Region):
#        dlambda, nlambda = region[1], region[2]
#        return dlambda*nlambda
#    return region.length

#def get_region(reg, wav, inten, left_padding = 0.0, right_padding = 0.0):
#    """
#    Returns the wav and inten values contained in the given region reg. Specifically
#    two arrays are returned. The first is the wavelengths wav contained in the region
#    and the second is the corresponding intensities inten.
#    
#    The padding paramters, left_padding and right_padding, allows pushing the bounderies of
#    the region on the left (lower then minimum) side and right (higher then maximum) respectively.
#    Specifically, for a given value of padding, the start of the region is given by
#        region_start(region) - padding_left
#    while the end is given by
#        ragion_end(region) + padding_right
#    The default value of both the padding parameters is 0.
#    """
#    rs = region_start(reg) - left_padding
#    re = region_end(reg) + right_padding
#    interval = (rs <= wav) & (wav <= re)
#    return wav[interval], inten[interval]

def get_within(lambda0, lambda_end, wav, inten, left_padding = 0.0, right_padding = 0.0):
    interval = (lambda0 - left_padding <= wav) & (wav <= lambda_end + right_padding)
    return wav[interval], inten[interval]

#def get_region_N(reg, wav, intensities):
#    rs = region_start(reg)
#    re = region_end(reg)
#    interval = (rs <= wav) & (wav <= re)
#    intens = [inten[interval] for inten in intensities]
#    return wav[interval], intens

#def get_interval(reg, wav, padding = 0.0):
#    rs = region_start(reg) - padding
#    re = region_end(reg) + padding
#    return (rs <= wav) & (wav <= re)

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
