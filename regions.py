"""
This module handles regions. A region can have two forms. One is an instance of the Region
class. This is also the recommended approach. The other is as a tuple with 4 elements
with information about regions.
"""

from __future__ import print_function
from __future__ import division

import scipy.interpolate as si
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
    """
    The Region class represents regions. Instances has the attributes
    
        wav                : The observed wavelengths, stored in a numpy array.
        
        inten              : The intensities corresponding to the observed wavelengths, scaled
                             to 1. These intensities are stored in a numpy array.
        
        inten_scale_factor : The scale factor for the observed intensities. To get the unscaled
                             intensities, simply multiply the inten attribute with this one.
        
        cont               : The continuum level intensities at the wavelengths of wav.
        
        noise              : The noise of the intensity data. This may be a number or an array.
        
        lambda0            : The starting wavelength of the region.
        
        lambda_end         : The ending wavelength of the region.
        
        dlambda            : The length of the steps in the region. This is essentially the step length
                             that are used when the spectral lines are synthezised.
        
        nlambda            : The number of steps in the regions. This is essentially how many steps are
                             used whne spectral lines are synthezised.
        
        lab_wav            : The wavelength corresponding to the laboratory central wavelength of the line.
        
        scale_factor       : The scale factor that is used when lines are synthezised.
        
        length             : The amount of observed wavelengths.
    """

    def __init__(self, lambda0, wav, inten, cont, dlambda, nlambda, lab_wav, scale_factor = 1.0, noise = 1.0):
        """
        Constructor for Region objects. It is not recommeneded to use this constructor directly. The required arguments are

            lambda0 : Starting wavelength.

            wav     : Wavelengths of observed data points.

            inten   : Intensity of the observed data points.
            
            cont    : The continuum level intensities at the wavelengths of wav.

            dlambda : Step size for synthezising a spectra.

            nlambda : Amount of steps for synthezising a spectra.
            
            lab_wav : The wavelength corresponding to the laboratory central wavelength of the line.

        The optional argument is
        
            scale factor : The scale factor for the synthesizing spectra.
                           Default is 1.
            
            noise        : The noise of the intensity data. This may be a number or an array. If the latter it must be of the
                           same length as wav, inten and cont.
                           Default is 1.
        """
        
        # Fail fast
        if len(wav) != len(inten) and len(wav) != len(cont):
            raise Exception("The arguments wav, inten and cont must be arrays of equal length")
        
        self.wav = wav
        self.inten_scale_factor = inten.max()
        self.inten = inten / self.inten_scale_factor
        self.cont = cont
        self.lambda0 = lambda0
        self.length = wav.size
        self.lab_wav = lab_wav
        
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
        
        # Make sure noise is correct
        if np.isscalar(noise):
            self.noise = noise
        elif len(noise) == len(wav):
            self.noise = np.array(noise)
        else:
            raise Exception("The argument 'noise' was an instance of " + type(noise).__name__ + " with length " + str(len(noise)) + ", but was expected to have length " + str(len(noise)) +
                            " (the same was the arguments 'wav', 'inten' and 'cont'). Make sure it has the correct length, or replace it with a number.")

        self.dlambda = dlambda
        self.nlambda = nlambda
        self.scale_factor = scale_factor
        self.lambda_end = lambda0 + dlambda*nlambda
    
    def to_tuple(self):
        """
        Returns the tuple representation of the region. Specifically a tuple
        
            (lambda0, dlambda, nlambda, scale_factor)
        
        is returned. Here lambda0 is the initial wavelength of the region, dlambda is the length of each step
        in the region, nlambda is the amount of steps and scale_factor is the scale factor.
        """
        
        return region(self.lambda0, self.dlambda, self.nlambda, self.scale_factor)
    
    def get_contained(self, wav, inten, left_padding = 0.0, right_padding = 0.0):
        """
        Returns the parts of wav and inten contained within this region. The required arguments are
        
            wav   : The wavelengths.
            
            inten : The intensities corresponding to the given wavelengths.
            
        The optional arguments are
        
            left_padding  : Padding towards the left. This allows the caller to push the bounderies of the region.
            
            right_padding : Padding towards the right. This allows the caller to push the bounderies of the region.
        
        Returns the wavelengths and intensities within the region. Specifically, it returns the wavelengths w (and the
        corresponding intensities) for which:
        
            lambda0 - left_padding <= w <= lambda_end + right_padding
        
        where lambda0 is the starting wavelength of the region and lambda_end is the ending wavelength.
        """
        
        return get_within(self.lambda0, self.lambda_end, wav, inten, left_padding = left_padding, right_padding = right_padding)

    def estimate_minimum(self, num = 1000):
        """
        Estimates the minimum intensity using quadratic interpolation. The optional argument is
        
            num : The number of points to use when estimating the minimum.
                  Default is 1000.
        """
        
        tck = si.splrep(self.wav, self.inten)
        wav = np.linspace(self.wav[0], self.wav[-1], num = 1000)
        inten = si.splev(wav, tck)
        return wav[inten == min(inten)][0]

    def copy(self):
        """
        Creates a copy of this region.
        """
        
        reg = Region(self.lambda0, self.wav, self.inten, self.cont, self.dlambda, self.nlambda, self.lab_wav, scale_factor = self.scale_factor)
        reg.inten_scale_factor = self.inten_scale_factor
        return reg
   
    def refine(self, left, right, dlambda = None, nlambda = None):
        """
        Refines this region by returning a new Region object with the same information as this one, but with the initial wavelength
        shifted "left" amount to the right and the ending wavelength shifted "right" amount to the left. The required arguments are
        
            left  : The amount to shift the initial wavelength towards higher wavelengths.
            
            right : The amount to shift the ending wavelength towards lower wavelengths.
            
        The optional arguments are
        
            dlambda : Allows the caller to set the length of the steps of the region. This is either None or a function.
                      Default is None.
            
            nlambda : Allows the caller to set the amount of steps of the region. This is either None or a function.
                      Default is None.
        
        If dlambda or nlambda are functions, they are expected to take two required arguments. The first one is the original region
        and the second one is the wavelengths of the new, refined region.
        
        If lambda0 is the initial wavelength and lambda_end the ending wavelength of the original region, the refined region has the
        initial wavelength lambda0 + left and the ending wavelength lambda_end - right.
        """
        
        reg = self.copy()
        lambda0 = reg.lambda0 + left
        lambda_end = reg.lambda_end - right
        wav, inten = get_within(lambda0, lambda_end, reg.wav, reg.inten)
        _, cont = get_within(lambda0, lambda_end, reg.wav, reg.cont)
        _, noise = get_within(lambda0, lambda_end, reg.wav, reg.noise)
        inten_max = inten.max()
        
        if nlambda == None:
            nlambda = wav.size
        elif hasattr(dlambda, "__call__"):
            nlambda = nlambda(reg, wav)
        else:
            raise Exception("nlambda must be None or a function, but was an instance of " + type(nlambda))
        
        if dlambda == None:
            dlambda = reg.dlambda
        elif hasattr(dlambda, "__call__"):
            dlambda = dlambda(reg, wav)
        else:
            raise Exception("dlambda must be None or a function, but was an instance of " + type(dlambda))
        
        if lambda0 + dlambda*nlambda > lambda_end + dlambda/100.0:
            raise Exception("The given dlambda and nlambda gives an ending wavelength larger then expected")
        
        reg.lambda0 = lambda0
        reg.lambda_end = lambda_end
        reg.inten_scale_factor = inten_max * reg.inten_scale_factor
        reg.wav = wav
        reg.inten = inten
        reg.cont = cont
        reg.noise = noise
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

def new_region_in(atlas, lambda0, lambda_end, lab_wav, scale_factor = 1.0, dlambda = None, nlambda = None, noise = 1, cgs = True):
    """
    Creates a new region in the form of a Region object between lambda0 and lambda_end, using the given atlas object.
    The arguments are
    
        atlas      : The atlas object, which contains observed data.
        
        lambda0    : The starting wavelength of the region.
        
        lambda_end : The ending wavelength of the region.
        
        lab_wav    : The wavelength corresponding to the laboratory central wavelength of the line.
        
    The optional arguments are
    
        scale_factor : The scale factor of the region. This can be a number or a function.
                       Default is 1.
        
        dlambda      : The step length of the region. This can be None, a number or a function.
                       Default is None.
        
        nlambda      : The amount of steps of the region. This can be None, a number or a function.
                       Defalt is None.
        
        noise        : The noise of the intensity data. This may be a number or an array. If the latter it must be of the
                       same length as wav, inten and cont.
                       Default is 1.
        
        cgs          : Determines is cgs units should be used when synthezising spectral lines.
                       Default is True.

    When scale_factor is a function, it must take two required arguments. The first one of these is the
    observed wavelengths in the region and the second is the corresponding intensities. It should then
    return a number, which will be used as the scale factor.
    
    When dlambda is None, the difference between the first two wavelengths will be used. Otherwise a number
    or function is expected. If dlambda is a function, it must take a single required argument. This argument
    will be the observed wavelengths in the region. The return value of the function will then be used
    as dlambda.
    
    When nlambda is None, the amount of data points among the observed wavelengths will be used. Otherwise a number
    or function is expected. If nlambda is a function, it must take a single required argument. This argument
    will be the observed wavelengths in the region. The return value of the function will then be used
    as dlambda.
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
    return Region(lambda0, wav, inten, cont, dlambda, nlambda, lab_wav, scale_factor = scale_factor, noise = noise)

def new_region(atlas, lambda0, dlambda, nlambda, lab_wav, scale_factor = 1.0, noise = 1, cgs = True):
    """
    Creates a region in Region form. The required arguments are

        atlas   : The atlas object.

        lambda0 : Starting wavelength.

        dlambda : The length of each step.

        nlambda : The number of steps.
        
        lab_wav : The wavelength corresponding to the laboratory central wavelength of the line.

    The optional arguments are
    
        scale_factor : The scale factor (used for normalization)
        
        noise        : The noise of the intensity data. This may be a number or an array. If the latter it must be of the
                       same length as wav, inten and cont.
                       Default is 1.
        
        cgs          : Determines is cgs units should be used when synthezising spectral lines.
                       Default is True.

    The end point can be found with: lambda_0 + dlambda*nlambda
    The return value is a tuple of the form:
        (lambda_0, dlambda, nlambda, scale_factor)
    """
    
    obs_wav, obs_inten, obs_cont = atlas.getatlas(lambda0, lambda0 + dlambda*nlambda, cgs = cgs)
    if len(obs_wav) != nlambda:
        raise Exception("The observed data had a length " + str(len(obs_wav)) + " while nlambda was " + str(nlambda) + ". They must match.")
    return Region(lambda0, obs_wav, obs_inten, obs_cont, dlambda, nlambda, lab_wav, scale_factor = scale_factor, noise = noise)

def _new_region_from(atlas, obj):
    """
    Creates a new region from the given object, if said object is not already a region. If it
    is then it will be returned unaltered. Otherwise it is expected to be on tuple form.
    """
    
    if not isinstance(obj, Region):
        obj = new_region(atlas, *obj)
    return obj

def get_nlambda(region):
    """
    Returns the amount of steps for a region. The region can be a Region object or a tuple of the form:
        (lambda0, dlambda, nlambda, scale_factor)
    """

    if isinstance(region, Region):
        return region.nlambda
    return region[2]

def get_dlambda(region):
    """
    Returns the difference between each step for a region. The region can be a Region object or a tuple of the form:
        (lambda0, dlambda, nlambda, scale_factor)
    """

    if isinstance(region, Region):
        return region.dlambda
    return region[1]

def region(lambda0, dlambda, nlambda, scale_factor = 1.0):
    """
    Creates a region in tuple form (use new_region and new_regon_in to create Region objects instead).
    The arguments are

        lambda0      : Starting wavelength.
        
        dlambda      : The length of each step.
        
        nlambda      : The number of steps.
        
        scale_factor : The scale factor (used for normalization).
        
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
    """
    Returns the wavelengths and intensities that lie within wav[0] + left and wav[-1] - right.
    """

    return get_within(wav[0] + left, wav[-1] - right, wav, inten)

