"""
This module contains the functions and classes that synthezises spectra for different iron abundances and fits them to
the observed spectra. Other elements are not directly supported.

ABOUT THE IRON ABUNDANCES:
The abundance used in this code is the relative abundance of iron compared to hydrogen. Specifically, if the number density
of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be

    A(Fe) = log(N(Fe)) - log(N(H))

To show what this means, let us consider an example. If A(Fe) = -4.5 then the we would know that

    N(Fe)/N(H) = 10^-4.5

There are other standards for abundance, but they are not used here.
"""

# Imported for python 3 interoperability
from __future__ import print_function
from __future__ import division

# Imports
import numpy as np
import pyLTE as p
import sparsetools as sp
import fitting
import regions as regs
import abundutils as au
import multiprocessing as mp
import astropy.units
import os
import time

# Constants... as in, not really constants, but should probably never be modified during runtime
DEFAULT_MODEL_FILE = "data/falc_filled.nc"
_CONTINUUM_CFG_FILE = "data/nolines.cfg"
_ELEMENT = "Fe"

# Original value: 101
_NSHIFT = 101

def _gaussian(x, p):
    """
    Gaussian function
        x : x position
        p : a tuple of (coeff, ?expected value?, ?FWHM?)
    DOCUMENT THIS!!!
    """
    sig = p[2] / (2.0*np.sqrt(2.0*np.log(2.0)))
    z = (x - p[1]) / sig
    return p[0] * np.exp(-0.5 * z**2)

def _convolve(var, tr):
    """
    DOCIMENT THIS!!!
    """
    
    # Get the dimensions
    n = len(var)
    n1 = len(tr)
    npad = n + n1
    
    # Handle if there are an uneven amount of elements in tr
    if(n1 % 2 == 1):
        npad -= 1
        off = 1
    else:
        off = 0
    
    # Pad arrays using wrap around effect
    # INSERT WHY THIS IS DONE HERE!!!
    # First: Pad "var"
    pvar = np.zeros(npad, dtype = np.float64)
    pvar[0:n] = var                 # First 0 to n (exclusive) values corresponds to the values in "var"
    pvar[n:n + n1//2] = var[-1]      # The next n1/2 values are set to the last value of "var"
    pvar[n + n1//2::] = var[0]       # The final n1/2 values are set to the first value of "var"
    
    # Padding "tr"
    ptr = np.zeros(npad, dtype = np.float64)
    ptr[0:n1] = tr / np.sum(tr)     # WHY IS THIS DONE LIKE THIS?!??!?!?
    
    # NOTE: Due to how python 2 handles division and math it has to be: -n1/2 + off
    #       The code breaks if it would be: off - n1/2
    #       This is because python rounds the number. In order to get python 3 interoperability,
    #       this should be fixed so it will work in both python 2 and python 3.
    ptr = np.roll(ptr, -n1//2 + off)  #
    
    # WHAT DOES THIS DO AND WHY??? EXPLAIN EACH STEP!!??!?!?!?
    pvar_fft = np.fft.rfft(pvar)
    ptr_fft = np.fft.rfft(ptr)
    conv = pvar_fft * ptr_fft
    fftresult = np.fft.irfft(conv)
    return fftresult[0:n]

class ChiRegionResult(object):
    """
    ChiRegionResult is a class that represents result of a region, using chi squared to fit the synthetic
    data to the observed data. The attributes are
    
        region                  : The region these results apply to. This is an instance of the regions.Region class.
        
        wav                     : An array of the synthetic wavelengths.
        
        inten                   : The synthetic intensities, for each abundance. It is a two dimensional array, where each
                                  row corresponds to the synthetic intensities for an abundance. Note that the intensities
                                  are scaled so that the maximum is 1. To get the unscaled intensities for an abundance, multiply
                                  them with the corresponding scale factor, given in inten_scale_factor.
        
        inten_no_macroturb      : The synthetic intensities for each abundance, neglecting the effects of macroturbulence.
                                  Macroturbulence is caused by macroscopic velocity fields of the gas in the solar
                                  atmosphere. It causes Doppler shifts and thus can lead to additional broadening effects
                                  for lines. To fully handle this, a 3D calculation is needed. In the case of a 1D
                                  calculation, it can be handled somewhat by convolving the synthetic spectra with a
                                  gaussian. This attribute contains a list of the the raw data per abundance that does not
                                  handle the macroturbulence. As with the inten attribute, this is scaled to 1. The
                                  unscaled intensities for an abundance can be obtained by multiplying with the corresponding
                                  scale factor in inten_scale_factor_nm.
        
        shift                   : An array of the best shift for each abundance. This is a 1 dimensional array.
        
        chisq                   : The chi squared of each abundance, for the best shift. This is a 1 dimensional array.
        
        shift_all               : A list of all shifts.
        
        chisq_all               : A list of the chi squared values for each shift and abundance. This is expected to be a two
                                  dimensional structure, like a matrix, where each row correlates to an abundance. The content
                                  of a row is the chi squared values for all the shifts, for the abundance in question.
        
        inten_scale_factor      : A list of the scale factors of each abundance.
        
        inten_scale_factor_nm   : A list of the scale factors of each abundance, for the intensities for which the macroturbulence
                                  has been neglected. This is essentially the maximum unscaled intensities for the abundances, when
                                  macroturbulence is not handled.
        
        abund                   : An array of the iron abundances for which the synthetic data was synthezised.
        
        no_macroturb_diffs      : The differences between the intensities that handles the macroturbulence and the intensities that
                                  neglect the macroturbulence, for each abundance. Note that it takes the intensity with macroturbulence
                                  minus the intensity without macriturbulence for a given abundance.
        
        no_macroturb_chisq      : The chi squared of the intensities that handles the macroturbulence and the intensities that neglect
                                  the macroturbulence, for each abundance.
        
        best_index              : The index of the best iron abundance.
        
        best_shift              : The best shift of the best iron abundance. This is essentially the shift corresponding to best_chisq
                                  of the best iron abundance.
        
        best_chisq              : The best chi squared value of the best iron abundance. This is the lowest chi squared value for the
                                  best iron abundance.
        
        best_abund              : The best iron abundance. This is the abundance with lowest chi squared value for its best shift.
        
        best_inten              : The best synthetic intensities of the best iron abundance.
        
        best_inten_scale_factor : The scale factor of the best synthetic intensities of the best iron abundance.
        
    The best values are all the values that corresponds to the best chi squared.
    
    The iron abundances are stored in an array of floats. These numbers represents the relative abundance of iron compared to hydrogen.
    Specifically, if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected
    to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    To show what this means, let us consider an example. If A(Fe) = -4.5 then the we would know that
    
        N(Fe)/N(H) = 10^-4.5
    
    There are other standards for abundance, but they are not used here.
    """

    def __init__(self, region, wav, inten, inten_no_macroturb, shift, chisq, shift_all, chisq_all, inten_scale_factor, inten_scale_factor_nm, abund):
        """
        ChiRegionResult is a class that represents result of a region, using chi squared to fit the synthetic
        data to the observed data. This is the constructor. The arguments are
        
            region                : The region these results apply to. This is an instance of the regions.Region class.
            
            wav                   : The synthetic wavelengths. This should be an array of one dimension, or of a similar type.
            
            inten                 : The synthetic intensities, for each abundance. It should be a two dimensional array. The
                                    rows represents the intensities of the different abundances. Note that the intensities
                                    should be scaled so that the maximum is 1.
            
            inten_no_macroturb    : The synthetic intensities for each abundance, neglecting the effects of macroturbulence.
                                    Macroturbulence is caused by macroscopic velocity fields of the gas in the solar
                                    atmosphere. It causes Doppler shifts and thus can lead to additional broadening effects
                                    for lines. To fully handle this, a 3D calculation is needed. In the case of a 1D
                                    calculation, it can be handled somewhat by convolving the synthetic spectra with a
                                    gaussian. This argument should be a list of the the raw data for each abundance that
                                    does not handle the macroturbulence.
            
            shift                 : A list of the best shift for each abundance.
            
            chisq                 : The chi squared of each abundance, for the best shift.
            
            shift_all             : A list of all shifts.
            
            chisq_all             : A list of the chi squared values for each shift and abundance. This is a two dimensional
                                    array where each row correlates to an abundance. The content of a row is the chi squared
                                    values for all the shifts, for the abundance in question.
            
            inten_scale_factor    : A list of the scale factors of each abundance.
            
            inten_scale_factor_nm : A list of the scale factors of each abundance, for the intensities for which the macroturbulence
                                    is neglected. These intensities are given in the inten_no_macroturb argument.
            
            abund                 : A list of the abundances for which the synthetic data was synthezised.
        
        The iron abundances are stored in an array of floats. These numbers represents the relative abundance of iron compared to hydrogen.
        Specifically, if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected
        to be
        
            A(Fe) = log(N(Fe)) - log(N(H))
        
        To show what this means, let us consider an example. If A(Fe) = -4.5 then the we would know that
        
            N(Fe)/N(H) = 10^-4.5
        
        There are other standards for abundance, but they are not used here.
        """
        
        self.region = region
        self.wav = wav
        self.inten = inten
        self.inten_no_macroturb = inten_no_macroturb
        self.inten_scale_factor = inten_scale_factor
        self.inten_scale_factor_nm = inten_scale_factor_nm
        self.shift = shift
        self.chisq = chisq
        self.shift_all = shift_all
        self.chisq_all = chisq_all
        self.abund = abund
        
        # Calculate the differences between the intensities that handles and the intensities that neglects the macroturbulence, as
        # well as the corresponding chi squared. This is done for each abundance.
        self.no_macroturb_diffs = np.array([i - i_nm for i, i_nm in zip(inten, inten_no_macroturb)])
        self.no_macroturb_chisq = np.array([(inten_diffs**2.0).sum() for inten_diffs in self.no_macroturb_diffs])
        
        # Get the best values. These are essentially the values that corresponds to a minimum
        # chi squared.
        best = np.argmin(chisq)
        self.best_index = best
        self.best_shift = shift[best]
        self.best_chisq = chisq[best]
        self.best_abund = abund[best]
        self.best_inten = inten[best]
        self.best_inten_scale_factor = inten_scale_factor[best]
    
    def minimized_quantity(self):
        """
        Returns the best value of the quantity, in this case the chi squared of the synthetic and observed line profiles, that
        was minimized in order to fit the synthetic spectrum to the observed spectrum for this region. 
        """
        
        return self.best_chisq 
    
    def _fuse_result(self, other):
        """
        Fuses the result of this result with the other result. Essentially this assumes that the abundances of this result
        are greater then the abundances of the other result, so that the other result can just be appended to the end of
        this result (technically a new result is created, so neither this or the other result are modified).
        """
        
        # Make sure the regions are the same
        if self.region != other.region:
            raise Exception("Invalid regions")

        # Concatenate the numpy arrays, with the data from this result first
        inten = np.concatenate((self.inten, other.inten), axis = 0)
        inten_no_macroturb = np.concatenate((self.inten_no_macroturb, other.inten_no_macroturb), axis = 0)
        shift = np.concatenate((self.shift, other.shift), axis = 0)
        chisq = np.concatenate((self.chisq, other.chisq), axis = 0)
        chisq_all = np.concatenate((self.chisq_all, other.chisq_all), axis = 0)
        inten_scale_factor = np.concatenate((self.inten_scale_factor, other.inten_scale_factor), axis = 0)
        inten_scale_factor_nm = np.concatenate((self.inten_scale_factor_nm, other.inten_scale_factor_nm), axis = 0)
        abund = np.concatenate((self.abund, other.abund), axis = 0)
        
        # Return the fused result
        return ChiRegionResult(self.region, self.wav, inten, inten_no_macroturb, shift, chisq, self.shift_all, chisq_all, inten_scale_factor, inten_scale_factor_nm, abund)

class EWRegionResult(object):
    """
    EWRegionResult is a class that represents result of a region, using the equivalent width to fit the synthetic
    data to the observed data. The attributes are
    
        region                  : The region these results apply to. This is an instance of the regions.Region class.
        
        wav                     : The synthetic wavelengths. This is an array of one dimension, or of a similar type.
        
        inten                   : The synthetic intensities, for each abundance. It is a two dimensional array. The
                                  rows represents the intensities of the different abundances. Note that the intensities
                                  should be scaled so that the maximum is 1. To get  the unscaled intensity for an
                                  abundance, multiply its intensity with the corresponding scale factor (which can
                                  be obtained from the inten_scale_factor attribute)
        
        inten_no_macroturb      : The synthetic intensities for each abundance, neglecting the effects of macroturbulence.
                                  Macroturbulence is caused by macroscopic velocity fields of the gas in the solar
                                  atmosphere. It causes Doppler shifts and thus can lead to additional broadening effects
                                  for lines. To fully handle this, a 3D calculation is needed. In the case of a 1D
                                  calculation, it can be handled somewhat by convolving the synthetic spectra with a
                                  gaussian. This attribute contains a list of the the raw data per abundance that does not
                                  handle the macroturbulence. As with the inten attribute, this is scaled to 1. The
                                  unscaled intensities for an abundance can be obtained by multiplying with the corresponding
                                  scale factor in inten_scale_factor_nm.
        
        inten_scale_factor      : A list of the scale factors of each abundance. This is essentially the maximum unscaled
                                  intensities for the abundances.
        
        inten_scale_factor_nm   : A list of the scale factors of each abundance, for the intensities for which the macroturbulence
                                  has been neglected. This is essentially the maximum unscaled intensities for the abundances, when
                                  macroturbulence is not handled.
        
        obs_eq_width            : The equivalent width of the observed line.
        
        eq_width                : The equivalent widths of the synthetic lines of the different abundances. This is an array.
        
        diff                    : The differences between the observed and synthetic equivalent widths, for each abundance.
                                  This is an array.
        
        abund                   : A list of the iron abundances for which the synthetic data was synthezised.
        
        no_macroturb_diffs      : The differences between the intensities that handles the macroturbulence and the intensities that
                                  neglect the macroturbulence, for each abundance. Note that it takes the intensity with macroturbulence
                                  minus the intensity without macriturbulence for a given abundance.
        
        no_macroturb_chisq      : The chi squared of the intensities that handles the macroturbulence and the intensities that neglect
                                  the macroturbulence, for each abundance.
        
        no_macroturb_eq_width   : The equivalent width of the intensities that neglects macroturbulence, for each intensity.
        
        eq_width_unit           : The unit of the equivalent width. This should come from astropy.units.
        
        best_index              : The index of the best values. This is essentially the index of the best iron abundance.
        
        best_inten              : The synthetic intensities of the best iron abundance.
        
        best_inten_scale_factor : The scale factor corresponding to the synthetic intensities of the best iron abundance.
        
        best_eq_width           : The equivalent width of the best iron abundance.
        
        best_diff               : The difference between the equivalent widths of the synthetic and observed lines, for
                                  the best iron abundance. Specifically, this is the smallest such difference and the best
                                  abundance is simply the corresponding abundance.
        
        best_abund              : The best iron abundance. This is the abundance which has the smallest differenece between
                                  the equivalent widths of the synthetic and observed lines.
    
    Note that the best values are the values for the abundance which has the smallest differenece between the equivalent widths
    of the synthetic and observed lines.
    
    The iron abundances are stored in an array of floats. These numbers represents the relative abundance of iron compared to hydrogen.
    Specifically, if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected
    to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    To show what this means, let us consider an example. If A(Fe) = -4.5 then the we would know that
    
        N(Fe)/N(H) = 10^-4.5
    
    There are other standards for abundance, but they are not used here.
    """
    
    def __init__(self, region, wav, inten, inten_no_macroturb, inten_scale_factor, inten_scale_factor_nm, obs_eq_width, eq_width, diff, abund, eq_width_unit):
        """
        EWRegionResult is a class that represents result of a region, using the equivalent width to fit the synthetic
        data to the observed data. This is the constructor. The arguments are
        
            region                : The region these results apply to. This is an instance of the regions.Region class.
            
            wav                   : The synthetic wavelengths. This should be an array of one dimension, or of a similar type.
            
            inten                 : The synthetic intensities, for each abundance. It should be a two dimensional array. The
                                    rows represents the intensities of the different abundances. Note that the intensities
                                    should be scaled so that the maximum is 1.
            
            inten_no_macroturb    : The synthetic intensities for each abundance, neglecting the effects of macroturbulence.
                                    Macroturbulence is caused by macroscopic velocity fields of the gas in the solar
                                    atmosphere. It causes Doppler shifts and thus can lead to additional broadening effects
                                    for lines. To fully handle this, a 3D calculation is needed. In the case of a 1D
                                    calculation, it can be handled somewhat by convolving the synthetic spectra with a
                                    gaussian. This argument should be a list of the the raw data for each abundance that
                                    does not handle the macroturbulence.
            
            inten_scale_factor    : A list of the scale factors of each abundance.
            
            inten_scale_factor_nm : A list of the scale factors of each abundance, for the intensities for which the macroturbulence
                                    is neglected. These intensities are given in the inten_no_macroturb argument.
            
            obs_eq_width          : The equivalent width of the observed line.
            
            eq_width              : The equivalent widths of the synthetic lines of the different abundances.
            
            diff                  : The differences between the observed and synthetic equivalent widths, for each abundance.
            
            abund                 : A list of the abundances for which the synthetic data was synthezised.
            
            eq_width_unit         : The unit of the equivalent width. This should come from astropy.units.
        
        The iron abundances are stored in an array of floats. These numbers represents the relative abundance of iron compared to hydrogen.
        Specifically, if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected
        to be
        
            A(Fe) = log(N(Fe)) - log(N(H))
        
        To show what this means, let us consider an example. If A(Fe) = -4.5 then the we would know that
        
            N(Fe)/N(H) = 10^-4.5
        
        There are other standards for abundance, but they are not used here.
        """

        # Store the data
        self.region = region
        self.wav = wav
        self.inten = inten
        self.inten_no_macroturb = inten_no_macroturb
        self.inten_scale_factor = inten_scale_factor
        self.inten_scale_factor_nm = inten_scale_factor_nm
        self.obs_eq_width = obs_eq_width
        self.eq_width = eq_width
        self.diff = diff
        self.abund = abund
        self.eq_width_unit = eq_width_unit
        
        # Calculate the differences between the intensities that handles and the intensities that neglects the macroturbulence, as
        # well as the corresponding chi squared. This is done for each abundance.
        self.no_macroturb_diffs = np.array([i - i_nm for i, i_nm in zip(inten, inten_no_macroturb)])
        self.no_macroturb_chisq = np.array([(inten_diffs**2.0).sum() for inten_diffs in self.no_macroturb_diffs])
        
        conv_factor = (1 * astropy.units.AA).to(eq_width_unit).value
        self.no_macroturb_eq_width = np.array([conv_factor*_equivalent_width(wav, i) for i in inten_no_macroturb])
        
        # Get the best values
        best = np.argmin(abs(diff))
        self.best_index = best
        self.best_inten = inten[best,:]
        self.best_inten_scale_factor = inten_scale_factor[best]
        self.best_eq_width = eq_width[best]
        self.best_diff = diff[best]
        self.best_abund = abund[best]

    def minimized_quantity(self):
        """
        Returns the best value of the quantity, in this case absolute value of the difference in equivalent width between
        the synthetic and observed lines, that was minimized in order to fit the synthetic spectrum to the observed spectrum
        for this region.
        """
        
        return abs(self.best_diff)

    def _fuse_result(self, other):
        """
        Fuses the result of this result with the other result. Essentially this assumes that the abundances of this result
        are greater then the abundances of the other result, so that the other result can just be appended to the end of
        this result (technically a new result is created, so neither this or the other result are modified).
        """
        
        # Make sure the regions are the same
        if self.region != other.region:
            print(self.region)
            print(other.region)
            raise Exception("Invalid region")

        # Conversion factor, makes sure the unit of equivalent width of this object always wins over the other objects unit
        conv_factor = (1 * other.eq_width_unit).to(self.eq_width_unit).value

        # Concatenate the numpy arrays, with the data from this result first
        inten = np.concatenate((self.inten, other.inten), axis = 0)
        inten_no_macroturb = np.concatenate((self.inten_no_macroturb, other.inten_no_macroturb), axis = 0)
        inten_scale_factor = np.concatenate((self.inten_scale_factor, other.inten_scale_factor), axis = 0)
        inten_scale_factor_nm = np.concatenate((self.inten_scale_factor_nm, other.inten_scale_factor_nm), axis = 0)
        eq_width = np.concatenate((self.eq_width, other.eq_width*conv_factor), axis = 0)
        diff = np.concatenate((self.diff, other.diff*conv_factor), axis = 0)
        abund = np.concatenate((self.abund, other.abund), axis = 0)
        
        # Return a new region result
        return EWRegionResult(self.region, self.wav, inten, inten_no_macroturb, inten_scale_factor, inten_scale_factor_nm, self.obs_eq_width, eq_width, diff, abund, self.eq_width_unit)

class SynthResult(object):
    """
    SynthResult encapsulates the result of a fit. This is the constructor. The arguments are
        
            region_result      : A list of region result objects. These are either instances of ChiRegionResult
                                 or EWRegionResult, depending on how the fit was done.
            
            region_data        : An array containing the region data used in the calculations. The elements in
                                 this array are tuples. Specifically they are regions on tuple form (see the
                                 regions module for more information).
            
            wav                : The wavelengths of the synthetic data.
            
            raw_synth_data     : A list containing the raw synthetic data.
            
            best_abunds        : An array of the best abundances of each region.
            
            abund              : The mean abundance. It is the mean of best_abunds.
            
            error_abund        : The error of the mean abundance. This is essentially the standard deviation
                                 of best_abunds.
            
            minimized_quantity : An array of the minimized values of each region of the quantity the was minimized to fit the
                                 synthetic and observed lines. 
    """
    
    def __init__(self, region_result, region_data, wav, raw_synth_data):
        """
        SynthResult encapsulates the result of a fit. This is the constructor. The arguments are
        
            region_result  : A list of region result objects. These are either instances of ChiRegionResult
                             or EWRegionResult, depending on how the fit was done.
            
            region_data    : An array containing the region data used in the calculations.
            
            wav            : The wavelengths of the synthetic data.
            
            raw_synth_data : A list containing the raw synthetic data.
        """
        
        self.region_result = region_result
        self.region_data = region_data
        self.wav = wav
        self.raw_synth_data = raw_synth_data
        
        #
        self.best_abunds = np.array([r.best_abund for r in region_result])
        self.abund = np.mean(self.best_abunds)
        self.error_abund = np.std(self.best_abunds)
        self.minimized_quantity = np.array([r.minimized_quantity() for r in region_result])
    
    def _fuse_result(self, other):
        """
        Fuses two results of a fit, assuming it was done with the chi squared method
        """
        
        if not np.array_equal(self.region_data, other.region_data):
            raise Exception("Region data must be the same for result1 and result2")
        if not np.array_equal(self.wav, other.wav):
            raise Exception("Wavelength data must be the same for result1 and result2")
        
        # Fuse the region results
        region_result = []
        for r1, r2 in zip(self.region_result, other.region_result):
            region_result.append(r1._fuse_result(r2))
        if len(self.region_result) > len(other.region_result):
            region_result.extend(self.region_result[len(other.region_result):])
        elif len(self.region_result) < len(other.region_result):
            region_result.extend(other.region_result[len(self.region_result):])

        # Append the raw data of result2 after result1 (note that since raw_synth_data are
        # python lists + means concatenation and not elementwise addition, as would be the
        # case if they where arrays)
        raw_synth_data = self.raw_synth_data + other.raw_synth_data
        
        # Return the fused result
        return SynthResult(region_result, self.region_data, self.wav, raw_synth_data)

class ResultPair(object):
    """
    Represents a result pair. One result comes from using the chi squared method, the other
    from using equivalent widths. The attributes are
    
        result_chi : The result from a fit, obtained using the chi squared method.
        
        result_ew  : The result from a fit, obtained using equivalent widths.
    """
    
    def __init__(self, result_chi, result_ew):
        """
        Constructor for the ResultPair class, which is a class that represents a result pair from two fits.
        One result comes from using the chi squared method, the other from using equivalent widths. The
        required arguments are
    
            result_chi : The result from a fit, obtained using the chi squared method.
        
            result_ew  : The result from a fit, obtained using equivalent widths.
        """
        
        self.result_chi = result_chi
        self.result_ew = result_ew
    
    def _fuse_result(self, other):
        """
        Fuses two ResultPair objects together
        """
        
        return ResultPair(self.result_chi._fuse_result(other.result_chi), self.result_ew._fuse_result(other.result_ew))

def save_computation(filename, result_pair):
    np.save(filename, np.array([synth_data[0,0,0,:,0] for synth_data in result_pair.result_chi.raw_synth_data]))

def _fuse_result(result_list):
    """
    Fuses a list of results together.
    """
    
    # Fuse the result together
    result = result_list[0]
    for r in result_list[1:]:
        result = result._fuse_result(r)
    return result

def _synth(s, m):
    """
    Helper function that synthazises a line.
    """
    
    return s.synth(m.ltau, m.temp, m.pgas, m.vlos, m.vturb, m.B, m.inc, m.azi, False)

def _setup_regions(atlas, regions):
    """
    Ensures all regions are instances of regions.Region. Regions on tuple form are converted.
    """
    
    # Setup the region data
    region_list = list(regions)
    for ri, r in enumerate(regions):
        if not isinstance(r, regs.Region):
            region_list[ri] = regs.new_region(atlas, *regions[ri])
    return region_list

def _setup_region_data(regions):
    """
    Creates a numpy array containing the region data. Specifically, each element in the array is
    a tuple
    
        (lambda0, dlambda, nlambda, scale_factor)

    where lambda0 is a float describing the wavelength the region starts at, dlambda is a float that
    determines the length of each step, nlambda is an integer that determines how many steps are used
    and scale_factor is a float that determines the scale of the synthetic data (specifically the
    synthetic intensity).
    """
    
    # Copy the region list and setup an array with the region data
    region_data = np.zeros(len(regions), dtype = "float64, float64, int32, float64")
    for ri, r in enumerate(regions):
        region_data[ri] = r.to_tuple()
    return region_data

def _parallel_call(conn, func, args):
    """
    This function ensures that when the calculation is done, it is sent to the main process, and if an
    error occured, it will not cause the main process to just wait forever. Instead None is sent back.
    And the connection is always closed.
    
    Note that this function is used internally and should be considered private. Use at own risk.
    """
    
    try:
        result = func(*args)
        conn.send(result)
    except:
        conn.send(None)
        raise
    finally:
        conn.close()

def _parallel_calc(abund_range, processes, func, args, verbose):
    """
    The function distributes abundance calculations over processes, take their results, fuses them together
    and returns that fused result. The required arguments are
    
        abund_range : The iron aundencies to distribute.
        
        processes   : The amount of processes to use.
        
        func        : The function that performs the calculations. Note that the function should take the abundance as first argument.
        
        args        : The arguments to the function, abundance excluded.
        
        verbose     : Determines if extra information should be printed.

    Returns a SynthResult object containing the result of all calculations.
    """
    
    # Ensure the arguments are in the form of a tuple
    args = tuple(args)
    
    # List the abundance range. By doing this, abund_range can be an iterator and the code will still work as
    # long as it is not infinite. Otherwise it might break down since we need the length of abund_range as well
    # as the ability to slice it in order to distribute the abundancies over the processes. This won't work
    # in general for an iterator.
    abund_range = list(abund_range)

    # Distribute the abundances amongst the processes
    abund_range = list(abund_range)
    abunds = [[] for _ in range(processes)]
    abunds_per_process = int(np.ceil(float(len(abund_range)) / float(processes)))
    for i in range(processes):
        si = i*abunds_per_process
        ei = (i + 1)*abunds_per_process
        abunds[i].extend(abund_range[si:ei])
    
    # If in verbose mode, print information
    if verbose:
        print("Parallel computation enabled. Processes used:", processes)

    # Spawn the processes
    proc_list = []
    conns = []
    for a in abunds:
        # Concatenate the abundance and the rest of the arguments
        curr_args = (a,) + args
        
        # Spawn the process
        rec, conn = mp.Pipe()
        p = mp.Process(target = _parallel_call, args = (conn, func, curr_args))
        p.start()
        proc_list.append(p)
        conns.append(rec)
    
    # Join the processes
    result_list = []
    for rec, p in zip(conns, proc_list):
        result_list.append(rec.recv())
        rec.close()
        p.join()
    
    # Fuse the results and return it   
    return _fuse_result(result_list)

def _fit_regions_chi(regions, wav, synth_data, abunds, verbose):
    """
    Fits the regions of the synthetic data to the corresponding observed data
    """
    
    abund_count = len(abunds)
    
    # A list of the chi sqaured values for each region
    region_result = []
    
    # Create shifts
    nshift = _NSHIFT
    shift = 0.2*(np.arange(nshift) / (nshift - 1.0)) - 0.1
    
    # Display some information, if in verbose mode
    if verbose:
        print("Number of shifts: ", nshift)
        print("Shift step length:", shift[1]-shift[0])
    
    # Take current time again, this time to time the fitting phase
    start_time = time.time()

    # For each region
    for ri, r in enumerate(regions):
        # Get the number of data points that the region should have
        nlambda = r.nlambda
        
        # Create the array containing the best shifts
        rshift = np.zeros(abund_count)
        rinten = []
        
        # Create an zeroed array of size (amount-of-abundencies,nshift) to store the chi squared for each abundance and for each shift
        # within the current region.
        rchisq = np.zeros((abund_count,nshift), dtype = np.float64)
        
        # Create a zeroed array containing only the best chi squared values for each abundance, within the current region.
        chisq = np.zeros(abund_count, dtype = np.float64)
        
        # Array containing the maximum synthetic intensities for each abundance
        inten_max = np.zeros(abund_count, dtype = np.float64)
        inten_max_nm = np.zeros(abund_count, dtype = np.float64)
        
        # Get the observed wavelengths and intensities in the current region
        robs_wav = r.wav
        robs_inten = r.inten
        noise = r.noise
#        robs_inten = r.inten*r.inten_scale_factor/r.cont

        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions. Specifically, it is used later on when convolving the synthetic data.
        tw = (np.arange(15)-7)*(robs_wav[1] - robs_wav[0])
        psf = _gaussian(tw, [1.0, 0.0, 1.83*r.lambda_end/300000.0])
        reduced_psf = psf / psf.sum()
        
        # Create lists that contains all the wavelengths and intensities for the different abundances
        # within the current region
        rwav_all = []
        rsynth_inten_all = []
        rsynth_inten_all_nm = []

        for a, synth_inten in enumerate(synth_data):
            # Get the relevant synthetic intensity
            synth_inten = synth_inten[0,0,0,:,0]
            
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rsynth_inten_nm = r.get_contained(wav, synth_inten, left_padding = 1e-9)
            
            # Handle errors due to math with floating point numbers and things like that
            if len(rwav) != nlambda:
                if verbose:
                    print("******************")
                    print("The length of the synthetic data in the region did not match nlambda.")
                    print("Length of synthetic data:", len(rwav), ", nlambda:", nlambda)
                    print("Region start:", r.lambda0, ", region end:", r.lambda_end)
                    print("******************")
                rwav = rwav[:nlambda]
                rsynth_inten_nm = rsynth_inten_nm[:nlambda]

            # CHECK IF THIS DESCRIPTION IS CORRECT!!!
            #
            # Convolve the synthetic data. This is done to handle broadening effects due to convective motions in the
            # solar atmosphere. These broadening effects has a Gaussian distribution, so to take it into account we
            # need to calculate the convolution of the synthetic lines and the gaussian profile calculated above.
            rsynth_inten = _convolve(rsynth_inten_nm, reduced_psf)
            inten_max[a] = rsynth_inten.max()
            rsynth_inten /= inten_max[a]
#            rsynth_inten = rsynth_inten * (r.cont/r.cont.max()) / rsynth_inten.max()
        
            # Store the synthetic wavelengths and intensities in this region for later calculations
            rwav_all.append(rwav)
            rsynth_inten_all.append(rsynth_inten)
        
            # Calculate the chi squared for each shift
            for ii in range(nshift):
                # To calculate the chi squared for each shift, we use interpolation to shift the synthetic
                # data a little bit.
                interp_syn = np.interp(rwav, rwav - shift[ii], rsynth_inten)

                # Calculate and store the chi squared
                # Note that this assumes that the wavelengths of the observed data and the wavelengths of
                # the synthetic (and thereby interpolated) data are the same. This is not really true, but
                # the error is small enough that this should not matter too much.
                rchisq[a,ii] = (((robs_inten - interp_syn)**2) / noise).sum()
            
            # Get and store the best shift
            best_shift = shift[np.argmin(rchisq[a,:])]
            rshift[a] = best_shift
            
            # Calculate the shifted intensity spectrum using linear interpolation
            shifted_inten = np.interp(rwav, rwav - best_shift, rsynth_inten)
            rinten.append(shifted_inten)
            
            # Store the scaled intensities and scaling factor (the max value of the unscaled intensities) for the intensities
            # that neglect macroturbulence... also make sure it is shifted according to the best shift.
            inten_max_nm[a] = rsynth_inten_nm.max()
            shifted_inten = np.interp(rwav, rwav - best_shift, rsynth_inten_nm / inten_max_nm[a])
            rsynth_inten_all_nm.append(shifted_inten)

        # Calculate the chi squared for each abundance, for the best shift
        for a, (rwav, rsynth_inten) in enumerate(zip(rwav_all, rsynth_inten_all)):
            interp_syn = np.interp(rwav, rwav - rshift[a], rsynth_inten)
            chisq[a] = ((robs_inten - interp_syn)**2).sum()

        # Add the result of the current region to the list of region results
        region_result.append(ChiRegionResult(r, rwav, np.array(rinten), np.array(rsynth_inten_all_nm), rshift, chisq, shift, rchisq, inten_max, inten_max_nm, abunds))

    # If in verbose mode, display fitting timing
    if verbose:
        end_time = time.time()
        print("Fitting time:", end_time - start_time, "seconds")
    
    # Return the region result
    return region_result

def _fit_chi(abund_range, cfg_file, regions, model_file, verbose):
    """
    Synthazises spectral lines and attempt to fit them to the observed data using the chi squared
    method.
    """
    
    # Create the abundance updates and check them
    abund_updates = [au.abund(_ELEMENT, a) for a in abund_range]
    abund_range = np.array(abund_range)
        
    # Copy the region list and setup an array with the region data
    region_data = _setup_region_data(regions)

    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(model_file if model_file != None else DEFAULT_MODEL_FILE)
    
    # Take current time
    start_time = time.time()
    
    # Generate the synthetic lines
    synth_data = []
    for a in abund_updates:
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))

    # If in verbose mode, display time
    if verbose:
        end_time = time.time()
        print("Synth time:", end_time - start_time, "seconds")

        # Display some information about the amount of regions and abundances to test
        print("Number of regions to test:", len(regions))
        print("Number of abundances to test:", len(abunds))

    # Get the wavelengths
    wav = s.getwav()
    
    # Fit the regions (kind of... technically this determines how to shift the regions and how well everything then fits)
    region_result = _fit_regions_chi(regions, wav, synth_data, abund_range, verbose)
    
    # Return the result
    return SynthResult(region_result, region_data, wav, synth_data)

def fit_chi(cfg_file, atlas, regions, abund_range, model_file = None, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
        
        model_file        : Sets the model file. If this is None, the default model file specified by DEFAULT_MODEL_FILE will be used.
                            Default is None.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.
    
    Returns a SynthResult object containing the result of all calculations.

    Fitting is done by using chi squared to compare the observed and synthetic spectrum for the different regions. Essentially, if we have
    a region, then for each abundance the chi squared of the observed and synthetic spectrum is calculated. The abundance with smallest
    chi squared is then taken as the best value.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance of iron compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, but they are not used here.
    """
    
    regions = _setup_regions(atlas, regions)
    return _fit_chi(abund_range, cfg_file, regions, model_file, verbose)

def fit_chi_parallel(cfg_file, atlas, regions, abund_range, processes = 2, model_file = None, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum, but distributes the work over several processes
    rather then doing it directly. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
    
        processes         : The amount of working processes.
                            Default is 2.
        
        model_file        : Sets the model file. If this is None, the default model file specified by DEFAULT_MODEL_FILE will be used.
                            Default is None.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object containing the result of all calculations.

    Fitting is done by using chi squared to compare the observed and synthetic spectrum for the different regions. Essentially, if we have
    a region, then for each abundance the chi squared of the observed and synthetic spectrum is calculated. The abundance with smallest
    chi squared is then taken as the best value.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance of iron compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, but they are not used here.
    
    The distribution of work over the processes is done by making the processes handle different abundancies. For example, if we have R
    different regions, A different abundancies and N processes are used, then each process works with R different regions and approximately
    A / N different abundancies. When R and A are small the overhead of using multiple processes and coordinating them can make this slower
    then just calling fit_chi. However, when A and R increases this parallel approach tends to save time since the calculations becomes
    the bottleneck rather then the overhead of using multiple processes.
    
    Note that each processes is in essence calling fit_chi but for different abundancies.
    """
    
    regions = _setup_regions(atlas, regions)
    return _parallel_calc(abund_range, processes, _fit_chi, (cfg_file, regions, model_file, verbose), verbose)

#def _fit_spectrum(

def _equivalent_width(wav, inten):
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

def _fit_regions_width(abund_range, regions, eq_width_unit, synth_data, wav, verbose):
    """
    Fits the regions using equivalent widths.
    """

    # Conversion factor
    conv_factor = (1 * astropy.units.AA).to(eq_width_unit).value
    
    # Get the abundance count
    abund_count = len(abund_range)
    
    # Get the synthetic intensities for the abundances and the wavelength
    synth_inten = [sd[0,0,0,:,0] for sd in synth_data]

    # Take current time again, this time to time the fitting phase
    start_time = time.time()

    # Fit the data in each region using equivalent widths
    result = []
    for ri, r in enumerate(regions):
        # Create an array that will contain the maximum intensities of the different abundances, for
        # this region.
        inten_max = np.zeros(abund_count, dtype = np.float64)

        # Create an array that will contain the maximum intensities of the different abundances, for
        # this region. Note that the intensities in question are neglecting macroturbulence.
        inten_max_nm = np.zeros(abund_count, dtype = np.float64)
        
        # Create an array that will contain the equivalent widths of the different abundances, for
        # this region.
        eq_width = np.zeros(abund_count, dtype = np.float64)
        
        # Create an array that will contain the differences between the equivalent widths of the
        # synthetic and observed lines of the different abundances, for this region.
        diff = np.zeros(abund_count, dtype = np.float64)
        
        # Create an array that will store the scaled and convolved synthetic intensities for each abundance.
        sinten = np.zeros((abund_count,r.nlambda), dtype = np.float64)
        
        # Create an array that will store the scaled synthetic intensities for each abundance. Note that it
        # does not carry the convolved intensities. This is essentially the synthetic intensities when
        # macroturbulence is neglected.
        sinten_nm = np.zeros((abund_count,r.nlambda), dtype = np.float64)
        
        # Get the observed wavelengths and intensities of the region
        robs_wav = r.wav
        robs_inten = r.inten

        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions. Specifically, it is used later on when convolving the synthetic data.
        tw = (np.arange(15)-7)*(robs_wav[1] - robs_wav[0])
        psf = _gaussian(tw, [1.0, 0.0, 1.83*r.lambda_end/300000.0])
        reduced_psf = psf / psf.sum()
        
        # Calculate the equivalent width of the observed data
        obs_ew = _equivalent_width(robs_wav, robs_inten) * conv_factor
        
        for ai, a in enumerate(abund_range):
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rsynth_inten_nm = r.get_contained(wav, synth_inten[ai], left_padding = 1e-9)
            
            # Handle errors due to math with floating point numbers
            if len(rwav) != r.nlambda:
                if verbose:
                    print("******************")
                    print("The length of the synthetic data in the region did not match nlambda.")
                    print("Length of synthetic data:", len(rwav), ", nlambda:", nlambda)
                    print("Region start:", r.lambda0, ", region end:", r.lambda_end)
                    print("******************")
                rwav = rwav[:r.nlambda]
                rsynth_inten_nm = rsynth_inten_nm[:r.nlambda]

            # CHECK IF THIS DESCRIPTION IS CORRECT!!!
            #
            # Convolve the synthetic data. This is done to handle broadening effects due to convective motions in the
            # solar atmosphere. These broadening effects has a Gaussian distribution, so to take it into account we
            # need to calculate the convolution of the synthetic lines and the gaussian profile calculated above.
            rsynth_inten = _convolve(rsynth_inten_nm, reduced_psf)
            inten_max[ai] = rsynth_inten.max()
            rsynth_inten /= inten_max[ai]
            
            # Store the scaled intensities and scaling factor (the max value of the unscaled intensities) for the intensities
            # that neglect macroturbulence
            inten_max_nm[ai] = rsynth_inten_nm.max()
            sinten_nm[ai,:] = rsynth_inten_nm / inten_max_nm[ai]
            
            # Calculate the equivalent width
            eq_width[ai] = _equivalent_width(rwav, rsynth_inten) * conv_factor
            diff[ai] = eq_width[ai] - obs_ew
            sinten[ai,:] = rsynth_inten
        result.append(EWRegionResult(r, rwav, sinten, sinten_nm, inten_max, inten_max_nm, obs_ew, eq_width, diff, abund_range, eq_width_unit))

    # If in verbose mode, display fitting timing
    if verbose:
        end_time = time.time()
        print("Fitting time:", end_time - start_time, "seconds")
    
    # Return the result
    return result

def _fit_width(abund_range, cfg_file, regions, eq_width_unit, model_file, verbose):
    """
    Creates a synthetic spectrum and fits it against the observed spectrum using equivalent widths.
    """
    
    # Create the abundance updates and check them
    abund_updates = [au.abund(_ELEMENT, a) for a in abund_range]
    abund_range = np.array(abund_range)
    
    # Setup the regions
    region_data = _setup_region_data(regions)
    
    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(model_file if model_file != None else DEFAULT_MODEL_FILE)
    
    # Take current time
    start_time = time.time()
    
    # Synth the spectrum
    synth_data = []
    for a in abund_updates:
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))
    
    # Get the wavelengths of the synthetic data
    wav = s.getwav()
    
    # Display the amount of abundances
    if verbose:
        end_time = time.time()
        print("Synth time:", end_time - start_time, "seconds")
        print("Number of regions to test:", len(regions))
        print("Number of abundances to test:", len(abund_updates))
        print("Equivalent width unit:", eq_width_unit)
    
    # Fit the regions
    region_result = _fit_regions_width(abund_range, regions, eq_width_unit, synth_data, wav, verbose)
    
    # Return the result
    return SynthResult(region_result, region_data, wav, synth_data)

def fit_width(cfg_file, atlas, regions, abund_range, eq_width_unit = astropy.units.pm, model_file = None, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum for different iron abundancies. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
    
        eq_width_unit     : The unit for the equivalent width. These units come from astropy, specifically the module astropy.units.
                            Default is astropy.units.pm, which stands for picometers.

        model_file        : Sets the model file. If this is None, the default model file specified by DEFAULT_MODEL_FILE will be used.
                            Default is None.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object that contains the results of all calculations.

    Fitting is done by calculating the equivalent width of each region, both for the synthetic data and for the observed data. The abundance which
    gives the equivalent width that matches with the equivalent width of the observed data is taken as the best iron abundance.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance of iron compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, but they are not used here.
    """
    
    regions = _setup_regions(atlas, regions)
    return _fit_width(abund_range, cfg_file, regions, eq_width_unit, model_file, verbose)

def fit_width_parallel(cfg_file, atlas, regions, abund_range, processes = 2, eq_width_unit = astropy.units.pm, model_file = None, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum for different iron abundancies. It does so in parallel, distributing
    the calculations over the given amount of processes. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
    
        processes         : The amount of processes to distribute the work over.
        
        eq_width_unit     : The unit for the equivalent width. These units come from astropy, specifically the module astropy.units.
                            Default is astropy.units.pm, which stands for picometers.
        
        model_file        : Sets the model file. If this is None, the default model file specified by DEFAULT_MODEL_FILE will be used.
                            Default is None.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object that contains the results of all calculations.

    Fitting is done by calculating the equivalent width of each region, both for the synthetic data and for the observed data. The abundance which
    gives the equivalent width that matches with the equivalent width of the observed data is taken as the best iron abundance.
    
    The iron abundances are given as a range of float numbers. These numbers represents the relative abundance of iron compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, but they are not used here.
    
    The distribution of work over the processes is done by making the processes handle different abundancies. For example, if we have R
    different regions, A different abundancies and N processes are used, then each process works with R different regions and approximately
    A / N different abundancies. When R and A are small the overhead of using multiple processes and coordinating them can make this slower
    then just calling fit_width. However, when A and R increases this parallel approach tends to save time since the calculations becomes
    the bottleneck rather then the overhead of using multiple processes.
    
    Note that each processes is in essence calling fit_width but for different abundancies.
    """
    
    regions = _setup_regions(atlas, regions)
    return _parallel_calc(abund_range, processes, _fit_width, (cfg_file, regions, eq_width_unit, model_file, verbose), verbose)

def _fit_spectrum(abund_range, cfg_file, regions, eq_width_unit, model_file, verbose):
    """
    Creates a synthetic spectrum and fits it against the observed spectrum using equivalent widths.
    """
    
    # Create the abundance updates and check them
    abund_updates = [au.abund(_ELEMENT, a) for a in abund_range]
    abund_range = np.array(abund_range)
    
    # Setup the regions
    region_data = _setup_region_data(regions)
    
    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(model_file if model_file != None else DEFAULT_MODEL_FILE)
    
    # Take current time
    start_time = time.time()
    
    # Synth the spectrum
    synth_data = []
    for a in abund_updates:
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))
    
    # Get the wavelengths of the synthetic data
    wav = s.getwav()
    
    # Display the amount of abundances
    if verbose:
        end_time = time.time()
        print("Synth time:", end_time - start_time, "seconds")
        print("Number of regions to test:", len(regions))
        print("Number of abundances to test:", len(abund_updates))
        print("Equivalent width unit:", eq_width_unit)
    
    # Fit the regions
    region_result_chi = _fit_regions_chi(regions, wav, synth_data, abund_range, verbose)
    region_result_ew = _fit_regions_width(abund_range, regions, eq_width_unit, synth_data, wav, verbose)
    
    # Return a pair of results, one from using chi squared and one from using equivalent widths
    return ResultPair(SynthResult(region_result_chi, region_data, wav, synth_data), SynthResult(region_result_ew, region_data, wav, synth_data))

def fit_spectrum(cfg_file, atlas, regions, abund_range, processes = 2, eq_width_unit = astropy.units.pm, model_file = None, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum for different iron abundancies. It does so in parallel, distributing
    the calculations over the given amount of processes. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
    
        processes         : The amount of processes to distribute the work over.
        
        eq_width_unit     : The unit for the equivalent width. These units come from astropy, specifically the module astropy.units.
                            Default is astropy.units.pm, which stands for picometers.
        
        model_file        : Sets the model file. If this is None, the default model file specified by DEFAULT_MODEL_FILE will be used.
                            Default is None.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a ResultPair object that contains the results of all calculations.

    Fitting is done twise. Once using chi squared to compare the observed and synthetic spectrum for the different regions. Essentially, if we have
    a region, then for each abundance the chi squared of the observed and synthetic spectrum is calculated. The abundance with smallest chi squared
    is then taken as the best value. The second fit is done by calculating the equivalent width of each region, both for the synthetic data and for
    the observed data. The abundance which gives the equivalent width that matches with the equivalent width of the observed data is taken as the
    best iron abundance.
    
    The iron abundances are given as a range of float numbers. These numbers represents the relative abundance of iron compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, but they are not used here.
    
    The distribution of work over the processes is done by making the processes handle different abundancies. For example, if we have R
    different regions, A different abundancies and N processes are used, then each process works with R different regions and approximately
    A / N different abundancies.
    """
    
    regions = _setup_regions(atlas, regions)
    return _parallel_calc(abund_range, processes, _fit_spectrum, (cfg_file, regions, eq_width_unit, model_file, verbose), verbose)

