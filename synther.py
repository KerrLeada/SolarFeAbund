"""
This module contains the functions and classes that synthezises spectra. Specifically
a single function and class.
"""

# Imported for python 3 interoperability (also, the division import will make
# division work as in python 3, where x / y always returns a float no matter
# if x and y are integers or not... been bitten by the normal python 2 behaviour
# so this helps).
from __future__ import print_function
#from __future__ import division

# Imports
import numpy as np
import pyLTE as p
import sparsetools as sp
import fitting
import regions as regs
from abundutils import abund, check_abund

_MODEL_FILE = "data/falc_filled.nc"

def _gaussian(x, p):
    """
    Gaussian function
        x - x position
        p - a tuple of (coeff, ?expected value?, ?FWHM?)
    """
    sig = p[2] / (2.0*np.sqrt(2.0*np.log(2.0)))
    z = (x - p[1]) / sig
    return p[0] * np.exp(-0.5 * z**2)

def _convolve(var, tr):
    """
    """
    
    # Get the dimensions
    n = len(var)
    n1 = len(tr)
    npad = n + n1
    
    # WHAT DOEES THIS DO??? DOES IT CHECK FOR AN EVEN AMOUNT OF ELEMENTS
    # IN tr???
    if((n1/2)*2 != n1):
        npad -= 1
        off = 1
    else:
        off = 0
    
    # Pad arrays using wrap around effect
    # INSERT WHY THIS IS DONE HERE!!!
    # First: Pad "var"
    pvar = np.zeros(npad, dtype = np.float64)
    pvar[0:n] = var                 # First 0 to n (exclusive) values corresponds to the values in "var"
    pvar[n:n + n1/2] = var[-1]      # The next n1/2 values are set to the last value of "var"
    pvar[n + n1/2::] = var[0]       # The final n1/2 values are set to the first value of "var"
    
    # Padding "tr"
    ptr = np.zeros(npad, dtype = np.float64)
    ptr[0:n1] = tr / np.sum(tr)     # WHY IS THIS DONE LIKE THIS?!??!?!?
    
    # NOTE: Due to how python 2 handles division and math it has to be: -n1/2 + off
    #       The code breaks if it would be: off - n1/2
    #       This is because python rounds the number. In order to get python 3 interoperability,
    #       this should be fixed so it will work in both python 2 and python 3.
    ptr = np.roll(ptr, -n1/2 + off)  #
    
    # WHAT DOES THIS DO AND WHY??? EXPLAIN EACH STEP!!??!?!?!?
    pvar_fft = np.fft.rfft(pvar)
    ptr_fft = np.fft.rfft(ptr)
    conv = pvar_fft * ptr_fft
    fftresult = np.fft.irfft(conv)
    return fftresult[0:n]

def _min_of(wav, data, region):
    # Create an interval over the region
    start_wl = region.lambda0
    end_wl = region.lambda_end
    interval = (wav >= start_wl) & (wav <= end_wl)
    
    # Find the minimum value
    region_data = data[interval]
    min_value = min(region_data)

    # Find the corresponding wavelength
    region_wl = wav[interval]
    min_value_wl = region_wl[region_data == min_value][0]

    # Return the wavelength and value of the minimum
    return min_value_wl, min_value

class SynthResult(object):
    """
    Encapsulates the result synth_spectrum
    """

    def __init__(self, synth_data, LTE, model, abund_updates, region_data):
        self.synth_data = synth_data
        self.inten = [s[0,0,0,:,0] for s in synth_data]
        self.wav = LTE.getwav()
        self.LTE = LTE
        self.model = model
        self.abund_updates = abund_updates
        self.region_data = region_data

    def region_min(self):
        """
        Calculates the minimum values of each region in the synthetic spectrum, for each
        abundance. Specifically, it will return two arrays. Each row of these arrays will
        represent an abundance while each column a region. The first array will contain
        the minimum values, while the second array contains the corresponding wavelengths.
        """
        result_wl = []
        result_vals = []
        for inten in self.inten:
            wl_min_values = []
            min_values = []
            for r in self.region_data:
                mwl, mv = _min_of(self.wav, inten, r)
                wl_min_values.append(mwl)
                min_values.append(mv)
            result_wl.append(wl_min_values)
            result_vals.append(min_values)
        return np.array(result_vals), np.array(result_wl)

class RegionResult(object):
    def __init__(self, region, wav, inten, shift, chisq, shift_all, chisq_all, inten_norm_factor, abund):
        self.region = region
        self.wav = wav
        self.inten = inten
        self.shift = shift
        self.chisq = chisq
        self.shift_all = shift_all
        self.chisq_all = chisq_all
        self.inten_norm_factor = inten_norm_factor
        self.abund = abund

class FittedSynthResult(object):
    """
    Encapsulates the result of a fitted specrum synthesis
    """
    
    def __init__(self, region_result, region_data, wav, raw_synth_data):
        self.region_result = region_result
        self.region_data = region_data
        self.wav = wav
        self.raw_synth_data = raw_synth_data

class FitResult:
    def __init__(self, abund, wav, inten, dlambda, chisq):
        self.abund = abund
        self.wav = wav
        self.inten = inten
        self.dlambda = dlambda
        self.chisq = chisq

def _synth(s, m):
    return s.synth(m.ltau, m.temp, m.pgas, m.vlos, m.vturb, m.B, m.inc, m.azi, False)

def fit_spectrum(cfg_file, obs_wav, obs_inten, regions, abund_range, elem = "Fe", use_default_abund = True, interp_obs = False, verbose = False):
    """
    This files synthesizes a spectrum and attempts to fit it to the given observed spectrum.
    """
    
    # Create the updates and check them
    abund_updates = [abund(elem, a) for a in abund_range]
    check_abund(abund_updates)

    # Setup the region data
    region_data = np.zeros(len(regions), dtype = "float64, float64, int32, float64")
    for ri in range(len(regions)):
        if isinstance(regions[ri], tuple):
            region_data[ri] = regions[ri]
            regions[ri] = regs.new_region(obs_wav, obs_inten, *regions[ri])
        else:
            print(regions[ri].to_tuple())
            region_data[ri] = regions[ri].to_tuple()

    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Generate the synthetic lines
    synth_data = []
    syn = []
    if use_default_abund:
        abund_updates = [[]] + abund_updates
#        synth_data.append(_synth(s, m))
    for a in abund_updates:
        # Update the abundence and synthasize a new spectrum
        print(a)
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))
#    if use_default_abund:
#        abund_updates = [[]] + abund_updates

    # Get the wavelengths
    wav = s.getwav()
    
    # Fit the data
    abund_count = len(abund_updates)
    
#    print("SHIFT!!!\n", shift, "\nNO MORE SHIFT!!!")
    
    # A list of the chi sqaured values for each region
    syn =  [[]]*len(region_data)
    region_result = []
    
    # For each region
    for ri, r in enumerate(regions):
        # Create shifts
        nshift = r.nmul*r.nshift
        shift = 0.2*(np.arange(nshift) / (nshift - r.nmul*1.0)) - 0.1

        # Get the number of data points that the region should have
        nlambda = r.nlambda
        
        # Create the array containing the best shifts
        rshift = np.zeros(abund_count)
        rinten = []
        
        # Create an zeroed array of size (amount-of-abundencies,nshift) to store the chi squared for each abundance and for each shift
        rchisq = np.zeros((abund_count,nshift), dtype = np.float64)
        chisq = np.zeros(abund_count, dtype = np.float64)
        
        #
        inten_max = np.zeros(abund_count, dtype = np.float64)
        
        # Get the region of the atlas spectrum
        robs_wav = r.wav
        robs_inten = r.inten
        print("*** Region:", str(r))
#        print("    Region end:", r.lambda_end, "\n")
#        print("*** robs_wav:\n", robs_wav, "\n***\n")
#        print("*** robs_inten:\n", robs_inten, "\n***\n")
#        print("*** robe_wav\n", robs_wav, "\n***\n")

        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions.
        # IMPORTANT QUESTIONS:
        # 1. Where does 1.83 km/s come from?
        # 2. What's "tw"?
        # 3. What's the different
        tw = (np.arange(15)-7)*(robs_wav[1] - robs_wav[0])
#        print("*** tw:\n", tw, "\n***\n")
        
        # WHAT IS THIS AND WHAT DOES IT DO???
        #
        # THE LAST THING, 1.83*np.ceil(r.lambda_end)/300000.0, IS A GUESS THAT ORIGINATED FROM
        # THAT THE EXAMPLES USED 6302.0 INSTEAD OF np.ceil(r.lambda_end), AND THE LATTER WOULD
        # YIELD THE FIRST IF THE REGION IS THE SAME. BUT IT'S STILL JUST A GUESS!!! HAVE
        # NO IDEA IF IT'S CORRECT OR NOT!!! IS IT CORRECT?
        # (1.83 is some velocity in km/s related to convection and 300000.0 is the speed of
        # ligh in km/s)
#        psf = _gaussian(tw, [1.0, 0.0, 1.83*np.ceil(r.lambda_end)/300000.0])
        psf = _gaussian(tw, [1.0, 0.0, 1.83*6302.0/300000.0])
#        print("*** gaussian:\n", psf, "\n***\n")
        
        # For each abundence
        for a, spec in enumerate(synth_data):
            #
            spec = spec[0,0,0,:,0]
            inten_max[a] = spec.max()
            spec /= inten_max[a]
            
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rspec = r.get_contained(wav, spec, left_padding = 1e-9)
#            print("*** rwav:\n", rwav, "\n***\n")
            
            # Handle errors due to math with floating point numbers
            if len(rwav) != nlambda:
                print("*** DAJSD(dj!!! :::: len(rwav) - nlambda =", len(rwav) - nlambda, "\n")
#                print("*** rwav (len ", len(rwav), "):\n", rwav, "\n*** robs_wav: (len ", len(robs_wav), ")\n", robs_wav, "***\n", sep = "")
                rwav = rwav[:nlambda]
                rspec = rspec[:nlambda]

            # Convolve the spectra
            rspec = _convolve(rspec, psf / psf.sum())
#            print("*** rspec:\n", rspec, "\n***\n")
        
            # For each shift
            for ii in range(nshift):
                # Interpolate the synthetic spectrum and shift it a little
                # IMPORTANT QUESTION: Should I interoplate at "obs_wav" or at "wav"? Example does "wav" but that
                #                     might not align properly due to inconsistent spacing between the wavelength
                #                     data points in the atlas data.
                #                     HOWEVER doing so gives the wrong result... the "best" shift becomes too large. WHY?
                if interp_obs or r.interp_obs:
                    isyn = np.interp(robs_wav, rwav - shift[ii], rspec)
                else:
                    isyn = np.interp(rwav, rwav - shift[ii], rspec)
 #               print("*** isyn: (len ", len(isyn), ")\n", isyn, "\n***\n", sep = "")
                
                # Calculate and store the chi squared
                rchisq[a,ii] = ((robs_inten - isyn)**2).sum()
            
            # Get and store the best shift
            best_indx = np.argmin(rchisq[a,:])
            chisq[a] = rchisq[a,best_indx]
            ishift = shift[best_indx]
            rshift[a] = ishift
            
            # Calculate the shifted intensity spectrum (this is done using linear interpolation)
            sinten = np.interp(rwav, rwav - ishift, rspec)
            rinten.append(sinten)
        region_result.append(RegionResult(r, rwav, np.array(rinten), rshift, chisq, shift, rchisq, inten_max, abund_updates))
        
    return FittedSynthResult(region_result, region_data, wav, synth_data)

def fit_test(cfg_file, obs_wav, obs_inten, regions, abund_range, elem = "Fe", use_default_abund = True, verbose = False):
    result_good = fit_spectrum(cfg_file, obs_wav, obs_inten, regions, abund_range, elem = elem, use_default_abund = use_default_abund, bad_interp = False, verbose = verbose)
    result_bad = fit_spectrum(cfg_file, obs_wav, obs_inten, regions, abund_range, elem = elem, use_default_abund = use_default_abund, bad_interp = True, verbose = verbose)
    return result_good, result_bad

def synth_spectrum(cfg_file, regions, abund_range, verbose = False):
    """
    Synthesizes the spectrum in the given regions for a normal abundance as well as for the given abundance updates.
    """
    abund_updates = [[[elem, abund]] for abund in abund_range]
    _check_abund(abund_updates)

    # Setup the region data
    region_data = np.array(regions, dtype = "float64, float64, int32, float64")

    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Add the default
    abund_updates = [[["Fe", -4.50]]] + abund_updates

    # Synth the lines
#    synth_data = [_synth(s, m)]
    synth_data = []
    for a in abund_updates:
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))

    # Return the result
    return SynthResult(synth_data, s, m, abund_updates, region_data)

def synth_range(cfg_file, regions, elem, abund_range, verbose = False):
    """
    TODO: DOCUMENT THIS!!!
    """
    abund_updates = [[[elem, abund]] for abund in abund_range]
    return synth_spectrum(cfg_file, regions, abund_updates, verbose = verbose)

