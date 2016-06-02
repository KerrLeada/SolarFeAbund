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
import abundutils as au

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

class RegionResult(object):
    """
    Represents the result within a region.
    """

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
        
        i = np.argmin(chisq)
        self.best_shift = shift[i]
        self.best_chisq = chisq[i]
        self.best_abund = abund[i]

class SynthResult(object):
    """
    Encapsulates the result of a fitted specrum synthesis
    """
    
    def __init__(self, region_result, region_data, wav, raw_synth_data):
        self.region_result = region_result
        self.region_data = region_data
        self.wav = wav
        self.raw_synth_data = raw_synth_data

def _synth(s, m):
    """
    Helper function that synthazises a line
    """
    return s.synth(m.ltau, m.temp, m.pgas, m.vlos, m.vturb, m.B, m.inc, m.azi, False)

def _setup_regions(obs_wav, obs_inten, regions):
    # Setup the region data
    region_data = np.zeros(len(regions), dtype = "float64, float64, int32, float64")
    region_list = list(regions)
    for ri in range(len(regions)):
        if isinstance(regions[ri], regs.Region):
            region_data[ri] = regions[ri].to_tuple()
        else:
            try:
                region_data[ri] = regions[ri]
            except TypeError:
                err_msg = ("The given region was not in an acceptable tuple-like or Region form, and was of type " + type(region[ri]).__name__ +
                           ". Make sure the region has type Region or is a tuple-like object with 4 elements, of which the first, " +
                           "second and fourth is of type numpy.float64, and the third element has type numpy.int32. Normal floats and inte can also be used.")
                raise TypeError(err_msg)
            except ValueError:
                err_msg = ("The given tuple-like region was not in an acceptable form. It should have 4 elements of which the first, second " +
                           "and fourth are of type numpy.float64 while the third is of type numpy.int32. Normal floats and inte can also be used.")
                raise ValueError(err_msg)
            region_list[ri] = regs.new_region(obs_wav, obs_inten, *regions[ri])
    return region_list, region_data

def _fit_regions(regions, wav, synth_data, abund_updates, interp_obs):
    # Fit the data
    abund_count = len(abund_updates)
    
    # A list of the chi sqaured values for each region
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
        psf = _gaussian(tw, [1.0, 0.0, 1.83*np.ceil(r.lambda_end)/300000.0])
#        psf = _gaussian(tw, [1.0, 0.0, 1.83*6302.0/300000.0])
#        print("*** gaussian:\n", psf, "\n***\n")
        
        # For each abundence
        rwav_all = []
        rspec_all = []
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
        
            # Store for later calculations
            rwav_all.append(rwav)
            rspec_all.append(rspec)
        
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
#            best_indx = np.argmin(rchisq[a,:])
#            chisq[a] = rchisq[a,best_indx]
            ishift = shift[np.argmin(rchisq[a,:])]
            rshift[a] = ishift
            
            # Calculate the shifted intensity spectrum (this is done using linear interpolation)
            sinten = np.interp(rwav, rwav - ishift, rspec)
            rinten.append(sinten)

        # Calculate the chi squared for each abundence, for the best shift
        for a, (rwav, rspec) in enumerate(zip(rwav_all, rspec_all)):
            if interp_obs or r.interp_obs:
                isyn = np.interp(robs_wav, rwav - rshift[a], rspec)
            else:
                isyn = np.interp(rwav, rwav - rshift[a], rspec)
            chisq[a] = ((robs_inten - isyn)**2).sum()
        
        #
        region_result.append(RegionResult(r, rwav, np.array(rinten), rshift, chisq, shift, rchisq, inten_max, abund_updates))
    return region_result

def fit_spectrum(cfg_file, obs_wav, obs_inten, regions, abund_range, elem = "Fe", use_default_abund = True, interp_obs = False, verbose = False):
    """
    This files synthesizes a spectrum and attempts to fit it to the given observed spectrum.
    """
    
    # Create the updates and check them
    abund_updates = [au.abund(elem, a) for a in abund_range]
    au.check_abund(abund_updates)

    # Copy the region list and setup an array with the region data
    regions, region_data = _setup_regions(obs_wav, obs_inten, regions)

    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Generate the synthetic lines
    synth_data = []
    if use_default_abund:
        abund_updates = au.empty_abund + abund_updates
    for a in abund_updates:
        # Update the abundence and synthasize a new spectrum
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))

    # Get the wavelengths
    wav = s.getwav()
    
    # Fit the regions (kind of... technically this determines how to shift the regions and how well everything then fits)
    region_result = _fit_regions(regions, wav, synth_data, abund_updates, interp_obs)
    
    # Return the result
    return SynthResult(region_result, region_data, wav, synth_data)
    
class _FitState(object):
    def __init__(self, reg, s, m, iteration_limit, interp_obs, verbose):
        # Store the region, synth and model objects
        self.reg = reg
        self.s = s
        self.m = m
        self.iteration_limit = iteration_limit
        
        # Create shifts
        self.nshift = reg.nmul*reg.nshift
        self.shift = 0.2*(np.arange(self.nshift) / (self.nshift - reg.nmul*1.0)) - 0.1
        
        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions.
        # IMPORTANT QUESTIONS:
        # 1. Where does 1.83 km/s come from?
        # 2. What's "tw"?
        # 3. What's the different
        self.tw = (np.arange(15)-7)*(reg.wav[1] - reg.wav[0])
    #        print("*** tw:\n", tw, "\n***\n")
        
        # WHAT IS THIS AND WHAT DOES IT DO???
        #
        # THE LAST THING, 1.83*np.ceil(r.lambda_end)/300000.0, IS A GUESS THAT ORIGINATED FROM
        # THAT THE EXAMPLES USED 6302.0 INSTEAD OF np.ceil(r.lambda_end), AND THE LATTER WOULD
        # YIELD THE FIRST IF THE REGION IS THE SAME. BUT IT'S STILL JUST A GUESS!!! HAVE
        # NO IDEA IF IT'S CORRECT OR NOT!!! IS IT CORREdCT?
        # (1.83 is some velocity in km/s related to convection and 300000.0 is the speed of
        # ligh in km/s)
        self.psf = _gaussian(self.tw, [1.0, 0.0, 1.83*np.ceil(reg.lambda_end)/300000.0])
        
        # Initialize some lists
        self.rchisq = []
        self.chisq = []
        self.rshift = []
        self.rinten = []
        self.inten_max = []
        
        # Store extra options
        self.interp_obs = interp_obs
        self.verbose = verbose
        
    def _synth_abund(self, abund):
        # Update the abundence and synth the line
        self.s.updateABUND(abund, verbose = self.verbose)
        spec = _synth(self.s, self.m)[0,0,0,:,0]
        inten_max = spec.max()
        spec /= inten_max
        self.inten_max.append(inten_max)
        
        # Get the wavelengths
        wav = self.s.getwav()
        
        # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
        rwav, rspec = self.reg.get_contained(wav, spec, left_padding = 1e-9)
        
        # Handle errors due to math with floating point numbers
        if len(rwav) != self.reg.nlambda:
            print("*** DAJSD(dj!!! :::: len(rwav) - nlambda =", len(rwav) - self.reg.nlambda, "\n")
    #                print("*** rwav (len ", len(rwav), "):\n", rwav, "\n*** robs_wav: (len ", len(robs_wav), ")\n", robs_wav, "***\n", sep = "")
            rwav = rwav[:self.reg.nlambda]
            rspec = rspec[:self.reg.nlambda]

        # Convolve the spectra
        rspec = _convolve(rspec, self.psf / self.psf.sum())
    #            print("*** rspec:\n", rspec, "\n***\n")
        
        # Get the region of the atlas spectrum
        robs_wav = self.reg.wav
        robs_inten = self.reg.inten

        # For each shift
        rchisq = np.zeros(len(self.shift), dtype = np.float64)
        for ii in range(len(self.shift)):
            # Interpolate the synthetic spectrum and shift it a little
            # IMPORTANT QUESTION: Should I interoplate at "obs_wav" or at "wav"? Example does "wav" but that
            #                     might not align properly due to inconsistent spacing between the wavelength
            #                     data points in the atlas data.
            #                     HOWEVER doing so gives the wrong result... the "best" shift becomes too large. WHY?
            if self.interp_obs or self.reg.interp_obs:
                isyn = np.interp(robs_wav, rwav - self.shift[ii], rspec)
            else:
                isyn = np.interp(rwav, rwav - self.shift[ii], rspec)
    #               print("*** isyn: (len ", len(isyn), ")\n", isyn, "\n***\n", sep = "")
            
            # Calculate and store the chi squared
            rchisq[ii] = ((robs_inten - isyn)**2).sum()
        
        # Store all the chi squared values
        self.rchisq.append(rchisq)
        
        # Get and store the best shift
        best_indx = np.argmin(rchisq)
        self.chisq.append(rchisq[best_indx])
        ishift = self.shift[best_indx]
        self.rshift.append(ishift)
        
        # Calculate the shifted intensity spectrum (this is done using linear interpolation)
        sinten = np.interp(rwav, rwav - ishift, rspec)
        self.rinten.append(sinten)
        
        return rchisq[best_indx]

    def fit(self, initial_abunds, elem, das):
        
        # Create the array containing the best shifts
        rshift = []
        rinten = []
        
        # Create an zeroed array of size (amount-of-abundencies,nshift) to store the chi squared for each abundance and for each shift
        rchisq = []
        chisq = []
        
        # Get the region of the atlas spectrum
        robs_wav = self.reg.wav
        robs_inten = self.reg.inten
        
        # Get the initial abundencies
        a1 = min(initial_abunds)
        a2 = max(initial_abunds)
        
        # For each abundence
        chisq1 = self._synth_abund(au.abund(elem, a1))
        chisq2 = self._synth_abund(au.abund(elem, a2))
        
        # Get the direction of the gradient
        graddir = np.sign(chisq1 - chisq2)
        if graddir == 0:
            raise Exception("???")
        
        #

    #    print("chisq1:", chisq1, "\nchisq2:", chisq2, "\ngraddir*(chisq1 - chisq2):", graddir*(chisq1 - chisq2), "\n")
        found_it = False
        prev = (a1, a2)
    #    print("***** All da:", das)
        for da in das:
            a1, a2 = prev
            print("**** NEW a1:", a1, "   a2:", a2, "****")
    #        print("\n*** da:", da, "***\n")
    #        print("a1:", a1, "\na2:", a2, "\n")
            for iiiii in range(self.iteration_limit):
                # Save the previous set of abundencies and move on to the next ones
                prev = (a1, a2)
                if graddir == -1:
                    a1, a2 = a1 - da, a1
                else:
                    a1, a2 = a2, a2 + da
                
                # Calculat chi squared for the two abundencies
                chisq1 = self._synth_abund(au.abund(elem, a1))
                chisq2 = self._synth_abund(au.abund(elem, a2))
    #            print("chisq1:", chisq1, "\nchisq2:", chisq2, "\ngraddir*(chisq1 - chisq2):", graddir*(chisq1 - chisq2), "\n")
    #            print("a1:", a1, "\na2:", a2, "\n")

                # If there was a sign change, roll back to the previous a1 and a2 move on to the next da
                if graddir*(chisq1 - chisq2) < 0:
                    print("**** OLD a1:", a1, "   a2:", a2, "****")
                    found_it = True
                    break

        print("*** graddir:", graddir)
        
        if not found_it:
            print("!!!!!!!!!!!!!!!!!!!!!!!! Found no minimum !!!!!!!!!!!!!!!!!!!!!!!!")

        best_chisq = min(chisq1, chisq2)
        best_a = a1 if chisq1 < chisq2 else a2
        return best_chisq, best_a

def fit_spectrum2(cfg_file, obs_wav, obs_inten, regions, abunds, da, elem = "Fe", interation_limit = 50, interp_obs = False, verbose = False):
    """
    NEED TO IMPLEMENT THIS PROBLERLY!!! RIGHT NOW IT'S BUGGED!!!
    """
    
    # Make sure all the regions are instances of Region
    regions = [regs.new_region_from(r, obs_wav, obs_inten) for r in regions]
    region_data = np.array([r.to_tuple() for r in regions], dtype = "float64, float64, int32, float64")
    
    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Make sure da is a list of changes to the abundance
    if np.isscalar(da):
        da = [da]
    else:
        da = list(da)

    # Loop through each region
    result = []
    for r, a in zip(regions, abunds):
        fs = _FitState(r, s, m, iteration_limit, interp_obs, verbose)
        result.append((fs.fit(a, elem, da), fs))
    return result
