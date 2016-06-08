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
import multiprocessing as mp
import astropy.units

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

class ChiRegionResult(object):
    """
    Represents the result within a region.
    """

    def __init__(self, region, wav, inten, shift, chisq, shift_all, chisq_all, inten_scale_factor, abund):
        self.region = region
        self.wav = wav
        self.inten = inten
        self.inten_scale_factor = inten_scale_factor
        self.shift = shift
        self.chisq = chisq
        self.shift_all = shift_all
        self.chisq_all = chisq_all
        self.abund = abund
        
        best = np.argmin(chisq)
        self.best_index = best
        self.best_shift = shift[best]
        self.best_chisq = chisq[best]
        self.best_abund = abund[best]
    
    def _fuse_result(self, other):
        # Make sure the regions are the same
        if self.region != other.region:
            print(self.region)
            print(other.region)
            raise Exception("Invalid region")

        # Concatenate the numpy arrays, with the data from this result first
        inten = np.concatenate((self.inten, other.inten), axis = 0)
        shift = np.concatenate((self.shift, other.shift), axis = 0)
        chisq = np.concatenate((self.chisq, other.chisq), axis = 0)
        chisq_all = np.concatenate((self.chisq_all, other.chisq_all), axis = 0)
        inten_scale_factor = np.concatenate((self.inten_scale_factor, other.inten_scale_factor), axis = 0)
        
        # Concatenate the abundencies. Note that these are not numpy arrays but ordinary python lists,
        # so self.abund + other.abund means concatenation and not elementwise addition.
        abund = self.abund + other.abund
        
        # Return the fused result
        return ChiRegionResult(self.region, self.wav, inten, shift, chisq, self.shift_all, chisq_all, inten_scale_factor, abund)

class EWRegionResult(object):
    """
    Encapsulates the result of a fit using equivalent width
    """
    def __init__(self, region, wav, inten, inten_scale_factor, obs_eq_width, eq_width, diff, abund):
        # Store the data
        self.region = region
        self.wav = wav
        self.inten = inten
        self.inten_scale_factor = inten_scale_factor
        self.obs_eq_width = obs_eq_width
        self.eq_width = eq_width
        self.diff = diff
        self.abund = abund
        
        # Get the best values
        best = np.argmin(abs(diff))
        self.best_index = best
        self.best_inten = inten[best,:]
        self.best_inten_scale_factor = inten_scale_factor[best]
        self.best_eq_width = eq_width[best]
        self.best_diff = diff[best]
        self.best_abund = abund[best]

    def _fuse_result(self, other):
        # Make sure the regions are the same
        if self.region != other.region:
            print(self.region)
            print(other.region)
            raise Exception("Invalid region")

        # Concatenate the numpy arrays, with the data from this result first
        inten = np.concatenate((self.inten, other.inten), axis = 0)
        inten_scale_factor = np.concatenate((self.inten_scale_factor, other.inten_scale_factor), axis = 0)
        eq_width = np.concatenate((self.eq_width, other.eq_width), axis = 0)
        diff = np.concatenate((self.diff, other.diff), axis = 0)
        abund = np.concatenate((self.abund, other.abund), axis = 0)
        
        # Return a new region result
        return EWRegionResult(self.region, self.wav, inten, inten_scale_factor, self.obs_eq_width, eq_width, diff, abund)

class SynthResult(object):
    """
    Encapsulates the result of a fitted specrum synthesis, using chi squared.
    """
    
    def __init__(self, region_result, region_data, wav, raw_synth_data):
        self.region_result = region_result
        self.region_data = region_data
        self.wav = wav
        self.raw_synth_data = raw_synth_data
    
def _fuse_chi_result(result1, result2):
    """
    Fuses two results of a fit, assuming it was done with the chi squared method
    """
    
    if not np.array_equal(result1.region_data, result2.region_data):
        raise Exception("Region data must be the same for result1 and result2")
    if not np.array_equal(result1.wav, result2.wav):
        raise Exception("Wavelength data must be the same for result1 and result2")
    
    # Fuse the region results
    region_result = []
    for r1, r2 in zip(result1.region_result, result2.region_result):
        region_result.append(r1._fuse_result(r2))
    if len(result1.region_result) > len(result2.region_result):
        region_result.extend(result1.region_result[len(result2.region_result):])
    elif len(result1.region_result) < len(result2.region_result):
        region_result.extend(result2.region_result[len(result1.region_result):])

    # Append the raw data of result2 after result1 (note that since raw_synth_data are
    # python lists + means concatenation and not elementwise addition, as would be the
    # case if they where arrays)
    raw_synth_data = result1.raw_synth_data + result2.raw_synth_data
    
    # Return the fused result
    return SynthResult(region_result, result1.region_data, result1.wav, raw_synth_data)

def _synth(s, m):
    """
    Helper function that synthazises a line
    """
    return s.synth(m.ltau, m.temp, m.pgas, m.vlos, m.vturb, m.B, m.inc, m.azi, False)

def _setup_regions(atlas, regions):
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
            region_list[ri] = regs.new_region(atlas, *regions[ri])
    return region_list, region_data

def _fit_regions(regions, wav, synth_data, abund_updates):
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

        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions.
        # IMPORTANT QUESTIONS:
        # 1. Where does 1.83 km/s come from?
        # 2. What's "tw"?
        # 3. What's the different
        tw = (np.arange(15)-7)*(robs_wav[1] - robs_wav[0])
        
        # WHAT IS THIS AND WHAT DOES IT DO???
        #
        # THE LAST THING, 1.83*np.ceil(r.lambda_end)/300000.0, IS A GUESS THAT ORIGINATED FROM
        # THAT THE EXAMPLES USED 6302.0 INSTEAD OF np.ceil(r.lambda_end), AND THE LATTER WOULD
        # YIELD THE FIRST IF THE REGION IS THE SAME. BUT IT'S STILL JUST A GUESS!!! HAVE
        # NO IDEA IF IT'S CORRECT OR NOT!!! IS IT CORRECT?
        # (1.83 is some velocity in km/s related to convection and 300000.0 is the speed of
        # ligh in km/s)
        psf = _gaussian(tw, [1.0, 0.0, 1.83*r.lambda_end/300000.0])
        
        # For each abundence
        rwav_all = []
        rspec_all = []
        for a, spec in enumerate(synth_data):
            #
            spec = spec[0,0,0,:,0]
            inten_max[a] = spec.max()
            
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rspec = r.get_contained(wav, spec, left_padding = 1e-9)
#            print("*** rwav:\n", rwav, "\n***\n")
            
            # Handle errors due to math with floating point numbers and things like that
            if len(rwav) != nlambda:
#                print("*** DAJSD(dj!!! :::: len(rwav) - nlambda =", len(rwav) - nlambda, "\n")
#                print("*** rwav (len ", len(rwav), "):\n", rwav, "\n*** robs_wav: (len ", len(robs_wav), ")\n", robs_wav, "***\n", sep = "")
                rwav = rwav[:nlambda]
                rspec = rspec[:nlambda]

            # Convolve the spectra
#            print("*** rspec max (before):", rspec.max(), "***\n")
            rspec = _convolve(rspec, psf / psf.sum())
            rspec /= rspec.max()
#            print("*** rspec max (after): ", rspec.max(), "***\n")
        
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
                isyn = np.interp(rwav, rwav - shift[ii], rspec)
                
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
            isyn = np.interp(rwav, rwav - rshift[a], rspec)
            chisq[a] = ((robs_inten - isyn)**2).sum()
        
        #
        region_result.append(ChiRegionResult(r, rwav, np.array(rinten), rshift, chisq, shift, rchisq, inten_max, abund_updates))
    return region_result

def fit_spectrum(cfg_file, atlas, regions, abund_range, elem = "Fe", use_default_abund = True, verbose = False):
    """
    This files synthesizes a spectrum and attempts to fit it to the given observed spectrum.
    """
    
    # Create the updates and check them
    abund_updates = [au.abund(elem, a) for a in abund_range]
    au.check_abund(abund_updates)

    # Copy the region list and setup an array with the region data
    regions, region_data = _setup_regions(atlas, regions)

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
    region_result = _fit_regions(regions, wav, synth_data, abund_updates)
    
    # Return the result
    return SynthResult(region_result, region_data, wav, synth_data)

def _parallel_chi(conn, cfg_file, atlas, regions, abunds, elem, use_default_abund, verbose):
    try:
        result = fit_spectrum(cfg_file, atlas, regions, abunds, elem, use_default_abund, verbose)
        conn.send(result)
    except:
        conn.send(None)
        raise
    finally:
        conn.close()

def fit_spectrum_para(cfg_file, atlas, regions, abund_range, processes = 2, elem = "Fe", use_default_abund = True, verbose = False):
    # Split up the abundencies between processes
    abund_range = list(abund_range)
    abunds = [[] for _ in range(processes)]
    abunds_per_process = int(np.ceil(float(len(abund_range)) / float(processes)))
    for i in range(processes):
        si = i*abunds_per_process
        ei = (i + 1)*abunds_per_process
        abunds[i].extend(abund_range[si:ei])

    # Spawn the processes
    proc_list = []
    conns = []
    for a in abunds:
        rec, conn = mp.Pipe()
        p = mp.Process(target = _parallel_chi, args = (conn, cfg_file, atlas, regions, a, elem, use_default_abund, verbose))
        p.start()
        proc_list.append(p)
        conns.append(rec)
    
    # Join the processes
    result_list = []
    for rec, p in zip(conns, proc_list):
        result_list.append(rec.recv())
        rec.close()
        p.join()
    
    # Fuse the result together
    result = result_list[0]
    for r in result_list[1:]:
        result = _fuse_chi_result(result, r)
    
    # Return the fused result    
    return result
    
class _FitState(object):
    def __init__(self, reg, s, m, abund_limits, verbose):
    
        if len(abund_limits) != 2:
            raise Exception("The abundency limits are given in the form of a tuple-like object with 2 elements, but the given limits had " + str(len(abund_limits)) + " elements")
        
        # Store the region, synth and model objects
        self.reg = reg
        self.s = s
        self.m = m
        self.abund_limits = abund_limits
        
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
        self.psf = _gaussian(self.tw, [1.0, 0.0, 1.83*reg.lambda_end/300000.0])
        
        # Initialize some lists
        self.rchisq = []
        self.chisq = []
        self.rshift = []
        self.rinten = []
        self.inten_max = []
        
        # Store extra options
        self.verbose = verbose
        
    def _synth_abund(self, abund):
        # Update the abundence and synth the line
        self.s.updateABUND(abund, verbose = self.verbose)
        spec = _synth(self.s, self.m)[0,0,0,:,0]
#        inten_max = spec.max()
#        spec /= inten_max
#        self.inten_max.append(inten_max)
        
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
        self.inten_max.append(rspec.max())
        rspec /= self.inten_max[-1]
        
        # Get the region of the atlas spectrum
        robs_wav = self.reg.wav
        robs_inten = self.reg.inten

        # For each shift
        rchisq = np.zeros(len(self.shift), dtype = np.float64)
        for ii in range(len(self.shift)):
            # Interpolate the synthetic spectrum and shift it a little
            isyn = np.interp(rwav, rwav - self.shift[ii], rspec)
            
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
        
        print("!!!!!!!!! FITTING REGION:", self.reg, " !!!!!!!!!")
        
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
        
        # Get the limits
        a1_limit = min(self.abund_limits)
        a2_limit = max(self.abund_limits)
        
        # For each abundence
        chisq1 = self._synth_abund(au.abund(elem, a1))
        chisq2 = self._synth_abund(au.abund(elem, a2))
        
        # Get the direction of the gradient
        graddir = np.sign(chisq1 - chisq2)
        if graddir == 0:
            raise Exception("Invalid starting abundencies")
 #       print("******* graddir =", graddir, " *******")
        
        #

    #    print("chisq1:", chisq1, "\nchisq2:", chisq2, "\ngraddir*(chisq1 - chisq2):", graddir*(chisq1 - chisq2), "\n")
        found_minimum = False
        prev = (a1, a2)
    #    print("***** All da:", das)
        for da in das:
            a1, a2 = prev
 #           print("\n**** NEW a1:", a1, "   a2:", a2, "   da:", da, "****")
            while a1 >= a1_limit or a2 <= a2_limit:
                # Save the previous set of abundencies and move on to the next ones
                prev = (a1, a2)
                if graddir == -1:
                    a1, a2 = a1 - da, a1
                    chisq2 = chisq1
                    chisq1 = self._synth_abund(au.abund(elem, a1))
                else:
                    a1, a2 = a2, a2 + da
                    chisq1 = chisq2
                    chisq2 = self._synth_abund(au.abund(elem, a2))
                
                # If there was a sign change, roll back to the previous a1 and a2 move on to the next da
#                if graddir*(chisq1 - chisq2) < 0:
#                print("a1:", a1, "    a2:", a2, "    chisq1:", chisq1, "    chisq2:", chisq2, "    graddir:", graddir, "    dir:", np.sign(chisq1 - chisq2), "    graddir==dir:", graddir == np.sign(chisq1 - chisq2))
                if graddir != np.sign(chisq1 - chisq2):
#                    print("**** OLD a1:", a1, "   a2:", a2, "****")
                    found_minimum = True
                    break

 #       print("*** graddir:", graddir)
        
        if not found_minimum:
            print("!!!!!!!!!!!!!!!!!!!!!!!! Found no minimum !!!!!!!!!!!!!!!!!!!!!!!!")

        best_chisq = min(chisq1, chisq2)
        best_a = a1 if chisq1 < chisq2 else a2
        return best_chisq, best_a

def fit_spectrum_seeking(cfg_file, atlas, regions, abunds, da, elem = "Fe", abund_limits = (-5.0, -4.0), verbose = False):
    """
    NEED TO IMPLEMENT THIS PROBLERLY!!! RIGHT NOW IT'S BUGGED!!!
    """
    
    # Make sure all the regions are instances of Region
    regions = [regs.new_region_from(atlas, r) for r in regions]
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
        fs = _FitState(r, s, m, abund_limits, verbose)
        result.append((fs.fit(a, elem, da), fs))
    return result

#def _equivalent_width(wav, inten, dlambda = None):
#    # The continuum level should be the maximum intensity
#    cont = inten.max()
#
#    # Calculate the area
#    if dlambda == None:
#        # If no dlambda was given, assume an uneven grid
#        area = 0
#        for a, b, ia, ib in zip(wav[:-1], wav[1:], inten[:-1], inten[1:]):
#            # Since we want the area of a line, we been to subtract the continuum level
#            ia -= cont
#            ib -= cont
#            
#            # Add the contribution of this part to the area
#            area += (b - a)*(ia + ib)/2
#    else:
#        # If dlambda was given, use trapz from numpy instead to calculate the area
#        area = np.trapz(inten - cont, x = wav, dx = dlambda)
#
#    # If ew is the equivalent width, we have that: cont*ew = area
#    # As such the equivalent width is given by ew = area/cont
#    return abs(area/cont)

def _equivalent_width(wav, inten, dlambda = None):
    # The continuum level should be the maximum intensity
    cont = inten.max()

    # Calculate the area
    if dlambda == None:
        # If no dlambda was given, assume an uneven grid
        area = 0
        for a, b, ia, ib in zip(wav[:-1], wav[1:], inten[:-1], inten[1:]):
            # Add the contribution of this part to the area
            area += (b - a)*(ia + ib)/2
    else:
        # If dlambda was given, use trapz from numpy instead to calculate the area
        area = np.trapz(inten, x = wav, dx = dlambda)
    
    # If the area under the spectrum curve from wav[0] to wav[-1] is area, then the
    # area of the line is: area_line = total_area - area
    # where total_area is: total_area = continuum*(wav[-1] - wav[0])
    # assuming the continuum is constant in the interval (or close enough to constant
    # that it's not a too rough approximation to treat it as constant).
    area_line = cont*(wav[-1] - wav[0]) - area

    # If ew is the equivalent width, we have that: cont*ew = area_line
    # As such the equivalent width is given by ew = area_line/cont
    return abs(area_line/cont)

def fit_width(cfg_file, atlas, regions, abund_range, refinements = None, eq_width_unit = astropy.units.pm, elem = "Fe", use_default_abund = True, verbose = False):
    """
    DOCUMENT THIS!!!
    (This function synthezises spectrums for different abundencies and regions and returns which ones "fit" the
     best, using equivalent widths)
    """
    
    # If no refinements where given, set them to an empty list
    if refinements == None:
        refinements = []
    
    # Create the abundancy updates and check them
    abund_updates = [au.abund(elem, a) for a in abund_range]
    au.check_abund(abund_updates)
    
    # Setup the regions
    regions, region_data = _setup_regions(atlas, regions)
    
    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Synth the spectrum
    synth_data = []
    if use_default_abund:
        abund_updates = au.empty_abund + abund_updates
    for a in abund_updates:
        s.updateABUND(a, verbose = verbose)
        synth_data.append(_synth(s, m))

    # Get the spectrums for the abundencies and the wavelength
    spec = [sd[0,0,0,:,0] for sd in synth_data]
    wav = s.getwav()
    
    # "Fit" the data in each region using equivalent widths
    result = []
    for ri, r in enumerate(regions):
        inten_max = np.zeros(len(abund_updates), dtype = np.float64)
        eq_width = np.zeros(len(abund_updates), dtype = np.float64)
        diff = np.zeros(len(abund_updates), dtype = np.float64)
        sinten = np.zeros((len(abund_updates),r.nlambda), dtype = np.float64)
        
        # Get the region of the atlas spectrum
        robs_wav = r.wav
        robs_inten = r.inten

        # Create the Gaussian for an about 1.83 km/s velocity. This is done to recreate line broadening
        # due to convective motions.
        tw = (np.arange(15)-7)*(robs_wav[1] - robs_wav[0])
        psf = _gaussian(tw, [1.0, 0.0, 1.83*r.lambda_end/300000.0])
        
        # Calculate the equivalent width of the observed data
        if r in refinements:
            robs_wav, robs_inten = regs.shrink(refinements[r][0], refinements[r][1], robs_wav, robs_inten)
        obs_ew = (_equivalent_width(robs_wav, robs_inten) * astropy.units.AA).to(eq_width_unit).value
        
        for ai, a in enumerate(abund_updates):
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rspec = r.get_contained(wav, spec[ai], left_padding = 1e-9)
            
            # Handle errors due to math with floating point numbers
            if len(rwav) != r.nlambda:
                print("*** DAJSD(dj!!! :::: len(rwav) - nlambda =", len(rwav) - r.nlambda, "\n")
                rwav = rwav[:r.nlambda]
                rspec = rspec[:r.nlambda]

            # Convolve the spectra
            rspec = _convolve(rspec, psf / psf.sum())
            inten_max[ai] = rspec.max()
            rspec /= inten_max[ai]
            
            # Calculate the equivalent width
            eq_width[ai] = (_equivalent_width(rwav, rspec, dlambda = r.dlambda) * astropy.units.AA).to(eq_width_unit).value
            diff[ai] = eq_width[ai] - obs_ew
            sinten[ai,:] = rspec
        result.append(EWRegionResult(r, rwav, sinten, inten_max, obs_ew, eq_width, diff, abund_updates))
    return SynthResult(result, region_data, wav, synth_data)

def _parallel_width(conn, cfg_file, atlas, regions, abunds, refinements, eq_width_unit, elem, use_default_abund, verbose):
    try:
        result = fit_width(cfg_file, atlas, regions, abunds, refinements, eq_width_unit, elem, use_default_abund, verbose)
        conn.send(result)
    except:
        conn.send(None)
        raise
    finally:
        conn.close()

def fit_width_para(cfg_file, atlas, regions, abund_range, processes = 2, refinements = None, eq_width_unit = astropy.units.pm, elem = "Fe", use_default_abund = True, verbose = False):
    # Split up the abundencies between processes
    abund_range = list(abund_range)
    abunds = [[] for _ in range(processes)]
    abunds_per_process = int(np.ceil(float(len(abund_range)) / float(processes)))
    for i in range(processes):
        si = i*abunds_per_process
        ei = (i + 1)*abunds_per_process
        abunds[i].extend(abund_range[si:ei])

    # Spawn the processes
    proc_list = []
    conns = []
    for a in abunds:
        rec, conn = mp.Pipe()
        p = mp.Process(target = _parallel_width, args = (conn, cfg_file, atlas, regions, a, refinements, eq_width_unit, elem, use_default_abund, verbose))
        p.start()
        proc_list.append(p)
        conns.append(rec)
    
    # Join the processes
    result_list = []
    for rec, p in zip(conns, proc_list):
        result_list.append(rec.recv())
        rec.close()
        p.join()
    
    # Fuse the result together
    result = result_list[0]
    for r in result_list[1:]:
        result = _fuse_chi_result(result, r)
    
    # Return the fused result    
    return result
