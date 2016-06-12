"""
This module contains the functions and classes that synthezises spectra. Specifically
a single function and class.
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

_MODEL_FILE = "data/falc_filled.nc"
_ELEMENT = "Fe"

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
    def __init__(self, region, wav, inten, inten_scale_factor, obs_eq_width, eq_width, diff, abund, eq_width_unit):
#        print("*** ABUNDS (EWRegionResult) ***")
#        print(abund)
#        print("*******************************")
        # Store the data
        self.region = region
        self.wav = wav
        self.inten = inten
        self.inten_scale_factor = inten_scale_factor
        self.obs_eq_width = obs_eq_width
        self.eq_width = eq_width
        self.diff = diff
        self.abund = abund
        self.eq_width_unit = eq_width_unit
        
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

        # Conversion factor, makes sure the unit of equivalent width of this object always wins over the other objects unit
        conv_factor = (1 * other.eq_width_unit).to(self.eq_width_unit).value

        # Concatenate the numpy arrays, with the data from this result first
        inten = np.concatenate((self.inten, other.inten), axis = 0)
        inten_scale_factor = np.concatenate((self.inten_scale_factor, other.inten_scale_factor), axis = 0)
        eq_width = np.concatenate((self.eq_width, other.eq_width*conv_factor), axis = 0)
        diff = np.concatenate((self.diff, other.diff*conv_factor), axis = 0)

        # Concatenate the abundencies. Note that these are not numpy arrays but ordinary python lists,
        # so self.abund + other.abund means concatenation and not elementwise addition.
        abund = self.abund + other.abund
        
        # Return a new region result
        return EWRegionResult(self.region, self.wav, inten, inten_scale_factor, self.obs_eq_width, eq_width, diff, abund, self.eq_width_unit)

class SynthResult(object):
    """
    Encapsulates the result of a fitted specrum synthesis, using chi squared.
    """
    
    def __init__(self, region_result, region_data, wav, raw_synth_data):
        self.region_result = region_result
        self.region_data = region_data
        self.wav = wav
        self.raw_synth_data = raw_synth_data
    
def _fuse_result2(result1, result2):
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

def _fuse_result(result_list):
    """
    Fuses a list of results together.
    """
    
    # Fuse the result together
    result = result_list[0]
    for r in result_list[1:]:
        result = _fuse_result2(result, r)
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

#def _setup_regions(atlas, regions):
#    # Setup the region data
#    region_data = np.zeros(len(regions), dtype = "float64, float64, int32, float64")
#    region_list = list(regions)
#    for ri in range(len(regions)):
#        if isinstance(regions[ri], regs.Region):
#            region_data[ri] = regions[ri].to_tuple()
#        else:
#            try:
#                region_data[ri] = regions[ri]
#            except TypeError:
#                err_msg = ("The given region was not in an acceptable tuple-like or Region form, and was of type " + type(region[ri]).__name__ +
#                           ". Make sure the region has type Region or is a tuple-like object with 4 elements, of which the first, " +
#                           "second and fourth is of type numpy.float64, and the third element has type numpy.int32. Normal floats and inte can also be used.")
#                raise TypeError(err_msg)
#            except ValueError:
#                err_msg = ("The given tuple-like region was not in an acceptable form. It should have 4 elements of which the first, second " +
#                           "and fourth are of type numpy.float64 while the third is of type numpy.int32. Normal floats and inte can also be used.")
#                raise ValueError(err_msg)
#            region_list[ri] = regs.new_region(atlas, *regions[ri])
#    return region_list, region_data

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

def _parallel_calc(abund_range, processes, func, args):
    """
    The function distributes abundance calculations over processes, take their results, fuses them together
    and returns that fused result. The required arguments are
    
        abund_range : The iron aundencies to distribute.
        
        processes   : The amount of processes to use.
        
        func        : The function that performs the calculations. Note that the function should take the abundance as first argument.
        
        args        : The arguments to the function, abundance excluded.

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

def _fit_regions(regions, wav, synth_data, abund_updates):
    """
    DOCUMENT THIS!!!
    """
    
    # Fit the data
    abund_count = len(abund_updates)
    
    # A list of the chi sqaured values for each region
    region_result = []
    
    # Create shifts
    nshift = 101
    shift = 0.2*(np.arange(nshift) / (nshift - 1.0)) - 0.1

    # For each region
    for ri, r in enumerate(regions):
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
        reduced_psf = psf / psf.sum()
        
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
            rspec = _convolve(rspec, reduced_psf)
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

def _fit_spectrum(abund_range, cfg_file, regions, use_default_abund, verbose):
    """
    DOCUMENT THIS!!!
    """
    
    # Create the updates and check them
    abund_updates = [au.abund(_ELEMENT, a) for a in abund_range]
    
    # Copy the region list and setup an array with the region data
    region_data = _setup_region_data(regions)

    # Init LTE class
    s = p.pyLTE(cfg_file, region_data, nthreads = 1, solver = 0)

    # Read a model
    m = sp.model(_MODEL_FILE)
    
    # Generate the synthetic lines
    synth_data = []
    if use_default_abund:
        abund_updates = au.EMPTY_ABUND + abund_updates
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

def fit_spectrum(cfg_file, atlas, regions, abund_range, use_default_abund = False, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are

        use_default_abund : Determines if the default iron abundance should be used first.
                            Default is False.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.
    
    Returns a SynthResult object containing the result of all calculations.

    Fitting is done by using chi squared to compare the observed and synthetic spectrum for the different regions. Essentially, if we have
    a region, then for each abundance the chi squared of the observed and synthetic spectrum is calculated. The abundance with smallest
    chi squared is then taken as the best value.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, so be sure the correct one is used.
    """
    
    regions = _setup_regions(atlas, regions)
    return _fit_spectrum(abund_range, cfg_file, regions, use_default_abund, verbose)

def fit_spectrum_parallel(cfg_file, atlas, regions, abund_range, processes = 2, use_default_abund = False, verbose = False):
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

        use_default_abund : Determines if the default iron abundance should be used first.
                            Default is False.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object containing the result of all calculations.

    Fitting is done by using chi squared to compare the observed and synthetic spectrum for the different regions. Essentially, if we have
    a region, then for each abundance the chi squared of the observed and synthetic spectrum is calculated. The abundance with smallest
    chi squared is then taken as the best value.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, so be sure the correct one is used.
    
    The distribution of work over the processes is done by making the processes handle different abundancies. For example, if we have R
    different regions, A different abundancies and N processes are used, then each process works with R different regions and approximately
    A / N different abundancies. When R and A are small the overhead of using multiple processes and coordinating them can make this slower
    then just calling fit_spectrum. However, when A and R increases this parallel approach tends to save time since the calculations becomes
    the bottleneck rather then the overhead of using multiple processes.
    
    Note that each processes is in essence calling fit_spectrum but for different abundancies.
    """
    
    regions = _setup_regions(atlas, regions)
    return _parallel_calc(abund_range, processes, _fit_spectrum, (cfg_file, regions, use_default_abund, verbose))

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

def _equivalent_width(wav, inten):
    """
    Calculates the equivalent width for a line obtained with the given wavelength and intensity.
    """
    
    # The continuum level should be the maximum intensity
    cont = inten.max()

    # Calculate the area under the curve
    area = np.trapz(inten, x = wav)
    
    # If the area under the spectrum curve from wav[0] to wav[-1] is area, then the
    # area of the line is: area_line = total_area - area
    # where total_area is: total_area = continuum*(wav[-1] - wav[0])
    # assuming the continuum is constant in the interval (or close enough to constant
    # that it's not a too rough approximation to treat it as constant).
    area_line = cont*(wav[-1] - wav[0]) - area

    # If ew is the equivalent width, we have that: cont*ew = area_line
    # As such the equivalent width is given by ew = area_line/cont
    return abs(area_line/cont)

def _fit_width(abund_range, cfg_file, regions, eq_width_unit, use_default_abund, verbose):
    """
    DOCUMENT THIS!!!
    """
    
    # Convsersion factor
    conv_factor = (1 * astropy.units.AA).to(eq_width_unit).value
    
    # Create the abundancy updates and check them
    abund_updates = [au.abund(_ELEMENT, a) for a in abund_range]
    
    # Setup the regions
    region_data = _setup_region_data(regions)
    
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
        reduced_psf = psf / psf.sum()
        
        # Calculate the equivalent width of the observed data
        obs_ew = _equivalent_width(robs_wav, robs_inten) * conv_factor
        
        for ai, a in enumerate(abund_updates):
            # Get the region (the padding is to handle float related stuff... at least I think it's float related stuff... CHECK IT!!!!)
            rwav, rspec = r.get_contained(wav, spec[ai], left_padding = 1e-9)
            
            # Handle errors due to math with floating point numbers
            if len(rwav) != r.nlambda:
                print("*** DAJSD(dj!!! :::: len(rwav) - nlambda =", len(rwav) - r.nlambda, "\n")
                rwav = rwav[:r.nlambda]
                rspec = rspec[:r.nlambda]

            # Convolve the spectra
            rspec = _convolve(rspec, reduced_psf)
            inten_max[ai] = rspec.max()
            rspec /= inten_max[ai]
            
            # Calculate the equivalent width
            eq_width[ai] = _equivalent_width(rwav, rspec) * conv_factor
            diff[ai] = eq_width[ai] - obs_ew
            sinten[ai,:] = rspec
        result.append(EWRegionResult(r, rwav, sinten, inten_max, obs_ew, eq_width, diff, abund_updates, eq_width_unit))
    return SynthResult(result, region_data, wav, synth_data)

def fit_width(cfg_file, atlas, regions, abund_range, eq_width_unit = astropy.units.pm, use_default_abund = True, verbose = False):
    """
    This function synthesizes a spectrum and attempts to fit it to the observed spectrum for different iron abundancies. The required arguments are
    
        cfg_file          : The name of the cfg file.

        atlas             : An atlas object, which contains the observed spectrum.

        regions           : An iterable of Region objects, or alternatively regions in tuple form (see the regions module for more information).

        abund_range       : A range over the iron abundancies to synthezise the spectrum form.
        
    The optional arguments are
    
        eq_width_unit     : The unit for the equivalent width. These units come from astropy, specifically the module astropy.units.
                            Default is astropy.units.pm, which stands for picometers.

        use_default_abund : Determines if the default iron abundance should be used first.
                            Default is False.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object that contains the results of all calculations.

    Fitting is done by calculating the equivalent width of each region, both for the synthetic data and for the observed data. The abundance which
    gives the equivalent width that matches with the equivalent width of the observed data is taken as the best abundance.
    
    The iron abundancies are given as a range of float numbers. These numbers represents the relative abundance compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, so be sure the correct one is used.
    """
    
    regions = _setup_regions(atlas, regions)
    return _fit_width(abund_range, cfg_file, regions, eq_width_unit, use_default_abund, verbose)

def fit_width_parallel(cfg_file, atlas, regions, abund_range, processes = 2, eq_width_unit = astropy.units.pm, use_default_abund = False, verbose = False):
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

        use_default_abund : Determines if the default iron abundance should be used first.
                            Default is False.
        
        verbose           : Determines if more information then usual should be displayed. This is mainly for debugging.
                            Default is False.

    Returns a SynthResult object that contains the results of all calculations.

    Fitting is done by calculating the equivalent width of each region, both for the synthetic data and for the observed data. The abundance which
    gives the equivalent width that matches with the equivalent width of the observed data is taken as the best abundance.
    
    The iron abundances are given as a range of float numbers. These numbers represents the relative abundance compared to hydrogen. Specifically,
    if the number density of hydrogen and iron is N(H) and N(Fe) respecively, the abundance of iron A(Fe) is expected to be
    
        A(Fe) = log(N(Fe)) - log(N(H))
    
    There are other standards for abundance, so be sure the correct one is used.
    
    The distribution of work over the processes is done by making the processes handle different abundancies. For example, if we have R
    different regions, A different abundancies and N processes are used, then each process works with R different regions and approximately
    A / N different abundancies. When R and A are small the overhead of using multiple processes and coordinating them can make this slower
    then just calling fit_spectrum. However, when A and R increases this parallel approach tends to save time since the calculations becomes
    the bottleneck rather then the overhead of using multiple processes.
    
    Note that each processes is in essence calling fit_spectrum but for different abundancies.
    """
    
    regions = _setup_regions(atlas, regions)
    return _parallel_calc(abund_range, processes, _fit_width, (cfg_file, regions, eq_width_unit, use_default_abund, verbose))

