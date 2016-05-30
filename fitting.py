from __future__ import print_function

"""
This file contains the code used to fit the synthetic an observed spectra
in different ways.
"""

import numpy as np
import regions as regs
import time
import multiprocessing
from decimal import Decimal
from scipy.interpolate import splrep, splev

def chi_squared(obs_data, exp_data, err):
    """
    Calculates chi squared for the given set of observed and expected data,
    using the given error.
    """
    return np.sum(((obs_data - exp_data) / err)**2)

def interp_chi_squared(x_obs, y_obs, x_exp, y_exp, err):
    """
    """
    y_interp = np.interp(x_exp, x_obs, y_obs)
#    tck = splrep(x_obs, y_obs, s = 0)
#    y_interp = splev(x_exp, tck, der = 0)
    return chi_squared(y_interp, y_exp, err)

def _decimal_array(xs):
    return np.array([Decimal(x) for x in xs], dtype = Decimal)

class RegionFit:
    """
    Represents the fit of a specific region.
    """
    def __init__(self, shift, chisq, shifts_all, chisq_all, abunds, region):
        self.shift = shift
        self.chisq = chisq
        self.shifts_all = shifts_all
        self.chisq_all = chisq_all
        self.abunds = abunds
        self.region = region
        
        # Find the best values
        self.best_chisq = min(chisq)
        indx = np.where(chisq == min(chisq))[0][0]
        self.best_shift = shift[indx]
        self.best_abund = indx

class SpectrumFit:
    def __init__(self, region_fits, abunds):
        self.region_fits = region_fits
        self.abunds = abunds
        self.best_chisq = []
        self.best_shift = []
        self.best_abund_index = []
        self.best_abund = []
        self.regions = []
        for r in region_fits:
            self.best_chisq.append(r.best_chisq)
            self.best_shift.append(r.best_shift)
            self.best_abund_index.append(r.best_abund)
            self.best_abund.append(abunds[r.best_abund])
            self.regions.append(r.region)

#def fit_region(obs_wav, obs_inten, synth_wav, synth_inten, region):
def fit_region_OLD(obs_wav, obs_inten, synth, region):
    # Get the length of the region, the size of the shift steps, the maximum shift and the error
    rlen = regs.region_length(region)
    step_size = rlen / 2e4
    max_shift = rlen / 5.0
    err = 1.0
    
    # Create the range of shifts
    shifts = np.arange(0.0, max_shift, step = step_size)
#    shifts = np.array([0.021])
    
    # Get the region of the observed data, as well as the regions for the synthetic data
    owav, ointen = regs.get_region(region, obs_wav, obs_inten, padding = max_shift)
    rwav, rintens = regs.get_region_N(region, synth.wav, synth.inten)
    
    # Initialize the lists that stores the best values for the shift and chi squared, as well as all chi squared
    best_shift = []
    best_chisq = []
    chisq_all = []
    
    # Loop through all the abundencies, finding the best values for each one
    for inten in rintens:
        chisq = np.array([interp_chi_squared(owav + s, ointen, rwav, inten, err) for s in shifts])
        indx = np.where(chisq == min(chisq))[0][0]
        best_shift.append(shifts[indx])
        best_chisq.append(chisq[indx])
        chisq_all.append(chisq)
    
    # Returns a RegionFit
    return RegionFit(np.array(best_shift), np.array(best_chisq), shifts, np.array(chisq_all), synth.abund_updates, region)

def _send_fit_region(con, obs_wav, obs_inten, synth, regions):
    result = [fit_region(obs_wav, obs_inten, synth, r) for r in regions]
    #queue.put(result)
    con.send(result)
    con.close()

def fit_spectrum(obs_wav, obs_inten, synth, fitting_processes = 1):
    fits = []
    if fitting_processes == 1:
        start_time = time.time()
        for r in synth.region_data:
            reg_fit = fit_region(obs_wav, obs_inten, synth, r)
            fits.append(reg_fit)
        end_time = time.time()
    elif isinstance(fitting_processes, int) and fitting_processes > 1:
        start_time = time.time()

        # Distribute the regions among the processes
        regions = [None]*fitting_processes
        regs_per_process = int(np.ceil(float(len(synth.region_data)) / float(fitting_processes)))
        for i in range(fitting_processes):
            si = i*regs_per_process
            ei = (i + 1)*regs_per_process
            regions[i] = synth.region_data[si:ei]
        
        # Start each process
        print("*** Spawning processes ***")
        processes = []
        for r in regions:
            con_p, con_c = multiprocessing.Pipe()
            p = multiprocessing.Process(target = _send_fit_region, args = (con_c, obs_wav, obs_inten, synth, r))
            p.start()
            processes.append((con_p, p))
            print("Process spawned")
        
        # Join each process
        print("*** Joining processes ***")
        for con_p, p in processes:
            fits.extend(con_p.recv())
            con_p.close()
            p.join()
            print("Process joined")
        end_time = time.time()
    else:
        if not isinstance(fitting_processes, int):
            message = "fitting_processes must of type int but was of type " + type(fitting_processes).__name__
        else:
            message = "fitting processes must be an int greater then 0, but it had the value " + str(fitting_processes)
        raise Exception(message)
        
#        print("    Best shift: ", reg_fit.best_shift, "\n    Best chi sq:", reg_fit.best_chisq, "\n    Best abund: ", reg_fit.best_abund)
#    print("Shifts:\n", np.array(shifts_best), "\n*********\nBest chi sq:\n", np.array(chisq_best), "\n")
    print("*** TIME:", end_time - start_time, " seconds")
    return SpectrumFit(fits, synth.abund_updates)

def fit_spectrum_OLD(obs_wav, obs_inten, synth_wav, synth_inten, region_data):
    """
    """
    region_count = len(region_data)
    frame_count = len(synth_inten)
    region_shifts = np.zeros((region_count, frame_count))
    region_chisq = np.zeros((region_count, frame_count))
    for i in range(frame_count):
        shifts, chisq = fit_region_data(obs_wav, obs_inten, synth_wav, synth_inten[i], region_data)
        for r in range(region_count):
            region_shifts[r,i] = shifts[r]
            region_chisq[r,i] = chisq[r]
    best = np.array([min(region_chisq[r,:]) for r in range(region_count)])
    best_chisq = np.array([region_chisq[r,:][region_chisq[r,:] == best[r]][0] for r in range(region_count)])
#    best_shifts = np.array([region_shifts[r,:][best][0] for r in range(region_count)])
    best_shifts = np.array([region_shifts[r,:][region_chisq[r,:] == best[r]][0] for r in range(region_count)])
    print("*** BEST ***")
    print(best)
    print("*** SHIFTS ***")
    print(best_shifts)
    print("*** CHI SQUARED ***")
    print(best_chisq)
    return region_shifts, region_chisq

def find_wav_disp(obs_wav, obs_inten, synth_wav, synth_inten, region_data):
    """
    TODO: FIX THIS FUNCTION, GIVE IT A BETTER NAME AND DOCUMENT IT!!!
    """
 #   print("FITTING!!!")
    corrections = []
    correction_chisq = []
    start_time = time.time()
    for r in region_data:
#        print("REMEBER TO ASK ABOUT REGION FIRST THEN SHIFT OR SHIFT FIRST REGION SECOND!!!")
        # IMPORTANT QUESTION!
        # Should I get the region of obs_wav and then shift it around or
        # should I shift obs_wav around and then get the region from it?
        # Doing the latter for now since the former might introduce extra
        # errors.

        # Get the regions
        rwav_obs, rinten_obs = regs.get_region(r, obs_wav, obs_inten)
        rwav_synth, rinten_synth = regs.get_region(r, synth_wav, synth_inten)
        
        # Get the step size
        dwav = regs.region_length(r) / 5e3
#        print("\ndwav:", dwav)

        # err
        err = 1e-4

        # Shift the observed spectrum a distance dwav as long as it causes
        # chi squared to decrease
        inten = np.interp(rwav_synth, rwav_obs, rinten_obs)
        chisq = chi_squared(inten, rinten_synth, err)
        delta = dwav

        # Get the next shifted region
        rwav_obs, rinten_obs = regs.get_region(r, obs_wav + delta, obs_inten)
        inten = np.interp(rwav_synth, rwav_obs, rinten_obs)
        chisq2 = chi_squared(inten, rinten_synth, err)
        print("chisq:", str(chisq), "\nchisq2:", str(chisq2), "\nchisq - chisq2:", str(chisq - chisq2))
        while chisq2 <= chisq:
            chisq = chisq2
#            print("\n**********\ndelta:", delta, "\n**********\n")
            delta += dwav
            rwav_obs, rinten_obs = regs.get_region(r, obs_wav + delta, obs_inten)
            inten = np.interp(rwav_synth, rwav_obs, rinten_obs)
            chisq2 = chi_squared(inten, rinten_synth, err)
        print("chisq:", chisq, "\nchisq2:", chisq2, "\nchisq - chisq2:", chisq - chisq2, "\nBest delta:", delta - dwav, "        (displacement in wav)\n")
        corrections.append(delta - dwav)
        correction_chisq.append(chisq)
    end_time = time.time()
    print("Avg region time:", (end_time - start_time) / len(region_data), " seconds\nTotal time:", end_time - start_time, " seconds")
    return corrections, correction_chisq

def find_wav_disp_p(obs_wav, obs_inten, synth_wav, synth_inten, region_data):
#    interp = np.interp
    corrections = []
    start_time = time.time()
    for r in region_data:
#        print("REMEBER TO ASK ABOUT REGION FIRST THEN SHIFT OR SHIFT FIRST REGION SECOND!!!")
        # IMPORTANT QUESTION!
        # Should I get the region of obs_wav and then shift it around or
        # should I shift obs_wav around and then get the region from it?
        # Doing the latter for now since the former might introduce extra
        # errors.

        # Get the regions
        rwav_obs, rinten_obs = regs.get_region(r, obs_wav, obs_inten)
        rwav_synth, rinten_synth = regs.get_region(r, synth_wav, synth_inten)
        rinten_sp = _decimal_array(rinten_synth)
        
        # Get the step size
        dwav = regs.region_length(r) / 5e4
        print("\ndwav:", dwav)

        # err
        err = Decimal(32)

        # Shift the observed spectrum a distance dwav as long as it causes
        # chi squared to decrease
        inten = _decimal_array(np.interp(rwav_synth, rwav_obs, rinten_obs))
        chisq = chi_squared(inten, rinten_sp, err)
        delta = dwav

        # Get the next shifted region
        rwav_obs, rinten_obs = regs.get_region(r, obs_wav + delta, obs_inten)
        inten = _decimal_array(np.interp(rwav_synth, rwav_obs, rinten_obs))
        chisq2 = chi_squared(inten, rinten_sp, err)
        print("chisq:", str(chisq), "\nchisq2:", str(chisq2), "\nchisq - chisq2:", str(chisq - chisq2))
        print("chisq2 < chisq:", str(chisq2 < chisq), "chisq2 == chisq:", str(chisq2 == chisq), "chisq2 > chisq:", str(chisq2 > chisq))
        while chisq2 < chisq:
            chisq = chisq2
            print("\n**********\ndelta:", delta, "\n**********\n")
            delta += dwav
            rwav_obs, rinten_obs = regs.get_region(r, obs_wav + delta, obs_inten)
            inten = _decimal_array(np.interp(rwav_synth, rwav_obs, rinten_obs))
            chisq2 = chi_squared(inten, rinten_sp, err)
        print("chisq:", chisq, "\ndelta:", delta, "        (displacement in wav)\n")
        corrections.append(delta - dwav)
    end_time = time.time()
    print("Avg region time:", (end_time - start_time) / len(region_data), " seconds\nTotal time:", end_time - start_time, " seconds")
    return corrections
