# -*- coding: utf8 -*-

from __future__ import print_function

import numpy as np
import pyLTE as p
import matplotlib.pyplot as plt
import time
import satlas as sa
import sparsetools as sp
import regions as regs
import cfg
import fitting
import astropy.constants
import astropy.units
import synther
import abundutils as au

#raise Exception("Ask about why the amount of regions affects the result!")

# Used to quickly switch code
_MODE = 0
_MODE_FIT_BEST_SPACING = True
_MODE_INTERP_OBS = False
_MODE_SHOW_PLOTS = False
_MODE_SHOW_UNSHIFTED = True
_MODE_USE_SEEKING = False

#
initial_abunds = None

# Get lightspeed in the correct units
lightspeed = astropy.constants.c.to(astropy.units.km / astropy.units.s).value

# Read the cfg file
CFG_FILE = "data/lines.cfg"
cfg_data = cfg.read_cfg(CFG_FILE)

# Set the region length (used in automatic mode)
reg_length = 0.6

# Get the wavelengths from the CFG file
cfg_wav = np.array(cfg.get_column(cfg_data, "wav", dtype = float))
reg_wav0 = cfg_wav - reg_length/2.0

# Get the atlas
at = sa.satlas()

if _MODE == -1:
    # Get Continuum intensity from the FTS atlas in CGS units
    #   at = atlas
    #   wl = wavelengths
    #   inten = intensity at the different wavelengths
    #   cont = the continuum level at the different wavelengths (according
    #          to the atlas, it can technically be wrong, since it's
    #          basically a fit some people thought looked like the correct
    #          fit).
    wl, inten_raw, cont_atlas = at.getatlas(6301.2, 6301.8, cgs = True)
#    print(wl)
    
    # Get the region
    dwa = wl[1] - wl[0]
    nwa = int(4.0 / dwa)+1
    regions = [
        regs.region(wl[0], dwa, wl.size, 1.0)
    ]
# Switch: From cfg file (True) or from testshift (False)
else:
    # Get Continuum intensity from the FTS atlas in CGS units
    #   at = atlas
    #   wl = wavelengths
    #   inten = intensity at the different wavelengths
    #   cont = the continuum level at the different wavelengths (according
    #          to the atlas, it can technically be wrong, since it's
    #          basically a fit some people thought looked like the correct
    #          fit).
    wl, inten_raw, cont_atlas = at.getatlas(min(cfg_wav) - 1.5,max(cfg_wav) + 1.5, cgs=True)
    
    # Get the regions from the cfg file
    if _MODE == 1:
        regions = regs.create_regions_in1(reg_wav0, reg_length, wl, 1.0, fit_best_spacing = _MODE_FIT_BEST_SPACING)
    elif _MODE == 2:
        regions = regs.create_regions_in2(reg_wav0, reg_length, wl, 1.0, fit_best_spacing = _MODE_FIT_BEST_SPACING)
    elif _MODE == 3:
        regions = regs.create_regions_in3(reg_wav0, reg_length, wl, 1.0, fit_best_spacing = _MODE_FIT_BEST_SPACING)
    else:
        regions = [
            # Line at: 6301.5 Å
            regs.new_region_in(at, 6301.2, 6301.8, interp_obs = False),

            # Line at: 6302.5 Å
            # POSSIBLY (BUT DOUBTFULLY) BETTER:
            #    6302.25 to 6302.75, gives shift of 0.004, cannot see much difference compared to the current
            #    interval 6302.25 to 6302.75. The chi squared is however higher for all abundencies, so it might
            #    just be worse.
            regs.new_region_in(at, 6302.25, 6302.75, interp_obs = False),
            
            # Line at: [6219.27] or 6219.28
            # CANDIDATES:
            #    dlambda = lambda w: 0.98*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.002... looks like a bit higher shift could be good
            #    dlambda = lambda w: 0.975*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.004... not sure if it fits or not...
            regs.new_region_in(at, 6219.07, 6219.47, dlambda = lambda w: 0.972*np.max(w[1:] - w[:-1]), interp_obs = False),
            
            # Line at: 6240.63 or [6240.64] or 6240.65
            # CANDIDATES: (NOT SURE WHICH ONE IS BEST, BUT THE FIRST ONE HAS BETTER CHI SQUARED THEN THE LATTER!!!)
            #    dlambda = lambda w: np.mean(w[1:] - w[:-1])          <---- shift = 0.008
            #    dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])      <---- shift = 0.006
            regs.new_region_in(at, 6240.44, 6240.84, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1]), interp_obs = False),
            
            # Line at: 6481.85 or 6481.86 or [6481.87] (maybe: 6481.8 to 6481.9)
            # (CAN MAYBE CHANGE WAVELENGTH A LITTLE)
            regs.new_region_in(at, 6481.87 - 0.3, 6481.87 + 0.3, dlambda = lambda w: 0.98*np.max(w[1:] - w[:-1]), interp_obs = False),
            
            # Line at: 6498.93 or 6498.94 (maybe: 6498.9 to 6498.97)
            # Currently we get a shift of about: 0.004 Å
            regs.new_region_in(at, 6498.938 - 0.3, 6498.938 + 0.3, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1]), interp_obs = False),

            # Line at: 6581.19 or 6581.2 (maybe: 6581.18 to 6581.22)
#            regs.new_region_in(at, , , interp_obs = False),

            # Line at: 6667.7 or 6667.71 (maybe: 6667.68 to 6667.74)
#            regs.new_region_in(at, , , interp_obs = False),

            # Line at: 6699.11 or 6699.12 or 6699.13 or 6699.14 or 6699.15
#            regs.new_region_in(at, , , interp_obs = False),

            # Line at: 6739.5 or 6739.51 or 6739.52 (maybe: 6739.48 to 6739.53)
#            regs.new_region_in(at, 6739.518 - 0.25, 6739.518 + 0.25, dlambda = lambda w: np.mean(w[1:] - w[:-1]), interp_obs = False),
            
            # **** STRONG LINES ****
            # Line at: 5232.93 or 5232.94 or 5232.95
            # CANDIDATES:
            #    dlambda = lambda w: 0.96*np.max(w[1:] - w[:-1])      <---- shift 0.002
            #    dlambda = lambda w: np.mean(w[1:] - w[:-1])          <---- shift 0.004   (best chi squared)
            regs.new_region_in(at, 5232.94 - 1.5, 5232.94 + 1.5, dlambda = lambda w: 0.955*np.max(w[1:] - w[:-1]), interp_obs = False),
        ]
        initial_abunds = [
            (-4.5, -4.55),
            (-4.5, -4.45),
            (-4.5, -4.45),
        ]

# Get the continuum and normalized intensity
cont = cont_atlas[0]
inten = inten_raw / inten_raw.max()

def _print_regions(regions):
    """
    Prints the regions
    """
    print("*** REGIONS ***")
    for r in regions:
        print(r)
    print("*** END REGIONS ***")
_print_regions(regions)

# Create the abundencies (default not included)
#abunds = []
#abunds = [-4.4, -4.42, -4.45, -4.48, -4.52, -4.55, -4.58, -4.6]
abunds = -np.arange(4.1, 4.9, step = 0.001)

def print_shifts(show_all = True):
    for r in result.region_result:
        print("Region:", str(r.region))
        for a, s, c2, ainten in zip(r.abund, r.shift, r.chisq, r.inten):
            line_wav = r.wav[np.argmin(ainten)]
            line_wav_em = line_wav + s
            print("    Abund:", a)
            print("    Shift:", s)
            print("    Chisq:", c2)
            if show_all:
                print("    Doppler velocity:", _calc_vel(s, line_wav_em), "km/s")
                print("    Unshifted line max wavelength:", line_wav_em, "Angstrom")
                print("    Shifted line max wavelength:  ", line_wav, "Angstrom")
            print("")
        print("")

# Synth the spectrum and attempt to fit it to the observed data
result = synther.fit_spectrum(CFG_FILE, wl, inten, regions, abunds, verbose = True, interp_obs = _MODE_INTERP_OBS)
print("REGION LEN:", reg_length)
print_shifts(show_all = False)

if _MODE_USE_SEEKING and initial_abunds:
    regions_abunds = zip(regions, initial_abunds)
    result2 = synther.fit_spectrum2(CFG_FILE, wl, inten, regions_abunds, [0.01, 0.001, 0.0001], verbose = True)
    print(result2)

def _calc_vel(delta_lambda, lambda_em):
    """
    Calculates the velocity that corresponds to a doppler shift
    with a given shift delta_lambda and an emitted wavelength lambda_em.
    """
    return delta_lambda*300000.0/lambda_em

def print_best():
    best_abunds = []
    for r in result.region_result:
        print("Region:", r.region)
        print("    Best chisq:", r.best_chisq)
        print("    Best shift:", r.best_shift)
        print("    Best abund:", r.best_abund)
        print("")
        best_abunds.append(au.get_value(r.best_abund))
    print("Mean abund:", np.mean(best_abunds))
    print("*** ", best_abunds)

print("FIX THE FITTING PROBLEM!!! SOMETIMES THE SHIFT DEPENDS ON OTHER LINES THEN THE ONE IN QUESTION, WHICH SHOULD NOT HAPPEN!!!\n*********\n")
if _MODE in {1, 2, 3}:
    print("MODE:", _MODE, "       (GENERATING REGIONS WITH: create_regions_in" + str(_MODE) + ")")
elif _MODE == -1:
    print("MODE:", _MODE, "       (REGIONS AND EVERYTHING COPIED FROM testshift.py)")
else:
    print("MODE:", _MODE, "       (REGIONS SPECIFIED MANALLY!!!)")
print("INTERPOLATING OBSERVED:", _MODE_INTERP_OBS)
print("REGION LEN:", reg_length)

plot_color_list = ["#FF0000", "#00FF00", "#FF00FF",
                   "#11FF0A", "#AAAA00", "#00AAAA",
                   "#AF009F", "#0F3FF0", "#F0FA0F",
                   "#A98765", "#0152A3", "#0FFFF0",
                   "#C0110C", "#0B0D0E", "#BDC0EB"]

def plot_region(region_nr, show_observed = True, show_unshifted = True):
    # Get the region
    region_result = result.region_result[region_nr]
    
    # Set the title to display the region number, the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Nr: " + str(region_nr) + 
              "  Interval: " + str((region_result.region.lambda0, round(region_result.region.lambda_end, 4))) +
              "  delta: " + str(region_result.region.dlambda) +
              "  steps: " + str(region_result.region.nlambda) +
              "  scale: " + str(region_result.region.scale_factor))
#              "   Mode: " + str(_MODE) +
#              "   Fit spacing: " + str(_MODE_FIT_BEST_SPACING) +
#              "   Interp obs: " + str(_MODE_INTERP_OBS) +
#              "   Region nr: " + str(region_nr))

    # Show the observed spectrum
    if show_observed:
        # Get the observed spectrum contained in the region
        rwav, rinten = region_result.region.get_contained(wl, inten)
        
        # Plot the spectrum, followed by the synth lines
        plt.plot(rwav, rinten, color = "blue")

    # Plot the synthetic spectrum
    for a in range(region_result.inten.shape[0]):
        # Unshifted
        if show_unshifted:
            plt.plot(region_result.wav + region_result.shift[a], region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.5)
        
        # Shifted
        plt.plot(region_result.wav, region_result.inten[a], color = plot_color_list[a % len(plot_color_list)])
    
    # Show the plot
    plt.show()

def plot_spec(show_observed = True, show_unshifted = True):
    # Set the title to display the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Mode: " + str(_MODE) +
 #             "   Fit spacing: " + str(_MODE_FIT_BEST_SPACING) +
              "   Interp obs: " + str(_MODE_INTERP_OBS) +
              "   Reg len: " + str(reg_length))

    # Setup the color list
#    color_list = ["#FF0000", "#00FF00", "#FF00FF",
#                  "#11FF0A", "#AAAA00", "#00AAAA",
#                  "#AF009F", "#0F3FF0", "#F0FA0F",
#                  "#A98765", "#0152A3", "#0FFFF0",
#                  "#C0110C", "#0B0D0E", "#BDC0EB"]
    
    # Plot the entire observed spectrum
    if show_observed:
        plt.plot(wl, inten, color = "blue")

    # Plot the regions
    for r in result.region_result:
        for a in range(r.inten.shape[0]):
            # Unshifted
            if show_unshifted:
                plt.plot(r.wav + r.shift[a], r.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.5)
            
            # Shifted
            plt.plot(r.wav, r.inten[a], color = plot_color_list[a % len(plot_color_list)])
    plt.show()

def plot_chisq(region_nr):
    r = result.region_result[region_nr]
    a = au.list_abund(r.abund, default_val = -4.5)
    chi2 = r.chisq
    plt.plot(a, chi2)
    plt.show()

if _MODE_SHOW_PLOTS:
#    plot_region(-1, show_unshifted = _MODE_SHOW_UNSHIFTED)
    plot_spec(show_unshifted = _MODE_SHOW_UNSHIFTED)

def countpts(lambda0, lambda_end, wav = None):
    """
    Counts the number of data points in wav between lambda0 and lambda_end. If wav is not given, wl will
    be used as default.
    """
    if wav == None:
        wav = wl
    return len(wav[(lambda0 <= wav) & (wav <= lambda_end)])

def plot_in(lambda0, lambda_end, *args, **kwargs):
    """
    Plots the observed spectra between lambda0 and lambda_end.
    """
    wav, intensity = regs.get_within(lambda0, lambda_end, wl, inten)
    plt.plot(wav, intensity, *args, **kwargs)
    plt.show()

#def _print_vel(dlambda):
#    _, lambda_min = result.region_min()
#    for a in range(lambda_min.shape[0]):
#        print("*** Abundance nr", a, "***")
#        for dl, l_em in zip(dlambda, lambda_min[a,:]):
#            print("dl: ", dl, "\nl_obs:  ", l_em - dl, "\nVel:", dl*lightspeed/l_em, "\n")
#    print("NOTE: WHEN dl IS DIRECTLY SET TO dl = 0.021 THE SYNTHETIC CURVE IS NOTICABLY DISPLACED COMPARED TO THE ATLAS SPECTRA AND DOES NOT FIT!!!")
#    print("      AND WITH NOTICABLY I MEAN IT'S VISIBLE TO THE NAKED EYE (DEPENDING ON ZOOM LEVEL)!!!")
#    print("      SHOULD PROBABLY MAKE SURE I HAVE COMPARED TO THE CORRECT ATLAS SPECTAL LINES!!!")

