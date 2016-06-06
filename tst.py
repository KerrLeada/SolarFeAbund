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
import bisect

#raise Exception("Ask about why the amount of regions affects the result!")

# Used to quickly switch code
_MODE = 0
_MODE_FIT_BEST_SPACING = True
_MODE_SHOW_PLOTS = False
_MODE_SHOW_UNSHIFTED = True
_MODE_USE_SEEKING = True
_MODE_PRINT_BEST = False

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
            regs.new_region_in(at, 6301.18, 6301.82),

            # Line at: 6302.5 Å
            # POSSIBLY (BUT DOUBTFULLY) BETTER:
            #    6302.25 to 6302.75, gives shift of 0.004, cannot see much difference compared to the current
            #    interval 6302.25 to 6302.75. The chi squared is however higher for all abundencies, so it might
            #    just be worse.
            regs.new_region_in(at, 6302.3, 6302.7),
            
            # Line at: [6219.27] or 6219.28
            # CANDIDATES:
            #    dlambda = lambda w: 0.98*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.002... looks like a bit higher shift could be good
            #    dlambda = lambda w: 0.975*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.004... not sure if it fits or not...
            regs.new_region_in(at, 6219.01, 6219.53, dlambda = lambda w: 0.972*np.max(w[1:] - w[:-1])),
            
            # Line at: 6240.63 or [6240.64] or 6240.65
            # CANDIDATES: (NOT SURE WHICH ONE IS BEST, BUT THE FIRST ONE HAS BETTER CHI SQUARED THEN THE LATTER!!!)
            #    dlambda = lambda w: np.mean(w[1:] - w[:-1])          <---- shift = 0.008
            #    dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])      <---- shift = 0.006
#            regs.new_region_in(at, 6240.44, 6240.84, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),
            
            # Line at: 6481.85 or 6481.86 or [6481.87] (maybe: 6481.8 to 6481.9)
            # SHOULD POSSIBLY TRY AND PUSH THE SHIFT TO 0.004, WHICH MIGHT FIT BETTER! (KEYWORD IS MIGHT!)
            regs.new_region_in(at, 6481.87 - 0.25, 6481.87 + 0.25, dlambda = lambda w: 0.98*np.max(w[1:] - w[:-1])),
            
            # Line at: 6498.93 or 6498.94 (maybe: 6498.9 to 6498.97)
            # Currently we get a shift of about: 0.004 Å
            regs.new_region_in(at, 6498.938 - 0.3, 6498.938 + 0.3, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

            # Line at: 6581.19 or 6581.2 (maybe: 6581.18 to 6581.22)
            # Think this line has blending problems!!!!
#            regs.new_region_in(at, 6581.28 - 0.3, 6581.28 + 0.3, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

            # Line at: 6667.7 or 6667.71 (maybe: 6667.68 to 6667.74)
#            regs.new_region_in(at, , ),

            # Line at: 6699.11 or 6699.12 or 6699.13 or 6699.14 or 6699.15
#            regs.new_region_in(at, , ),

            # Line at: 6739.5 or 6739.51 or 6739.52 (maybe: 6739.48 to 6739.53)
#            regs.new_region_in(at, 6739.518 - 0.25, 6739.518 + 0.25, dlambda = lambda w: np.mean(w[1:] - w[:-1])),
            
            # Line at: 5329.98 or 5329.98
            # BLENDING: MIGHT NOT BE RELIABLE?
#            regs.new_region_in(at, 5329.987 - 0.10, 5329.987 + 0.30, dlambda = lambda w: 0.93*np.max(w[1:] - w[:-1])),

            # Line at: 5778.44 or 5778.45            
            regs.new_region_in(at, 5778.45 - 0.2, 5778.45 + 0.2, dlambda = lambda w: np.mean(w[1:] - w[:-1])),
            
            # Line at: 5701.53 or 5701.54 or 5701.55
            regs.new_region_in(at, 5701.54 - 0.3, 5701.54 + 0.3, dlambda = lambda w: np.max(w[1:] - w[:-1])),
            
            # Line at: 6836.99 or 6837 or 6837.01
            # REFINE
            regs.new_region_in(at, 6837 - 0.2, 6837 + 0.2, dlambda = lambda w: 0.967*np.max(w[1:] - w[:-1])),
            
            # Line at: 5253.4617
            # REFINE
            regs.new_region_in(at, 5253.4617 - 0.25, 5253.4617 + 0.25, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),
            
            # Line at: 5412.7833
            # REFINE
            regs.new_region_in(at, 5412.7833 - 0.2, 5412.7833 + 0.2, dlambda = lambda w: 0.95*np.max(w[1:] - w[:-1])),
            
            # Line at: 6750.1510
            regs.new_region_in(at, 6750.1510 - 0.3, 6750.1510 + 0.3),
            
            # Line at: 5705.4639
            # REFINE: WOULD BE BETTER IF IT COULD BE PUSHED 0.002 TO THE LEFT!!! (TO 0.008 FROM 0.006)
            regs.new_region_in(at, 5705.4639 - 0.25, 5705.4639 + 0.25, dlambda = lambda w: 0.991*np.max(w[1:] - w[:-1])),
            
            # Line at: 5956.6933
            regs.new_region_in(at, 5956.6933 - 0.3, 5956.6933 + 0.3, dlambda = lambda w: np.mean(w[1:] - w[:-1])),
            
            # **** STRONG LINES **** 6173.3339 6173.3347
#            regs.new_region_in(at, 6173.3339 - 0.3, 6173.3339 + 0.3),
#            regs.new_region_in(at, 6173.3347 - 0.3, 6173.3347 + 0.3),
            # Line at: 5232.93 or 5232.94 or 5232.95
            # CANDIDATES:
            #    dlambda = lambda w: 0.96*np.max(w[1:] - w[:-1])      <---- shift 0.002
            #    dlambda = lambda w: np.mean(w[1:] - w[:-1])          <---- shift 0.004   (best chi squared)
#            regs.new_region_in(at, 5232.94 - 1.5, 5232.94 + 1.5, dlambda = lambda w: 0.955*np.max(w[1:] - w[:-1])),
            
            # Line at: 4957.29 or 4957.3 or 4957.31
#            regs.new_region_in(at, 4957.3 - 1.5, 4957.3 + 1.5),
            
            # Line at: 4890.74 or 4890.75 or 4890.76 or 4890.77
            # Regarding dlambda: The synthetic line should be shifted to the left, towards lower wavelengths. If dlambda is the mean wavelength difference
            # or higher the synthetic line is shifted towards the right (higher wavelengths). Meanwhile, if dlambda is the minimum wavelength difference
            # it is shifted too much to the left. As such, we can conclude that dlambda should be between the minimum and the mean wavelength difference.
#            regs.new_region_in(at, 4890.75 - 1.5, 4890.75 + 1.5, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),
        ]
        initial_abunds = [
            (-4.45, -4.47),
            (-4.57, -4.59),
            (-4.51, -4.53),
            (-4.49, -4.5),
            (-4.4, -4.41),
            (-4.54, -4.55),
            (-4.44, -4.45),
            (-4.57, -4.58),
            (-4.61, -4.62),
            (-4.56, -4.57),
            (-4.5, -4.51),
            (-4.61, -4.62),
            (-4.42, -4.3),
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
#abunds = [-4.35, -4.4, -4.45, -4.55, -4.6, -4.65]
#abunds = -np.arange(4.1, 4.8, step = 0.005)
abunds = -np.arange(4.1, 5.0, step = 0.001)

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
time_start = time.time()
try:
    result = synther.fit_spectrum_para(CFG_FILE, at, regions, abunds, verbose = True)
finally:
    time_end = time.time()

#print_shifts(show_all = False)

def print_best2():
    if not (_MODE_USE_SEEKING and initial_abunds):
        raise Exception("No result from seeking mode")
    best_abunds = []
    for (chi2, a), fo in result2:
        print("Region:", fo.reg)
        print("    Best chisq:", chi2)
        print("    Best abund:", a)
        best_abunds.append(a)
    print("Mean abund:", np.mean(best_abunds), "     or:", np.mean(best_abunds) + 12, " (as 12 + mean)")
    print("*** ", best_abunds)

if _MODE_USE_SEEKING and initial_abunds:
    print("\n**********************")
    print("**** SEEKING MODE ****")
    print("**********************\n")
    result2 = synther.fit_spectrum_seeking(CFG_FILE, at, regions, initial_abunds, 0.001, verbose = True)

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
        if r.best_abund != []:
            best_abunds.append(au.get_value(r.best_abund))
        else:
            print("\n!!!!!!!!!!!!!! WHAT WAS THE DEFAULT ABUND AGAIN? WAS IT -4.5? BECAUSE I'M USING -4.5 !!!!!!!!!!!!!!\n")
            best_abunds.append(-4.5)
    print("Mean abund:", np.mean(best_abunds), "     or:", np.mean(best_abunds) + 12, " (as 12 + mean)")
    print("*** ", best_abunds)
if _MODE_PRINT_BEST:
    print_best()

print("FIX THE FITTING PROBLEM!!! SOMETIMES THE SHIFT DEPENDS ON OTHER LINES THEN THE ONE IN QUESTION, WHICH SHOULD NOT HAPPEN!!!\n*********\n")
if _MODE in {1, 2, 3}:
    print("MODE:", _MODE, "       (GENERATING REGIONS WITH: create_regions_in" + str(_MODE) + ")")
elif _MODE == -1:
    print("MODE:", _MODE, "       (REGIONS AND EVERYTHING COPIED FROM testshift.py)")
else:
    print("MODE:", _MODE, "       (REGIONS SPECIFIED MANALLY!!!)")

plot_color_list = ["#FF0000", "#00FF00", "#FF00FF",
                   "#11FF0A", "#AAAA00", "#00AAAA",
                   "#AF009F", "#0F3FF0", "#F0FA0F",
                   "#A98765", "#0152A3", "#0FFFF0",
                   "#C0110C", "#0B0D0E", "#BDC0EB"]

def plot_region(region_nr, offset = 0.0, alpha = 0.75, show_observed = True, show_unshifted = False, obs_pad = 0.0):
    # Get the region
    region_result = result.region_result[region_nr]
    
    # Set the title to display the region number, the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Nr: " + str(region_nr) + 
              "  Interval: " + str((region_result.region.lambda0, round(region_result.region.lambda_end, 4))) +
              "  delta: " + str(region_result.region.dlambda) +
              "  steps: " + str(region_result.region.nlambda) +
              "  scale: " + str(region_result.region.scale_factor))
#              "   Mode: " + str(_MODE) +
#              "   Region nr: " + str(region_nr))

    # Plot the synthetic spectrum
    for a in range(region_result.inten.shape[0]):
        # Unshifted
        if show_unshifted:
            plt.plot(region_result.wav + region_result.shift[a], region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.25, linestyle = "--")
        
        # Shifted
        plt.plot(region_result.wav - offset, region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = alpha)
    
    # Show the observed spectrum
    if show_observed:
        # Get the observed spectrum contained in the region
#        rwav, rinten = region_result.region.get_contained(wl, inten)
        if obs_pad == 0.0:
            rwav = region_result.region.wav
            rinten = region_result.region.inten
        else:
            # If the pad is a scalar, apply it to both ends, otherwise assume its a tuple with two elements, where the
            # first is the left pad and the second the right pad.
            if np.isscalar(obs_pad):
                lambda0 = region_result.region.lambda0 - obs_pad
                lambda_end = region_result.region.lambda_end + obs_pad
            else:
                lambda0 = region_result.region.lambda0 - obs_pad[0]
                lambda_end = region_result.region.lambda_end + obs_pad[1]
            
            # Get the wavelengths and intensities
            rwav, rinten, cont = at.getatlas(lambda0, lambda_end, cgs = True)
            rinten /= region_result.region.inten_scale_factor
        
        # Plot the spectrum, followed by the synth lines
        plt.plot(rwav, rinten, color = "blue")
    
    # Show the plot
    plt.show()

def plot_spec(show_observed = True, show_unshifted = False):
    # Set the title to display the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Mode: " + str(_MODE) + "   Reg len: " + str(reg_length))

    # Plot the regions
    for r in result.region_result:
        for a in range(r.inten.shape[0]):
            # Unshifted
            if show_unshifted:
                plt.plot(r.wav + r.shift[a], r.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.25, linestyle = "--")
            
            # Shifted
            plt.plot(r.wav, r.inten[a], color = plot_color_list[a % len(plot_color_list)])
    
    # Plot the entire observed spectrum
    if show_observed:
        plt.plot(wl, inten / inten.max(), color = "blue")
        
    plt.show()

def plot_chisq(region_nr):
    r = result.region_result[region_nr]
    a = au.list_abund(r.abund, default_val = -4.5)
    chi2 = r.chisq
    plt.plot(a, chi2)
    plt.show()

def plot_bisect(region_nr, abund_filter = None, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    if not (plot_observed or plot_synth):
        print("Must plot something")
    regres = result.region_result[region_nr]

    # Plot the bisection of the synthetic data    
    if plot_synth:
        # Get the wavelengths
        rwav = regres.wav
        
        # Filter the abundances (if needed)
        if isinstance(abund_filter, int):
            rinten_all = regres.inten[abund_filter]
        elif hasattr(abund_filter, "__call__"):
            rinten_all = abund_filter(regres.inten)
        elif abund_filter != None:
            rinten_all = regres.inten[abund_filter]
        else:
            rinten_all = regres.inten
        
        # Plot the bisections
        for a, rinten in enumerate(rinten_all):
            bwav, binten = bisect.get_bisection(rwav, rinten, num = num)
            if show_synth:
                plt.plot(rwav - offset, rinten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.4, linestyle = "--")
            plt.plot(bwav - offset, binten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.8)
    
    # Plot the bisection of the observed data
    if plot_observed:
        rwav = regres.region.wav
        rinten = regres.region.inten
        bwav, binten = bisect.get_bisection(rwav, rinten, num = num)
        if show_observed:
            plt.plot(rwav, rinten, color = "blue", alpha = 0.75, linestyle = "--")
        plt.plot(bwav, binten, color = "blue")

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
    wav, intensity, cont = at.getatlas(lambda0, lambda_end, cgs = True)
    if "normalize" not in kwargs or kwargs["normalize"]:
        intensity /= intensity.max()
#    wav, intensity = regs.get_within(lambda0, lambda_end, wl, inten)
    plt.plot(wav, intensity, *args, **kwargs)
    plt.show()

def plot_around(lambda_mid, length, *args, **kwargs):
    plot_in(lambda_mid - length, lambda_mid + length, *args, **kwargs)

def _conv(energy):
    """
    Converts energy from 1/cm to eV
    """
    return (astropy.constants.h*astropy.constants.c*(energy * (1/astropy.units.cm))).to(astropy.units.eV)

def print_best_eq(fit_data):
    best_abunds = []
    for f in fit_data:
        print("Region:", f.region)
        print("    Best eq width:", f.best_eq_width)
        print("    Obs eq width: ", f.obs_eq_width)
        print("    Diff:         ", f.best_diff)
        print("    Abund:        ", f.best_abund)
        print("")
        if f.best_abund != []:
            best_abunds.append(au.get_value(f.best_abund))
        else:
            print("\n!!!!!!!!!!!!!! WHAT WAS THE DEFAULT ABUND AGAIN? WAS IT -4.5? BECAUSE I'M USING -4.5 !!!!!!!!!!!!!!\n")
            best_abunds.append(-4.5)
    print("Mean abund:", np.mean(best_abunds), "    or:", 12 + np.mean(best_abunds), " (as 12 + mean)")

#def _print_vel(dlambda):
#    _, lambda_min = result.region_min()
#    for a in range(lambda_min.shape[0]):
#        print("*** Abundance nr", a, "***")
#        for dl, l_em in zip(dlambda, lambda_min[a,:]):
#            print("dl: ", dl, "\nl_obs:  ", l_em - dl, "\nVel:", dl*lightspeed/l_em, "\n")
#    print("NOTE: WHEN dl IS DIRECTLY SET TO dl = 0.021 THE SYNTHETIC CURVE IS NOTICABLY DISPLACED COMPARED TO THE ATLAS SPECTRA AND DOES NOT FIT!!!")
#    print("      AND WITH NOTICABLY I MEAN IT'S VISIBLE TO THE NAKED EYE (DEPENDING ON ZOOM LEVEL)!!!")
#    print("      SHOULD PROBABLY MAKE SURE I HAVE COMPARED TO THE CORRECT ATLAS SPECTAL LINES!!!")

# Show the time the calculation took
print("Time:", time_end - time_start, " (seconds)")

