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
import regfile
import plotting
from plotting import plot_in, plot_around

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
        regions = regfile.regions
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
abunds = -np.arange(4.1, 5.0, step = 0.01)

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

def print_best_seeking():
    if not (_MODE_USE_SEEKING and initial_abunds):
        raise Exception("No result from seeking mode")
    best_abunds = []
    for (chi2, a), fo in result2:
        print("Region:", fo.reg)
        print("    Best chisq:", chi2)
        print("    Best abund:", a)
        print("")
        best_abunds.append(a)
    print("Mean abund:", np.mean(best_abunds), "     or:", np.mean(best_abunds) + 12, " (as 12 + mean)")
    print("*** ", best_abunds)

if _MODE_USE_SEEKING and initial_abunds:
    print("\n**********************")
    print("**** SEEKING MODE ****")
    print("**********************\n")
    result2 = synther.fit_spectrum_seeking(CFG_FILE, at, regions, initial_abunds, 0.01, verbose = True)

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
        print("    Velocity: ~", _calc_vel(r.best_shift, r.wav[np.argmin(r.inten)]), "     ( Using delta_lambda =", r.best_shift, "lambda_em =", r.wav[np.argmin(r.inten)], ")")
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

def plot_region(region_nr, offset = 0.0, alpha = 0.75, show_observed = True, show_unshifted = False, obs_pad = 0.0):
    plotting.plot_region(result.region_result[region_nr], offset = offset, alpha = alpha, show_observed = show_observed, show_unshifted = show_unshifted, obs_pad = obs_pad)

def plot_spec(show_observed = True, show_continuum = False, show_unshifted = False, padding = 0.0, cgs = True):
    # Set the title to display the mode and if the spacing between the datapoints in the synth region was fitted
    plot_title = "Mode: " + str(_MODE)
    plotting.plot_spec(result.region_result, show_observed = show_observed, show_continuum = show_continuum, show_unshifted = show_unshifted, padding = padding, plot_title = plot_title, cgs = cgs)

def plot_chisq(region_nr):
    regres = result.region_result[region_nr]
    plotting.plot_vs_abund(regres.abund, regres.chisq)

def plot_bisect(region_nr, abund_filter = None, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    plotting.plot_bisect(result.region_result[region_nr], abund_filter = abund_filter, offset = offset, plot_observed = plot_observed, plot_synth = plot_synth, show_observed = show_observed, show_synth = show_synth, num = num)

if _MODE_SHOW_PLOTS:
#    plot_region(-1, show_unshifted = _MODE_SHOW_UNSHIFTED)
    plot_spec(show_unshifted = _MODE_SHOW_UNSHIFTED)

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
            best_abunds.append(_au.get_value(f.best_abund))
        else:
            print("\n!!!!!!!!!!!!!! WHAT WAS THE DEFAULT ABUND AGAIN? WAS IT -4.5? BECAUSE I'M USING -4.5 !!!!!!!!!!!!!!\n")
            best_abunds.append(-4.5)
    print("Mean abund:", np.mean(best_abunds), "    or:", 12 + np.mean(best_abunds), " (as 12 + mean)")

def countpts(lambda0, lambda_end, wav = None):
    """
    Counts the number of data points in wav between lambda0 and lambda_end. If wav is not given, wl will
    be used as default.
    """
    if wav == None:
        wav = wl
    return len(wav[(lambda0 <= wav) & (wav <= lambda_end)])

def _conv(energy):
    """
    Converts energy from 1/cm to eV
    """
    return (astropy.constants.h*astropy.constants.c*(energy * (1/astropy.units.cm))).to(astropy.units.eV)

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

