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
_MODE_FIT_BEST_SPACING = True
_MODE_SHOW_PLOTS = False
_MODE_SHOW_UNSHIFTED = True
_MODE_USE_SEEKING = True
_MODE_PRINT_BEST = False

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

# Get the regions
regions = regfile.regions

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

# Synth the spectrum and attempt to fit it to the observed data
time_start = time.time()
try:
    result = synther.fit_width(CFG_FILE, at, regions, abunds, verbose = True)
finally:
    time_end = time.time()

def _calc_vel(delta_lambda, lambda_em):
    """
    Calculates the velocity that corresponds to a doppler shift
    with a given shift delta_lambda and an emitted wavelength lambda_em.
    """
    return delta_lambda*300000.0/lambda_em

def print_best(fit_data):
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
if _MODE_PRINT_BEST:
    print_best()

def plot_diff(region_nr):
    regres = result.region_result[region_nr]
    plotting.plot_vs_abund(regres.abund, regres.diff)

def plot_bisect(region_nr, abund_filter = None, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    plotting.plot_bisect(result.region_result[region_nr], abund_filter = abund_filter, offset = offset, plot_observed = plot_observed, plot_synth = plot_synth, show_observed = show_observed, show_synth = show_synth, num = num)

def countpts(lambda0, lambda_end, wav):
    """
    Counts the number of data points in wav between lambda0 and lambda_end.
    """
    return len(wav[(lambda0 <= wav) & (wav <= lambda_end)])

def _conv(energy):
    """
    Converts energy from 1/cm to eV
    """
    return (astropy.constants.h*astropy.constants.c*(energy * (1/astropy.units.cm))).to(astropy.units.eV)

# Show the time the calculation took
print("Time:", time_end - time_start, " (seconds)")

