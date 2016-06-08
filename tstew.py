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
_MODE_PRINT_BEST = True
_MODE_SHOW_REGIONS = True
_MODE_VERBOSE = True

# Get lightspeed in the correct units
lightspeed = astropy.constants.c.to(astropy.units.km / astropy.units.s).value

# Read the cfg file
CFG_FILE = "data/lines.cfg"

# Get the atlas
at = sa.satlas()

# Get the regions
regions = regfile.get_regions()

# Refinements
# Bad (relatively) regions:
#       5253.4619
#regfile.refine(regions, 6219.28, 0.1, 0.0)
#regfile.refine(regions, , , 0.0)
#refinements = {
#    regfile.region_at(6219.28, regions): (0.1, 0.0),
#}
refinements = []

def _print_regions(regions):
    """
    Prints the regions
    """
    print("*** REGIONS ***")
    for r in regions:
        print(r)
    print("*** END REGIONS ***")
if _MODE_SHOW_REGIONS:
    _print_regions(regions)

# Create the abundencies (default not included)
abunds = -np.arange(4.1, 5.0, step = 0.01)

# Synth the spectrum and attempt to fit it to the observed data
time_start = time.time()
try:
    result = synther.fit_width_para(CFG_FILE, at, regions, abunds, refinements = refinements, verbose = _MODE_VERBOSE)
finally:
    time_end = time.time()

def print_best():
    best_abunds = []
    for r in result.region_result:
        print("Region:", r.region)
        print("    Best eq width:", r.best_eq_width)
        print("    Obs eq width: ", r.obs_eq_width)
        print("    Diff:         ", r.best_diff)
        print("    Abund:        ", r.best_abund)
        print("")
        if r.best_abund != []:
            best_abunds.append(au.get_value(r.best_abund))
        else:
            print("\n!!!!!!!!!!!!!! WHAT WAS THE DEFAULT ABUND AGAIN? WAS IT -4.5? BECAUSE I'M USING -4.5 !!!!!!!!!!!!!!\n")
            best_abunds.append(-4.5)
    print("Mean abund:", np.mean(best_abunds), "    or:", 12 + np.mean(best_abunds), " (as 12 + mean)")
if _MODE_PRINT_BEST:
    print_best()

def print_ew():
    for r in result.region_result:
        print("Region:   ", r.region)
        print("    Eq. width:", r.best_eq_width)
        print("    Observed: ", r.obs_eq_width)
        print("")

def plot_diff(region_nr):
    regres = result.region_result[region_nr]
    plotting.plot_vs_abund(regres.abund, regres.diff)

def plot_bisect(region_nr, abund_filter = None, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    plotting.plot_bisect(result.region_result[region_nr], abund_filter = abund_filter, offset = offset, plot_observed = plot_observed, plot_synth = plot_synth, show_observed = show_observed, show_synth = show_synth, num = num)

def countpts(lambda0, lambda_end, wav = None, padding = 0.0):
    """
    Counts the number of data points in wav between lambda0 and lambda_end. If wav is not given, the data from the
    atlas in the given region will be used as default.
    """
    if wav == None:
        wav, _, _ = at.getatlas(lambda0 - padding, lambda_end + padding, cgs = True)
    return len(wav[(lambda0 <= wav) & (wav <= lambda_end)])

def _conv(energy):
    """
    Converts energy from 1/cm to eV
    """
    return (astropy.constants.h*astropy.constants.c*(energy * (1/astropy.units.cm))).to(astropy.units.eV)

# Show the time the calculation took
print("Time:", time_end - time_start, " (seconds)")

