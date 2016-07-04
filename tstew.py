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
import regfile
import plotting
from plotting import plot_in, plot_around

#raise Exception("Ask about why the amount of regions affects the result!")

# Used to quickly switch code
_MODE_FIT_BEST_SPACING = True
_MODE_SHOW_PLOTS = False
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
#regfile.refine(regions, 6481.85, 0.07, 0.0)

# Bad (relatively) regions:
#       5253.4619
#regfile.refine(regions, 6219.28, 0.1, 0.0)
#regfile.refine(regions, , , 0.0)
#regfile.refine(regions, 6481.86, 0.05, 0.0)
#refinements = {
#    14: (5232.43, 5232.55),
#}
#refinements = []

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
    result = synther.fit_width_parallel(CFG_FILE, at, regions, abunds, verbose = _MODE_VERBOSE)
finally:
    time_end = time.time()

def print_best():
    for r in result.region_result:
        print("Region:", r.region)
#        print("    Best eq width:", r.best_eq_width[0], "+-", r.best_eq_width[1])
#        print("    Obs eq width: ", r.obs_eq_width[0], "+-", r.obs_eq_width[1])
#        print("    Diff:         ", r.best_diff[0], "+-", r.best_diff[1])
        print("    Best eq width:", r.best_eq_width)
        print("    Obs eq width: ", r.obs_eq_width)
        print("    Diff:         ", r.best_diff)
        print("    Abund:        ", r.best_abund)
        print("")
    print("Best abunds:", result.best_abunds)
    print("Min abund: ", min(result.best_abunds), "     or:", min(result.best_abunds) + 12.0, " (as 12 + min abund)")
    print("Max abund: ", max(result.best_abunds), "     or:", max(result.best_abunds) + 12.0, " (as 12 + max abund)")
    print("Mean abund:", result.abund, "+-", result.error_abund, "     or:", result.abund + 12.0, "+-", result.error_abund, " (as 12 + mean abund)")
if _MODE_PRINT_BEST:
    print_best()

def print_ew():
    for r in result.region_result:
        print("Region:   ", r.region)
        print("    Eq. width:", r.best_eq_width)
        print("    Observed: ", r.obs_eq_width)
        print("")

def plot_diff(region_nr, **kwargs):
    regres = result.region_result[region_nr]
    plotting.plot_vs_abund(regres.abund, regres.diff, **kwargs)

def plot_bisect(region_nr, **kwargs):
    plotting.plot_bisect(result.region_result[region_nr], **kwargs)

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

