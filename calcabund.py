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
import latexgen
from plotting import plot_in, plot_around

# Used to quickly switch code
_MODE = 0
_MODE_PRINT_BEST = True
_MODE_SHOW_REGIONS = False
_MODE_VERBOSE = True

# Get lightspeed in the correct units
lightspeed = astropy.constants.c.to(astropy.units.km / astropy.units.s).value

# Read the cfg file
CFG_FILE = "data/lines.cfg"

# Get the atlas
at = sa.satlas()

# Get the regions
regions = regfile.get_regions()

def _print_regions():
    """
    Prints the regions
    """
    
    print("*** REGIONS ***")
    for r in regions:
        print(r)
    print("*** END REGIONS ***")
if _MODE_SHOW_REGIONS:
    _print_regions()

def twfor(region_nr):
    return (np.arange(15)-7)*(regions[region_nr].wav[1] - regions[region_nr].wav[0])

def gaussfor(region_nr, tw, vel = 1.83):
    return synther._gaussian(tw, [1.0, 0.0, vel*regions[region_nr].lambda_end/300000.0])

def gaussfor2(region_nr, vel = 1.83):
    tw = twfor(region_nr)
    return tw, gaussfor(region_nr, tw, vel = vel)

def plotgauss(region_nr, vels = None):
    if vels == None:
        vels = [1.83]
    tw = twfor(region_nr)
    for v in vels:
        plt.plot(tw, gaussfor(region_nr, tw, vel = v))
    plt.show()

def print_shifts(show_all = True):
    """
    Prints the shifts. By default the abundance and chi squared is printed as well, however
    more information can be printed by setting the show_all argument to true. If show_all is
    true then the doppler velocity (which is the velocity that would give a doppler shift
    corresponding to the obtained shift), the wavelength of the line maximum without shifting
    the line as well as the wavelength of the maximum when shifting the line are shown as well.
    """
    
    for r in result.region_result:
        print("Region:", str(r.region))
        for a, s, c2, ainten in zip(r.abund, r.shift, r.chisq, r.inten):
            line_wav = r.estimate_minimum()
            line_wav_em = line_wav + s
            print("    Abund:", a)
            print("    Shift:", s)
            print("    Chisq:", c2)
            if show_all:
                print("    Doppler velocity:", _calc_vel(s, line_wav_em), "km/s")
                print("    Unshifted line max wavelength:", line_wav_em, u"Ångström")
                print("    Shifted line max wavelength:  ", line_wav, u"Ångström")
            print("")
        print("")

# Create the abundances (these limits where chosen after calculating
# for a larger interval and noticing it was too large... specifically
# it used to be from -4.1 to -5.0).
abunds = -np.arange(4.3, 4.8, step = 0.01)

# Synth the spectrum and attempt to fit it to the observed data using the chi squared method
time_start = time.time()
try:
    result = synther.fit_spectrum(CFG_FILE, at, regions, abunds, verbose = _MODE_VERBOSE)
finally:
    time_end = time.time()
    time_tot = time_end - time_start
result_chi = result.result_chi
result_ew = result.result_ew

def _calc_vel(delta_lambda, lambda_em):
    """
    Calculates the velocity that corresponds to a doppler shift
    with a given shift delta_lambda and an emitted wavelength lambda_em.
    """
    
    return delta_lambda*300000.0/lambda_em

def print_best():
    # Print the results for each region
    for r_chi, r_ew in zip(result_chi.region_result, result_ew.region_result):
        lambda_em = r_chi.wav[np.argmin(r_chi.inten[r_chi.best_index])]
        print("Region:", r_chi.region)
        print("    Chi squared method:")
        print("        Best chisq:", r_chi.best_chisq)
        print("        Best shift:", r_chi.best_shift)
        print("        Best abund:", r_chi.best_abund)
        print("        Velocity: ~", _calc_vel(r_chi.best_shift, lambda_em), "     ( Using delta_lambda =", r_chi.best_shift, "lambda_em =", lambda_em, ")")
        print("    Equivalent widths:")
        print("        Best eq width:", r_ew.best_eq_width)
        print("        Obs eq width: ", r_ew.obs_eq_width)
        print("        Best diff:    ", r_ew.best_diff)
        print("        Best abund:   ", r_ew.best_abund)
        print("")
    
    # Print the final results
    for result, method in [(result_chi, "chi squared"), (result_ew, "equivalent widths")]:
        print("Result using", method, "yields:")
        print("    Best abunds:", result.best_abunds)
        print("    Min abund: ", min(result.best_abunds), "     or:", min(result.best_abunds) + 12.0, " (as 12 + min abund)")
        print("    Max abund: ", max(result.best_abunds), "     or:", max(result.best_abunds) + 12.0, " (as 12 + max abund)")
        print("    Mean abund:", result.abund, "+-", result.error_abund, "     or:", result.abund + 12.0, "+-", result.error_abund, " (as 12 + mean abund)")
if _MODE_PRINT_BEST:
    print_best()

def plot_region(region_nr, **kwargs):
    plotting.plot_region(result_chi.region_result[region_nr], **kwargs)

#def plot_spec(show_observed = True, show_continuum = False, show_unshifted = False, padding = 0.0, cgs = True):
#    # Set the title to display the mode and if the spacing between the datapoints in the synth region was fitted
#    plotting.plot_spec(result_chi.region_result, show_observed = show_observed, show_continuum = show_continuum, show_unshifted = show_unshifted, padding = padding, cgs = cgs)

def plot_chisq(region_nr):
    regres = result_chi.region_result[region_nr]
    plotting.plot_chisq(regres)

def plot_diff(region_nr):
    regres = result.region_result[region_nr]
    plotting.plot_vs_abund(regres.abund, regres.diff)

def plot_bisect(region_nr, **kwargs):
    plotting.plot_bisect(result_chi.region_result[region_nr], **kwargs)

def plot_dwav(region_nr):
    plotting.plot_delta(regions[region_nr].wav, xlabel = "$i$", ylabel = "$\\lambda_i - \\lambda_{i - 1}$")

def plot_chi_mosaic(region_nrs, rows, columns, plot_func, *args, **kwargs):
    """
    Same as plotting.plot_mosaic, except for the first argument 'region_nrs'. Specifically, 'region_nrs' is an
    iterable over the indices of the region results to pass on to plotting.plot_mosaic. The region results come
    from the result of the chi squared calculation.
    """
    
    regres = [r for i, r in enumerate(result_chi.region_result) if i in region_nrs]
    plotting.plot_mosaic(regres, rows, columns, plot_func, *args, **kwargs)
    
def plot_ew_mosaic(region_nrs, rows, columns, plot_func, *args, **kwargs):
    """
    Same as plotting.plot_mosaic, except for the first argument 'region_nrs'. Specifically, 'region_nrs' is an
    iterable over the indices of the region results to pass on to plotting.plot_mosaic. The region results come
    from the result of the equivalent width calculation.
    """
    
    regres = [r for i, r in enumerate(result_ew.region_result) if i in region_nrs]
    plotting.plot_mosaic(regres, rows, columns, plot_func, *args, **kwargs)

def plot_mosaic2(rows, columns, plot_func, *args, **kwargs):
    """
    Same as plotting.plot_mosaic except that it automatically passes in a list with the result objects (first
    the result for chi squared, then the result for equivalent widths). This is why it is called plot_mosaic2.
    The "2" comes from that it passes two objects to plotting.plot_mosaic.
    """
    
    results = [result_chi.region_result, result_ew.region_result]
    plotting.plot_mosaic(results, rows, columns, plot_func, *args, **kwargs)

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
print("")
print("Total time:", time_tot, "seconds")

# Final reminder of where the results are stored
print("")
print("The variable 'result' is an instance of ResultPair and contains the results of both fits.")
print("The variable 'result_chi' contains the result of the calculation using the chi squared method, and")
print("the variable 'result_ew' contains the results of the calculation using equivalent widths.")

