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
from abundutils import abund

# Get lightspeed in the correct units
lightspeed = astropy.constants.c.to(astropy.units.km / astropy.units.s).value

# Read the cfg file
CFG_FILE = "data/lines.cfg"
cfg_data = cfg.read_cfg(CFG_FILE)

# Set the region length (used in automatic mode)
reg_length = 0.8

# Get the wavelengths from the CFG file
cfg_wav = np.array(cfg.get_column(cfg_data, "wav", dtype = float))
reg_wav0 = cfg_wav - reg_length/2.0

# Get the atlas
at = sa.satlas()

# Used to quickly switch code
_MODE = 3
_MODE_FIT_BEST_SPACING = True
_MODE_INTERP_OBS = True

# Switch: From cfg file (True) or from testshift (False)
if _MODE in {1, 2, 3}:
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
        raise Exception("MODE NOTE IMPLEMENTED: " + str(_MODE))
elif _MODE == -1:
    # Get Continuum intensity from the FTS atlas in CGS units
    #   at = atlas
    #   wl = wavelengths
    #   inten = intensity at the different wavelengths
    #   cont = the continuum level at the different wavelengths (according
    #          to the atlas, it can technically be wrong, since it's
    #          basically a fit some people thought looked like the correct
    #          fit).
    wl, inten_raw, cont_atlas = at.getatlas(6301.2,6301.8, cgs=True)
    
    # Get the region
    dwa = wl[1] - wl[0]
    nwa = int(4.0 / dwa)+1
    regions = [
        regs.region(wl[0], dwa, wl.size, 1.0)
    ]
else:
    raise Exception("NOT THERE YET...")

if _MODE in {1, 2, 3}:
    print("GENERATING REGIONS WITH: create_regions_in" + str(_MODE))
elif _MODE == -1:
    print("REGIONS AND EVERYTHING COPIED FROM testshift.py")
else:
    print("REGIONS SPECIFIED MANALLY!!!")

# Get the continuum
cont = cont_atlas[0]
inten = inten_raw / inten_raw.max()

# Create the abundencies (default not included)
abunds = []

# Synth the spectrum and attempt to fit it to the observed data
result = synther.fit_spectrum(CFG_FILE, wl, inten, regions, abunds, verbose = True, interp_obs = _MODE_INTERP_OBS)

def _calc_vel(delta_lambda, lambda_em):
    """
    Calculates the velocity that corresponds to a doppler shift
    with a given shift delta_lambda and an emitted wavelength lambda_em.
    """
    return delta_lambda*300000.0/lambda_em

def _print_shifts():
    for r in result.region_result:
        print("Region:", r.region)
        for a, s, c2, ainten in zip(r.abund, r.shift, r.chisq, r.inten):
            line_wav = r.wav[np.argmin(ainten)]
            line_wav_em = line_wav + s
            print("    Abund:", a)
            print("    Shift:", s)
            print("    Chisq:", c2)
            print("    Doppler velocity:", _calc_vel(s, line_wav_em), "km/s")
            print("    Unshifted line max wavelength:", line_wav_em, "Angstrom")
            print("    Shifted line max wavelength:  ", line_wav, "Angstrom")
        print("")
_print_shifts()

print("FIX THE FITTING PROBLEM!!! SOMETIMES THE SHIFT DEPENDS ON OTHER LINES THEN THE ONE IN QUESTION, WHICH SHOULD NOT HAPPEN!!!\n*********\n")
if _MODE in {1, 2, 3}:
    print("GENERATING REGIONS WITH: create_regions_in" + str(_MODE))
elif _MODE == -1:
    print("REGIONS AND EVERYTHING COPIED FROM testshift.py")
else:
    print("REGIONS SPECIFIED MANALLY!!!")
print("INTERPOLATING OBSERVED:", _MODE_INTERP_OBS)

def _plot_region(region_nr):
    # Set the title to display the region number, the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Region: " + str(region_nr) +
              "   Mode: " + str(_MODE) +
              "   Fit spacing: " + str(_MODE_FIT_BEST_SPACING) +
              "   Interp obs: " + str(_MODE_INTERP_OBS) +
              "   Reg len: " + str(reg_length))

    
    # Setup the color list and get the region
    color_list = ["#FF0000", "#00FF00", "#FF00FF",
                  "#11FF0A", "#AAAA00", "#00AAAA",
                  "#AF009F", "#0F3FF0", "#F0FA0F",
                  "#A98765", "#0152A3", "#0FFFF0",
                  "#C0110C", "#0B0D0E", "#BDC0EB"]
    region_result = result.region_result[region_nr]

    # Get the observed spectrum contained in the region
    rwav, rinten = regs.get_region(region_result.region, wl, inten)
    
    # Plot the spectrum, followed by the synth lines
    plt.plot(rwav, rinten, color = "blue")
    for a in range(region_result.inten.shape[0]):
        plt.plot(region_result.wav, region_result.inten[a], color = color_list[a % len(color_list)])
        plt.plot(region_result.wav + region_result.shift, region_result.inten[a], color = color_list[a % len(color_list)], linestyle = "--", alpha = 0.5)
    
    # Show the plot
    plt.show()
#_plot_region(0)

def _plot_spec():
    # Set the title to display the mode and if the spacing between the datapoints in the synth region was fitted
    plt.title("Mode: " + str(_MODE) +
              "   Fit spacing: " + str(_MODE_FIT_BEST_SPACING) +
              "   Interp obs: " + str(_MODE_INTERP_OBS) +
              "   Reg len: " + str(reg_length))

    # Setup the color list    
    color_list = ["#FF0000", "#00FF00", "#FF00FF",
                  "#11FF0A", "#AAAA00", "#00AAAA",
                  "#AF009F", "#0F3FF0", "#F0FA0F",
                  "#A98765", "#0152A3", "#0FFFF0",
                  "#C0110C", "#0B0D0E", "#BDC0EB"]
    
    # Plot the entire observed spectrum
    plt.plot(wl, inten, color = "blue")

    # Plot the regions
    for r in result.region_result:
        for a in range(r.inten.shape[0]):
            print(r.shift)
            plt.plot(r.wav, r.inten[a], color = color_list[a % len(color_list)])
            plt.plot(r.wav + r.shift, r.inten[a], color = color_list[a % len(color_list)], linestyle = "--", alpha = 0.5)
    plt.show()
_plot_spec()

#def _print_vel(dlambda):
#    _, lambda_min = result.region_min()
#    for a in range(lambda_min.shape[0]):
#        print("*** Abundance nr", a, "***")
#        for dl, l_em in zip(dlambda, lambda_min[a,:]):
#            print("dl: ", dl, "\nl_obs:  ", l_em - dl, "\nVel:", dl*lightspeed/l_em, "\n")
#    print("NOTE: WHEN dl IS DIRECTLY SET TO dl = 0.021 THE SYNTHETIC CURVE IS NOTICABLY DISPLACED COMPARED TO THE ATLAS SPECTRA AND DOES NOT FIT!!!")
#    print("      AND WITH NOTICABLY I MEAN IT'S VISIBLE TO THE NAKED EYE (DEPENDING ON ZOOM LEVEL)!!!")
#    print("      SHOULD PROBABLY MAKE SURE I HAVE COMPARED TO THE CORRECT ATLAS SPECTAL LINES!!!")

