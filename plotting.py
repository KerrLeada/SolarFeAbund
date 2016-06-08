from __future__ import print_function

import numpy as np
import matplotlib.pyplot as _plt
import abundutils as _au
import satlas as _sa
import bisect

# Get the atlas
_at = _sa.satlas()

# A list of colors to use in plots
plot_color_list = ["#FF0000", "#00FF00", "#FF00FF",
                   "#11FF0A", "#AAAA00", "#00AAAA",
                   "#AF009F", "#0F3FF0", "#F0FA0F",
                   "#A98765", "#0152A3", "#0FFFF0",
                   "#C0110C", "#0B0D0E", "#BDC0EB"]

def plot_region(region_result, offset = 0.0, alpha = 0.75, show_observed = True, show_unshifted = False, obs_pad = 0.0):
    """
    Plots the given region result.
    """
    # Set the title to display the region number, the mode and if the spacing between the datapoints in the synth region was fitted
    _plt.title("Interval: " + str((region_result.region.lambda0, round(region_result.region.lambda_end, 4))) +
              "  delta: " + str(region_result.region.dlambda) +
              "  steps: " + str(region_result.region.nlambda) +
              "  scale: " + str(region_result.region.scale_factor))

    # Plot the synthetic spectrum
    for a in range(region_result.inten.shape[0]):
        # Unshifted
        if show_unshifted:
            _plt.plot(region_result.wav + region_result.shift[a], region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.25, linestyle = "--")
        
        # Shifted
        _plt.plot(region_result.wav - offset, region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = alpha)
    
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
        _plt.plot(rwav, rinten, color = "blue")
    
    # Show the plot
    _plt.show()

def plot_spec(region_results, show_observed = True, show_continuum = False, show_unshifted = False, padding = 0.0, plot_title = None, cgs = True):
    """
    Plots the spectrum that is coverd by the given region results.
    """

    # Set the plot title
    if plot_title:
        _plt.title(plot_title)

    # Plot the regions
    for r in region_results:
        for a in range(r.inten.shape[0]):
            # Unshifted
            if show_unshifted:
                _plt.plot(r.wav + r.shift[a], r.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.25, linestyle = "--")
            
            # Shifted
            _plt.plot(r.wav, r.inten[a], color = plot_color_list[a % len(plot_color_list)])
    
    if show_observed or show_continuum:
        # Find the interval of the spectrum
        # This is rather inefficient but it is unlikely there will be enough regions for this to matter
        min_wl = min([r.region.lambda0 for r in region_results]) - padding
        max_wl = max([r.region.lambda_end for r in region_results]) + padding
        wl, inten, cont = _at.getatlas(min_wl, max_wl, cgs = cgs)
        
        # Normalize using the continuum
        inten /= cont
        
        # Plot the entire observed spectrum
        if show_observed:
            _plt.plot(wl, inten, color = "blue", alpha = 0.5)
        
        # Plot the continuum level of the atlas
        if show_continuum:
            _plt.plot(wl, cont, color = "blue", linestyle = "--", alpha = 0.5)
    
    # Show everything
    _plt.show()

# Plots chi squared of a region result
def plot_vs_abund(abund, values):
    """
    Plots the quantity in values vs the abundance
    """
    a = _au.list_abund(abund, default_val = -4.5)
    _plt.plot(a, values)
    _plt.show()

def plot_chisq(region_result):
    """
    Plots chi squared for the given region result vs the abundance
    """
    plot_vs_abund(region_result.abund, region_result.chisq)

def plot_bisect(region_result, abund_filter = None, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    """
    Plots the bisection of the given region result. It is possible to plot this for both synthetic and observed data. By default both are shown.
    """
    
    if not (plot_observed or plot_synth):
        print("Must plot something")

    # Plot the bisection of the synthetic data    
    if plot_synth:
        # Get the wavelengths
        rwav = region_result.wav
        
        # Filter the abundances (if needed)
        if hasattr(abund_filter, "__call__"):
            rinten_all = abund_filter(region_result.inten)
        elif abund_filter != None:
            rinten_all = region_result.inten[abund_filter]
        else:
            rinten_all = region_result.inten
        
        # Plot the bisections
        for a, rinten in enumerate(rinten_all):
            bwav, binten = bisect.get_bisection(rwav, rinten, num = num)
            if show_synth:
                _plt.plot(rwav - offset, rinten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.4, linestyle = "--")
            _plt.plot(bwav - offset, binten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.8)
    
    # Plot the bisection of the observed data
    if plot_observed:
        rwav = region_result.region.wav
        rinten = region_result.region.inten
        bwav, binten = bisect.get_bisection(rwav, rinten, num = num)
        if show_observed:
            _plt.plot(rwav, rinten, color = "blue", alpha = 0.75, linestyle = "--")
        _plt.plot(bwav, binten, color = "blue")

    _plt.show()

def plot_in(lambda0, lambda_end, *args, **kwargs):
    """
    Plots the observed spectra between lambda0 and lambda_end.
    """

    # Get the wavelength, intensity and continuum
    wav, intensity, cont = _at.getatlas(lambda0, lambda_end, cgs = True)
    
    # Normalize the intensity if a keyword agrument "normalize" was present and true, or
    # if the keyword argument was missing
    normalize = kwargs.pop("normalize", True)
    if normalize:
        intensity /= intensity.max()

    # Plot the spectrum
    _plt.plot(wav, intensity, *args, **kwargs)
    _plt.show()

def plot_around(lambda_mid, length, *args, **kwargs):
    """
    Plots around the mid wavelenth
    """
    plot_in(lambda_mid - length, lambda_mid + length, *args, **kwargs)

def plot_delta(y, x = None, *args, **kwargs):
    """
    Plots the changes in y between data points.
    """
    dy = y[1:] - y[:-1]
    if x == None:
        x = np.arange(len(dy))
    _plt.plot(x, dy, ".", *args, **kwargs)
    _plt.show()

