"""
This module contains functions for plotting observed and synthetic spectra.
"""

from __future__ import print_function
from __future__ import division

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

def plot_region(region_result, offset = 0.0, shifts = None, alpha = 0.75, alpha_best = 0.9, alpha_shifted = 0.25, show_observed = True, obs_pad = 0.0):
    """
    Plots the given region result.

        region_result : The result of an abundance fit for a region. This is an instance of ChiRegionResult or EWRegionResult
                        from the synther module.
                        
    The optional arguments are

        offset        : Determines the offset of the synthetic region. Specifically, it shifts the synthetic data.
                        Default is 0.

        shifts        : If not set to None, this plots a separate set of the synthetic data shifted by a given amount. If this is
                        a number, the synthetic data for every abundance is shifted by the same amount. If it is an iterable, that
                        iterable must have the same length as the amount of abundancies in region_result. In that case each individual
                        abundance will be shifted with the corresponding amount in shifts.
                        Default is None.
                         
        alpha         : The alpha of the synthetic data.
                        Default is 0.7.
                         
        alpha_best    : The alpha of the synthetic data corresponding to the best abundance.
                        Default is 0.9.

        alpha_shifted : The alpha of the shifted synthetic data. This only matters if shifts is not None.
                        Default is 0.25.
                         
        show_observed : Determines if the observed data should be shown in the plot.
                        Default is True.
                        
        obs_pad       : The padding of the observed data. Specifically, it can be used to show observed data from outside of the
                        region. A positive value expands the observed region shown while a negative value decreases it.
                        Default is 0.
    """
    
    # Make sure everything is shifted by the correct amount, if shifted data should be shown
    if shifts != None:
        shifts = shifts*np.ones(len(region_result.abund), dtype = np.float64) if np.isscalar(shifts) else list(shifts)
    
    # Set the title to display the region number, the mode and if the spacing between the datapoints in the synth region was fitted
    _plt.title("Interval: " + str((region_result.region.lambda0, round(region_result.region.lambda_end, 4))) +
              "  delta: " + str(region_result.region.dlambda) +
              "  steps: " + str(region_result.region.nlambda) +
              "  scale: " + str(region_result.region.scale_factor))

    # Plot the synthetic spectrum
    for a in range(region_result.inten.shape[0]):
        # Plot the unshifted spectrum
        # Checking this way to avoid future issues with numpy, which will make an elementwise check if we checked this as: shifts != None
        # Cannot check for just truthyness since numpy doesn't consider arrays to be "truthy" or "falsy".
        if None != shifts:
            _plt.plot(region_result.wav + shifts[a], region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = alpha_shifted, linestyle = "--")
        
        # Plot the shifted spectrum
        _plt.plot(region_result.wav - offset, region_result.inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = alpha if a != region_result.best_index else alpha_best)
    
    # Show the observed spectrum
    if show_observed:
        # Get the observed spectrum contained in the region
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

def plot_spec(region_results, show_observed = True, show_continuum = False, show_unshifted = False, padding = 0.0, plot_title = None):
    """
    Plots the spectrum that is coverd by the given region results. The required argument is

        region_results : An iterable of region results. Not that these results must be instances of ChiRegionResult.
        
    The optional arguments are

        show_observed  : Determines if the observed data should be shown in the plot.
                         Default is True.
        
        show_continuum : Determines if the continuum level should be shown. This continuum level is assumed to be the the continuum given
                         by the atlas.
                         Default is False.
                         
        show_unshifted : Determines if the unshifted synthetic data should be shown.
                         Default is False.
                         
        padding        : Determines how much excess data from the observed spectrum should be shown.
                         Default is 0.

        plot_title     : Sets the plot title.
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
    Plots a quantity against the abundance. The arguments are

        abund  : An iterable over abundencies. An individual abundance is expected to be in the same
                 form as if it was created with the function abundutils.abund.

        values : An iterable of values.
    """
    a = _au.list_abund(abund, default_val = -4.5)
    _plt.plot(a, values)
    _plt.show()

def plot_chisq(region_result):
    """
    Plots chi squared for the given region result vs the abundance.

        region_result : The region result from which the chi squared should be plotted. This will work
                        for an instance of ChiRegionResult, but not EWRegionResult.
    """
    plot_vs_abund(region_result.abund, region_result.chisq)

def plot_bisect(region_result, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, num = 50):
    """
    Plots the bisection of the given region result. It is possible to plot this for both synthetic and observed data. By default both are shown.

        region_result : The region result for the region where the bisection is plotted.
        
    The optional arguments are

        offset        : Offsets the synthetic spectrum. Positive values offsets it to the right while negative to the left.
                        Default is 0.
        
        plot_observed : Determines if the bisection of the observed spectrum should be shown.
                        Default is True.
        
        plot_synth    : Determines if the bisection of the synthetic spectrum should be shown.
                        Default is True.
        
        show_observed : Determines if the observed spectrum should be shown.
                        Default is True.
        
        show_synth    : Determines if the synthetic spectrum should be shown.
                        Default is True.
        
        num           : The amount of points for which the bisection should be calculated.
                        Default is 50.
    
    Note that at least one of plot_observed or plot_synth must be true. Otherwise an exception is raised.
    """
    
    if not (plot_observed or plot_synth):
        print("Must plot something")

    # Plot the bisection of the synthetic data    
    if plot_synth:
        # Get the wavelengths
        rwav = region_result.wav
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
    Plots the observed spectrum in the given interval. The required arguments are

        lambda0    : The starting wavelength of the interval.
        
        lambda_end : The final wavelength of the interval.
        
    Any additional parameters are passed on to the plotting function.
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

def plot_around(lambda_mid, delta, *args, **kwargs):
    """
    Plots the spectrum surrounding the given wavelength. The required arguments are

        lambda_mid : The wavelength around which the spectrum should be plotted.
        
        delta      : Sets the amount of wavelengths that should be included. Specifically, the
                     plotted spectrum lies between
                         lambda_mid - delta
                     and
                         lambda_mid + delta.
                         
    Any additional parameters are passed on to the plotting function.
    """
    plot_in(lambda_mid - delta, lambda_mid + delta, *args, **kwargs)

def plot_delta(y, x = None, *args, **kwargs):
    """
    Plots the changes in y between data points. Specifically, it plots the differences between the
    individual elements in y. The required argument is
    
        y : The data points for which to plot the differences between individual elements.
        
    There is an optional argument
        
        x : The points at which to plot the differences. If set to None, an interval from 0 to the
            length of y with steps of 1 will be used.
    
    Any additional parameters are passed on to the plotting function.
    """
    dy = y[1:] - y[:-1]
    if x == None:
        x = np.arange(len(dy))
    _plt.plot(x, dy, ".", *args, **kwargs)
    _plt.show()

