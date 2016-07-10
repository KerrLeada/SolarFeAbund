# -*- coding: utf8 -*-

"""
This module contains functions for plotting observed and synthetic spectra.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as _plt
import matplotlib.ticker as ticker
import matplotlib.colors
import abundutils as _au
import satlas as _sa
import scipy.interpolate as si
import bisect
import astropy.units

def plot_stuff(result_pair):
    """
    This function plots arbitrary stuff. It is not always constant what it does, since
    it might be edited. It takes a single argument
    
        result_pair : An instance of ResultPair that contains the result of a calculation.
    """
    
    result_chi = result_pair.result_chi
    result_ew = result_pair.result_ew
    regions = [r.region for r in result_chi.region_result]
    
    # *** Plot the mosaic of the observed lines in the regions
    if False:
        plot_region_mosaic(regions, 3, 5, figsize = (6, 5))
    
    # *** Plots the effects of macroturbulance... specifically it shows up the effects of convolving the synthetic spectrum
    if False:
        plot_macroturb(result_chi.region_result[3], xticks = 5, yticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], figsize = (4, 3))
    
    # *** Plots the effects of shifts and how the chi squared changes with shifts, for a particular region
    if False:
        pltshift = partial(plot_shifted, xticks = 3, xlim = (5778.41, 5778.49), ylim = (0.67, 0.92))
        pltchisq = plot_vs(lambda r: (r.shift_all, r.chisq_all[r.best_index]), xlabel = u"Shift [Å]", ylabel = "$\\chi^2$", xlim = (-0.10, 0.10))
        plot_row(result_chi.region_result[5], [pltshift, pltchisq], figsize = (6, 3))
    
    # *** Plots several abundances for a region and how chi squared changes with abundance, for that region... the region is the line at approximately 5778 Å
    if False:
        pltshift = partial(plot_region, show_abunds = True, abund_filter = [7, 490], xticks = 3, xlim = (5778.33, 5778.59), ylim = (0.3, 1.1))
        pltchisq = plot_vs(lambda r: (_abund(r.abund), r.chisq), xlabel = "$" + _LOG_ABUND_CONV + "$", ylabel = "$\\chi^2$", xlim = (7.2, 7.7), xfmt = "%0.1f")
        plot_row(result_chi.region_result[5], [pltshift, pltchisq], figsize = (6, 3))
    
    # *** Plots several abundances for a region and how chi squared changes with abundance, for that region... the region is the line at approximately 5232 Å
    if False:
        pltshift = partial(plot_region, show_abunds = True, abund_filter = [7, 490], xlim = (5232.17, 5234.16), xticks = 3)
        pltchisq = plot_vs(lambda r: (_abund(r.abund), r.chisq), xlabel = "$" + _LOG_ABUND_CONV + "$", ylabel = "$\\chi^2$", xlim = (7.2, 7.7), xfmt = "%0.1f")
        plot_row(result_chi.region_result[-1], [pltshift, pltchisq], figsize = (6, 3))
    
    # *** Plots a close up on what happens for different abundances for the strong line at approximately 5232 Å
    if False:
        _plt.figure(figsize = (6,3.5))
        plot_region(result_chi.region_result[-1], show_abunds = True, abund_filter = [0, 58, 201, 309, 499], xlim = (5232.89, 5233.01), ylim = (0.165, 0.192))
    
    # *** Histogram over the distribution of best shifts among the results from the regions
    # Using the inbuilt histogram plotting function
    if False:
        _plt.figure(figsize = (6, 2.8))
        _plt.hist([r.best_shift for r in result_chi.region_result], bins = 15)
        _plt.xlim(-0.0022, 0.0102)
        _plt.xlabel(u"Shift [Å]", fontsize = plot_font_size)
        _plt.ylabel("Number of lines", fontsize = plot_font_size)
        _plt.tight_layout()
        _plt.show()
    # Using a custom histogram plotting function
    if False:
        _plt.figure(figsize = (5, 2.8))
        plot_hist([1e3*r.best_shift for r in result_chi.region_result], bin_width = 1, xlabel = u"Shift [mÅ]", ylabel = "Number of lines")
    
    # *** Histogram over the distribution of the absolute value of the difference in equivalent width between the synthetic and observed lines
    if False:
        _plt.figure(figsize = (5, 2.8))
        _plt.hist([1e3*abs(r.best_diff) for r in result_ew.region_result])
        _plt.xlabel(u"$| \\Delta W |$ [mÅ]")
        _plt.ylabel("Number of lines")
        _plt.yticks([0, 1, 2, 3, 4])
        _plt.tight_layout()
        _plt.show()
    
    # *** Plots histograms over the distributions of abundances for the results obtained using chi squared and equivalent widths
    if False:
        bins = 10
        f, ax = _plt.subplots(1,2, figsize = (6,2.8))
        
        ax[0].hist(_abund(result_chi.best_abunds), bins = bins)
        ax[0].set_yticks([0,1,2,3])
        ax[0].set_xlim(7.33, 7.55)
        ax[0].set_xlabel("$" + _LOG_ABUND_CONV + "$")
        ax[0].set_ylabel("Number of lines")
        
        ax[1].hist(_abund(result_ew.best_abunds), bins = bins)
        ax[1].set_yticks([0,1,2,3])
        ax[1].set_xlim(7.38, 7.65)
        ax[1].set_xlabel("$" + _LOG_ABUND_CONV + "$")
        ax[1].set_ylabel("Number of lines")
        
        _plt.tight_layout()
        _plt.show()
    
    # *** Plots the line at approximately 5323 Å (happens to be the strongest line of the ones that has been used)
    if False:
        _plt.figure(figsize = (6,3))
        plot_region(result_chi.region_result[-1], obs_last = True, xticks = 7)
    
    # *** Plots the line at approximately 5778 Å
    if False:
        _plt.figure(figsize = (6,3))
        plot_region(result_chi.region_result[5], obs_last = True, xticks = 7)
    
    # *** Plots the line at approximately 5705 Å
    if False:
        _plt.figure(figsize = (6,3))
        plot_region(result_chi.region_result[11], obs_last = True, xticks = 7)
    
    # *** Plots the first 8 lines together
    if False:
        plot_mosaic(result_chi.region_result[:8], 4, 2, partial(plot_region, xticks = 3), figsize = (6,9))
    
    # *** Plots the rest of the lines together
    if False:
        def plotfunc(region_result, figure_axes = None):
            #if np.array_equal(region_result.region.wav, result_chi.region_result[-1].region.wav):
            if region_result is result_chi.region_result[-1]:
                plot_region(region_result, xticks = 3, xlim = (5232.13, 5234.12), figure_axes = figure_axes)
            else:
                plot_region(region_result, xticks = 3, figure_axes = figure_axes)
        plot_mosaic(result_chi.region_result[8:], 4, 2, plotfunc, figsize = (6,9))
    
    # *** Plots the bisector of the line at approximately 6302 Å (both for the observed and synthetic spectra)
    if False:
        regres = result_chi.region_result[1]
        f, ax = _plt.subplots(nrows = 1, ncols = 2, figsize = (6,3))
        plot_bisector(regres.region.wav, regres.region.inten, xy_args = options(color = "blue"), xticks = 3, xlabel = u"Wavelength [Å]", ylabel = "Normalized intensity", xfmt = "%0.2f", figure_axes = ax[0], color = "blue")
        plot_bisector(regres.wav, regres.best_inten, xy_args = options(color = "red"), xticks = 3, xlabel = u"Wavelength [Å]", ylabel = "Normalized intensity", xfmt = "%0.2f", figure_axes = ax[1], color = "red")
        _plt.tight_layout()
        _plt.show()
    
    # *** Plots the bisector of the line at approximately 6302 Å (only for the observed spectra)
    if False:
        line_nr = 1
        rwav = result_chi.region_result[line_nr].region.wav
        rinten = result_chi.region_result[line_nr].region.inten
        f, ax = _plt.subplots(nrows = 1, ncols = 2, figsize = (6,3))
        plot_bisector(rwav, rinten, xy_args = options(color = "blue"), xticks = 3, xlabel = u"Wavelength [Å]", ylabel = "Normalized intensity", xfmt = "%0.2f", xlim = (rwav[0], rwav[-1]), ylim = (0.0, 1.1), figure_axes = ax[0], color = "blue")
        plot_bisector(rwav, rinten, xy_args = options(color = "blue"), xticks = 3, xlabel = u"Wavelength [Å]", ylabel = "Normalized intensity", xfmt = "%0.2f", xlim = (6302.48, 6302.50), ylim = (0.3, 1.0), figure_axes = ax[1], color = "blue")
        _plt.tight_layout()
        _plt.show()
    
    # *** Plots the differences in abundance between when chi squared is used and when equivalent widths are used
    if False:
        # Calculate the differences in abundance between when chi squared is used and when equivalent widths are used 
        data_unordered = [(r_chi.region.estimate_minimum(), r_chi.best_abund - r_ew.best_abund, r_ew, r_chi) for r_chi, r_ew in zip(result_chi.region_result, result_ew.region_result)]
        data_diffs = sorted(data_unordered, key = lambda x: x[0])
        rwav, abund_diff, _, _ = map(np.array, zip(*data_diffs))
        
        # Plot the differences
        _plt.figure(figsize = (4, 3))
        _plt.plot(rwav, abund_diff, ".")
        _plt.xlabel(u"Wavelength [Å]", fontsize = plot_font_size)
        _plt.ylabel("$(" + _LOG_ABUND_CONV + ")_{\\chi^2} - (" + _LOG_ABUND_CONV + ")_{EW}$", fontsize = plot_font_size)
        _plt.xlim(rwav[0] - 50, rwav[-1] + 50)
        _plt.xticks(np.linspace(rwav[0], rwav[-1], num = 6))
        _plt.tight_layout()
        _plt.show()
    
    # *** Show the difference in abundance between the best synthetic lines obtained by chi squared and equivalent widths
    if True:
        # Get the data
        data_unordered = [(r_chi.best_abund - r_ew.best_abund, r_chi, r_ew) for r_chi, r_ew in zip(result_chi.region_result, result_ew.region_result)]
        abund_diffs, region_result_chi, region_result_ew = zip(*sorted(data_unordered, key = lambda x: x[0]))
        regres_chi = region_result_chi[0]
        regres_ew = region_result_ew[0]
        region = regres_chi.region
        
        # Plot the difference
        _plt.figure(figsize = (4,3))
        lbl_obs = _plt.plot(region.wav, region.inten, color = "blue", linestyle = "--", label = "FTS atlas")
        lbl_chi = _plt.plot(regres_chi.wav, regres_chi.best_inten, color = "red", label = "$\\chi^2$")
        lbl_ew = _plt.plot(regres_ew.wav - regres_chi.best_shift, regres_ew.best_inten, color = "green", label = "$EW$")
        legend_labels = [lbl_obs[0], lbl_chi[0], lbl_ew[0]]
        wav_min, wav_max = min([region.wav[0], regres_chi.wav[0], regres_ew.wav[0]]), max([region.wav[-1], regres_chi.wav[-1], regres_ew.wav[-1]])
        _plt.xlim(wav_min, wav_max)
        _plt.ylim(0.45, 1.05)
        _plt.xticks(np.linspace(wav_min, wav_max, num = 4))
        _plt.xlabel(u"Wavelength [Å]", fontsize = plot_font_size)
        _plt.ylabel("Normalized intensity", fontsize = plot_font_size)
        _plt.legend(handles = legend_labels, loc = 4, frameon = False, fontsize = legend_font_size)
        _plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))
        _plt.tight_layout()
        _plt.show()
    
    # *** 
    if False:
        region_result = result_chi.region_result
        plot_row(region_result[14], [partial(plot_region, show_abunds = True, abund_filter = [127, 350, 490], xticks = 4), plot_vs(lambda r: (_abund(r.abund), r.chisq), xlabel = "$" + _LOG_ABUND_CONV + "$", ylabel = "$\\chi^2$", xticks = np.array([7.2 ,  7.35,  7.5 ,  7.65,  7.8]))], figsize = (7,3.5))

# Get the atlas
_at = _sa.satlas()

# A list of colors to use in plots
plot_color_list = ["#AA0000", "#008800", "#AA00AA",
                   "#519A2A", "#AAAA00", "#00AAAA",
                   "#AF009F", "#0F3FF0", "#F0F50F",
                   "#A98765", "#0152A3", "#0FAAF0",
                   "#C0110C", "#0B0D0E", "#BDC0EB"]

# Set the font size of the plots
plot_font_size = 11
title_font_size = plot_font_size
legend_font_size = 8

# Set the font to the default
_plt.rc("font", **{"family": u"sans-serif", u"sans-serif": [u"Helvetica"]})
_plt.rcParams.update({"font.size": plot_font_size})

# The figure size (in inches) of some functions
plot_figure_size = (7, 7)

# Determines the notation used for the abundance
_LOG_ABUND_CONV = "\\log A"

def _log_abund_str(value):
    """
    Returns a string stating what the abundance is
    """
    
#    value = "{:0.3f}".format(value)
    value = str(_abund(value))
    return "$" + _LOG_ABUND_CONV + "=" + value + "$"

def estimate_minimum(wav, inten, num = 1000):
    """
    Estimates the minimum intensity using quadratic interpolation. The optional argument is
    
        num : The number of points to use when estimating the minimum.
              Default is 1000.
    """
    
    tck = si.splrep(wav, inten)
    wav = np.linspace(wav[0], wav[-1], num = 1000)
    inten = si.splev(wav, tck)
    return wav[inten == min(inten)][0]

def _calc_ticks(ticks, values):
    if isinstance(ticks, int):
        ticks = np.linspace(min(values), max(values), num = ticks)
    elif hasattr(ticks, "__call__"):
        ticks = ticks(values)
    return ticks

def _abund(abund):
    """
    Determines the convension for abundance numbers
    """
    
    return 12.0 + abund

def partial(func, *args, **kwargs):
    """
    Partially applies the given function
    """
    
    def partial_application(*args2, **kwargs2):
        argums = args + args2
        keywargs = kwargs.copy()
        keywargs.update(kwargs2)
        return func(*argums, **keywargs)
    return partial_application

def figured(plotting_func, *fig_args, **fig_kwargs):
    """
    Returns a function that creates a figure with the given arguments and then calls
    the given plotting functions. The required argument is
    
        plotting_func : A function that plots something.
    
    All other arguments to this functions are arguments to the figure, specifically they
    are passed on to matplotlib.pyplot.figure.
    
    Any arguments given to the returned function are passed on to plotting_func.
    """
    
    def plotter(*args, **kwargs):
        _plt.figure(*fig_args, **fig_kwargs)
        plotting_func(*args, **kwargs)
    return plotter

def _set_title(ax, title):
    """
    Sets the title, and makes sure the font size is correct
    """
    
    ax.set_title(title, fontsize = title_font_size)

def options(*args, **kwargs):
    return args, kwargs

def _get_figure_axes(figure_axes):
    """
    Gets the axis object used to plot stuff. Specifically, if figure_axes is None the current
    axis object will be returned. Otherwise, figure_axes will be returned.
    """
    
    if figure_axes == None:
        figure_axes = _plt.gca()
    return figure_axes

def _adjust_ticks(setter, ax, ticks, curr_ticks, limits, error_message):
    if isinstance(ticks, int) and ticks > 0:
        if None == limits:
            limits = (curr_ticks[0], curr_ticks[-1])
        setter(np.linspace(limits[0], limits[1], num = ticks))
    elif hasattr(ticks, "__call__"):
        if None != limits:
            curr_ticks = curr_ticks[(limits[0] <= curr_ticks) & (curr_ticks <= limits[1])]
        setter(ticks(curr_ticks))
    elif None != ticks:
        try:
            setter(ticks)
        except:
            raise Exception(error_message(ticks))

def _adjust_xyticks(ax, xticks, yticks, xlimits = None, ylimits = None):
    _adjust_ticks(ax.set_xticks, ax, xticks, ax.get_xticks(), xlimits, lambda ticks: "Illegal value for argument 'xticks'. It must be None, a positive integer, a function or a list of xticks to use, but it had type " + type(ticks).__name__ + " and value " + str(ticks))
    _adjust_ticks(ax.set_yticks, ax, yticks, ax.get_yticks(), ylimits, lambda ticks: "Illegal value for argument 'yticks'. It must be None, a positive integer, a function or a list of yticks to use, but it had type " + type(ticks).__name__ + " and value " + str(ticks))

def _filter_abund(abund_filter, abund, inten):
    """
    Filters out the intensities using the given abundance filter abund_filter, the abundencies abund and
    the intensities inten (which is an array of 2 dimensions, for which the rows represent abundencies and
    the columns the corresponding values for the intensity). A new numpy array of the same format as inten
    is returned.
    """
    
    # Filter out abundances
    if None != abund_filter:
        if hasattr(abund_filter, "__call__"):
            filtered = np.array([[a, i] for ai, (a, i) in enumerate(zip(abund, inten)) if abund_filter(ai, a, i)])
            abund = filtered[:,0]
            inten = filtered[:,1]
        else:
            abund = abund[abund_filter]
            inten = inten[abund_filter]
    return abund, inten

def plot_bisector(x, y, num = 50, xticks = None, yticks = None, xlim = None, ylim = None, xlabel = None, ylabel = None, xfmt = None, yfmt = None, plot_values = True, xy_args = None, show = True, figure_axes = None, **kwargs):
    """
    Plots a bisector given by x and y. Required arguments
    
        x : The x values.
        
        y : The y values.
    
    The optional arguments are
    
        num          : The number of points in the bisector.
                       Default is 50.
        
        xticks       : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        yticks       : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        xlim         : Sets the limits of the x axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                       the first element in the tuple is the minimum x value shown and the second element is the maximum x value shown.
                       Default is None.
        
        ylim         : Sets the limits of the y axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                       the first element in the tuple is the minimum y value shown and the second element is the maximum y value shown.
                       Default is None.
        
        xlabel       : If not None, sets the label of the x axis.
                       Default is None.
        
        ylabel       : If not None, sets the label of the y axis.
                       Default is None.
        
        xfmt         : If not None, sets how the numbers on the x axis are formatted. This is expected to be a string that is given to a
                       matplotlib.ticker.FormatStrFormatter object. See matplotlib.ticker.FormatStrFormatter for more information about how this
                       works.
                       Default is None.
        
        yfmt         : If not None, sets how the numbers on the y axis are formatted. This is expected to be a string that is given to a
                       matplotlib.ticker.FormatStrFormatter object. See matplotlib.ticker.FormatStrFormatter for more information about how this
                       works.
                       Default is None.
        
        plot_values  : Determines if the x and y values should be plotted as well as the bisector. True means they are plotted, False that they
                       are not.
                       Default is True.
        
        xy_args      : If not None, this contains the arguments for the plot of the values. Specifically it is a tuple with 2 elements. The first
                       element is a list and the second a dict. These are the arguments passed on to the function call that plots the x and y values,
                       if plot_values is set to True. If plot_values is False, this argument does nothing.
                       Default is None.
        
        show         : Determines, assuming figure_axes is None, if the plot should be shown directly. If figure_axes is None, the plot is never
                       shown directly.
                       Default is True.
        
        figure_axes  : Sets the axes object. If this is None, then the result of matplotlib.pyplot.gca() will be used. And if this is not None
                       then it will be used to plot the abundance. Also note that if this is not None, the plot will not be shown implicitly. Thereby
                       this can be used to have several plots in the same figure.
                       Default is None.
    
    Any additional keyword arguments are passed on to matplotlib.pyplot.plot.
    
    By default, if the plot_values is True, then the plotted x and y values will have the line style "--". This can be set using the xy_args argument.
    
    The return value is a list of the lines that where added.
    """
    
    # If no axes object was given, get the current one from _plt
    if figure_axes == None:
        ax = _plt.gca()
    else:
        ax = figure_axes

    # Plot the bisector    
    bx, by = bisect.get_bisector(x, y, num = num)
    if plot_values:
        if xy_args != None:
            args_vals, kwargs_vals = xy_args
            if "linestyle" not in kwargs_vals:
                kwargs_vals["linestyle"] = "--"
            plotted = ax.plot(x, y, *args_vals, **kwargs_vals)
        else:
            plotted = ax.plot(x, y, linestyle = "--")
        plotted_bisect = ax.plot(bx, by, **kwargs)
        plotted.extend(plotted_bisect)
    else:
        plotted = ax.plot(bx, by, **kwargs)

    # Set the formattings of the x and y axis
    if xfmt != None:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(xfmt))
    if yfmt != None:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(yfmt))
    
    # Adjust the ticks for the x and y axes
    _adjust_xyticks(ax, xticks, yticks, xlimits = xlim, ylimits = ylim)
    
    # Set the limits and labels
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)

    # If show is true and no axis object was given, show the plot    
    if show and figure_axes == None:
        _plt.tight_layout()
        _plt.show()
    
    # Return the plotted lines
    return plotted

def plot_compared(region_result, show_labels = True, abund_filter = None, figure_axes = None):
    """
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    wav = region_result.region.wav
    intensities = _filter_abund(abund_filter, region_result.abund, region_result.inten)
    
    for a, inten in enumerate(intensities):
        if not np.all(inten == region_result.best_inten):
            ax.plot(wav, inten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.5)
    ax.plot(wav, region_result.best_inten, color = "red")
    ax.plot(wav, region_result.region.inten, color = "blue")

    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()
        

def plot_abund_compared(region_result, abund = None, show_labels = True, show_legend = True, legend_pos = 4, figure_axes = None):
    """
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    if abund == None:
        abund = region_result.best_index
    
    obs_wav = region_result.region.wav
    synth_wav = region_result.wav
    inten = region_result.inten[abund]
    
    lbl_comp = ax.plot(obs_wav, inten, color = "red", label = "Comp")
    lbl_real = ax.plot(synth_wav, inten, color = "red", linestyle = "--", label = "Real")
    lbl_obs = ax.plot(obs_wav, region_result.region.inten, color = "blue", label = "Obs")
    
    if show_legend:
        ax.legend(handles = [lbl_comp[0], lbl_real[0], lbl_obs[0]], loc = legend_pos, fontsize = legend_font_size, frameon = False)
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_abund_compared2(region_result, linear_interp = False, abund = None, show_labels = True, show_legend = True, legend_pos = 4, figure_axes = None):
    """
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    if abund == None:
        abund = region_result.best_index
    
    obs_wav = region_result.region.wav
    synth_wav = region_result.wav
    inten_real = region_result.inten[abund]
    
    if linear_interp:
        inten_interp = np.interp(obs_wav, synth_wav, inten_real)
    else:
        tck = si.splrep(synth_wav, inten_real)
        inten_interp = si.splev(obs_wav, tck, der = 0)
    
    lbl_comp = ax.plot(obs_wav, inten_interp, color = "red", label = "Comp")
    lbl_real = ax.plot(synth_wav, inten_real, color = "red", linestyle = "--", label = "Real")
    lbl_obs = ax.plot(obs_wav, region_result.region.inten, color = "blue", label = "Obs")
    
    if show_legend:
        ax.legend(handles = [lbl_comp[0], lbl_real[0], lbl_obs[0]], loc = legend_pos, fontsize = legend_font_size, frameon = False)
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_shifted(region_result, show_labels = True, show_legend = True, legend_pos = 4, obs_pad = 0.0, abund_filter = None, xlim = None, ylim = None, xticks = None, yticks = None, figure_axes = None):
    """
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    # Get the wavelengths
    wav = region_result.wav
    
    # Make sure entire y scale is shown
#    ax.set_xlim([wav[0], wav[-1]])
#    ax.set_ylim([0, 1.1])
    
    # Plot the unshifted, shifted and observed spectrums
    lbl_obs = ax.plot(region_result.region.wav, region_result.region.inten, color = "blue", label = "FTS atlas")
    lbl_best = ax.plot(wav, region_result.best_inten, color = "red", label = "Shifted")
    lbl_shifted = ax.plot(wav + region_result.best_shift, region_result.best_inten, color = "red", linestyle = "--", label = "Original")

    # Set the formatter
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))
    
    # Make sure there are always limits
    if None == xlim:
        xlim = (min(wav[0], region_result.region.wav[0]), max(wav[-1], region_result.region.wav[-1]))
    if None == ylim:
        ylim = (0.0, 1.1)
    
    # Adjust the x ticks
    _adjust_xyticks(ax, xticks, yticks, xlimits = xlim, ylimits = ylim)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if show_legend:
        ax.legend(handles = [lbl_best[0], lbl_shifted[0], lbl_obs[0]], loc = legend_pos, fontsize = legend_font_size, frameon = False)
    if figure_axes == None:
        _plt.tight_layout()
        _plt.show()

def _create_bins(values, delta):
    bins = []
    bin_content = set()
    
    # Loop through each value
    for v in values:
        # If the value has not already been added to the bins...
        if v not in bin_content:
            # Add the value to the bins
            bins.append(v)
            bin_content.add(v)
            
            # Add any other value that is close enough and not already in the bins to the bins
            for val in values:
                if val not in bin_content and abs(v - val) <= delta:
                    bin_content.add(v)
    return bins

def plot_hist(values, delta = 0.0, bins = None, bin_width = None, bin_comparator = None, xlabel = None, ylabel = None, xticks = None, yticks = None, figure_axes = None):
    """
    Plots a histogram. The required argument is
    
        values : The values to plot.
        
    The optional arguments are
    
        delta       : Determines how far away two values can be while being in the same bin.
                      Default is 0.
        
        bins        : If not None, sets the bins in which to place the different values.
                      Default is None.
        
        bin_width   : If not None, sets the width of the bars in the histogram. Can be a number or a function. If it is a function it takes
                      a single required argument, which is the bins. If None, the width of the bins will be calculated internally instead.
                      Default is None.
        
        xlabel      : The label for the y axis. If set to None, no such label is shown.
                      Default is None.
        
        ylabel      : The label for the y axis. If set to None, no such label is shown.
                      Default is None.
        
        xticks       : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        yticks       : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        figure_axes  : Sets the axes object. If this is None, then the result of
                       matplotlib.pyplot.gca() will be used. And if this is not None
                       then it will be used to plot the abundance. Also note that
                       if this is not None, the plot will not be shown implicitly.
                       Thereby this can be used to have several plots in the same figure.
                       Default is None.
    """
    
    if len(values) == 0:
        raise Exception("Need at least 1 value, but 'values' was empty.")

    ax = _get_figure_axes(figure_axes)
    
    if None == bins:
        bins = _create_bins(values, delta)
    
    content = np.zeros(len(bins), dtype = np.float64)
    for i, b in enumerate(bins):
        for v in values:
            if abs(v - b) <= delta:
                content[i] += 1
    bin_data = sorted(zip(bins, content), cmp = bin_comparator, key = lambda bd: bd[0])
    
    bins, content = map(np.array, zip(*bin_data))
    
    # Set the bin width
    if bin_width == None:
        bin_width = min(bins[1:] - bins[:-1]) if len(bins) > 1 else 1
    elif hasattr(bin_width, "__call__"):
        bin_width = bin_width(bins)
    
    # Plot the bars
    for b, c in bin_data:
        ax.bar(b - bin_width/2.0, c, width = bin_width, bottom = 0)

    ax.set_xlim(bins[0] - 3.0*bin_width/4.0, bins[-1] + 3.0*bin_width/4.0)
    ax.set_ylim(0, max(content))

    if None != xticks:
        ax.set_xticks(_calc_ticks(ticks))
    if None != yticks:
        ax.set_yticks(_calc_ticks(ticks))
    if xlabel != None:
        ax.set_xlabel(xlabel, fontsize = plot_font_size)
    if ylabel != None:
        ax.set_ylabel(ylabel, fontsize = plot_font_size)
    if figure_axes == None:
        _plt.tight_layout()
        _plt.show()

def _plot_obs(region_result, obs_pad, ax):
    # Get the observed spectrum contained in the region
    if obs_pad == (0.0, 0.0):
        rwav = region_result.region.wav
        rinten = region_result.region.inten
    else:
        lambda0 = region_result.region.lambda0 - obs_pad[0]
        lambda_end = region_result.region.lambda_end + obs_pad[1]
        
        # Get the wavelengths and intensities
        rwav, rinten, cont = _at.getatlas(lambda0, lambda_end, cgs = True)
        rinten /= region_result.region.inten_scale_factor
    
    # Plot the observed spectrum, followed by the synth lines
    return ax.plot(rwav, rinten, color = "blue", label = "FTS atlas"), rwav
    

def plot_region(region_result, offset = 0.0, show_abunds = False, show_labels = True, show_legend = True, legend_pos = 4, obs_pad = 0.0, obs_last = True, abund_filter = None, xticks = None, yticks = None, xlim = None, ylim = None, figure_axes = None):
    """
    Plots the given region result.

        region_result : The result of an abundance fit for a region. This is an instance of ChiRegionResult or EWRegionResult
                        from the synther module.
                        
    The optional arguments are

        offset       : Determines the offset of the synthetic region. Specifically, it shifts the synthetic data.
                       Default is 0.
        
        show_abunds  : Determines if the synthetic data for other abundances then the best abundance should be shown. If set to True,
                       the abundance filter can be used to select specific abundances to show or not. The abundance filter is given
                       through abund_filter.
                       Default is False.
        
        show_labels  : Determines if the labels for the axes should be shown or not.
                       Default is True.
        
        show_legend  : Determines if the legend should be shown.
                       Default is True.
        
        legend_pos   : Determines the position of the legend, if it is shown. Valid values are
                       
                           0  : best
                           1  : upper right
                           2  : upper left
                           3  : lower left
                           4  : lower right
                           5  : right
                           6  : center left
                           7  : center right
                           8  : lower center
                           9  : upper center
                           10 : center
                       
                       Alternatively a 2 element tuple can be used to specify the x and y position (first element
                       is x, second element is y) of the lower left corner of the legend. This position has to be
                       in the coordinates of the plot, so x is wavelength and y is the normalized intensity.
                       Default is 4.
                        
        obs_pad      : The padding of the observed data. Specifically, it can be used to show observed data from outside of the
                       region. A positive value expands the observed region shown while a negative value decreases it.
                       Default is 0.
        
        obs_last     : If True, the observed spectrum is drawn first. Otherwise the observed spectrum is drawn last.
                       Default is True.
        
        abund_filter : A filter that determines which abundances should be shown. It can be None, a function of anything that a numpy
                       array can be indexed or sliced with. If it is None, nothing is filtered out. If it is a function, it is expected
                       to take 3 arguments. The first is the abundance index, the second is the abundance and the third is the synthetic
                       intensities for that abundance.
                       Default is None.
        
        xticks       : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        yticks       : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                       ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                       array of filtered ticks. And if None is used, nothing will happen.
                       Default is None.
        
        xlim         : Sets the limits of the x axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                       the first element in the tuple is the minimum x value shown and the second element is the maximum x value shown.
                       Default is None.
        
        ylim         : Sets the limits of the y axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                       the first element in the tuple is the minimum y value shown and the second element is the maximum y value shown.
                       Default is None.
        
        figure_axes  : Sets the axes object. If this is None, then the result of
                       matplotlib.pyplot.gca() will be used. And if this is not None
                       then it will be used to plot the abundance. Also note that
                       if this is not None, the plot will not be shown implicitly.
                       Thereby this can be used to have several plots in the same figure.
                       Default is None.
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    # Make sure the padding for the observable spectrum is a 2 element tuple contining
    # the padding on both the left (first element) and right (secon element) side of the
    # region start and end points.
    if np.isscalar(obs_pad):
        obs_pad = (obs_pad, obs_pad)
    
    # Get the wavelengths
    wav = region_result.wav
    
    # List of the legend labels
    legend_labels = []
    
    # Get the observed spectrum contained in the region
#    if obs_pad == (0.0, 0.0):
#        rwav = region_result.region.wav
#        rinten = region_result.region.inten
#    else:
#        lambda0 = region_result.region.lambda0 - obs_pad[0]
#        lambda_end = region_result.region.lambda_end + obs_pad[1]
#        
#        # Get the wavelengths and intensities
#        rwav, rinten, cont = _at.getatlas(lambda0, lambda_end, cgs = True)
#        rinten /= region_result.region.inten_scale_factor
#    
#    # Plot the observed spectrum, followed by the synth lines
#    lbl_obs = ax.plot(rwav, rinten, color = "blue", label = "FTS atlas")
    # Plot the observed spectrum, of obs_last is not True
    if not obs_last:
        lbl_obs, rwav = _plot_obs(region_result, obs_pad, ax)

    # Plot the synthetic spectrum
    if show_abunds:
        # Get the intensities for the filtered (or unfiltered) abundances
        abund, inten = _filter_abund(abund_filter, region_result.abund, region_result.inten)
        
        # Plot the intensities for the chosen abundances       
        for a in range(inten.shape[0]):
            # Plot everything but the best abundance
            if not np.all(inten[a] == region_result.best_inten):
                lbl = ax.plot(wav - offset, inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.75, label = _log_abund_str(abund[a]), linestyle = "--")
                legend_labels.append(lbl[0])
    
    # Plot the best abundance
    lbl_best = ax.plot(wav - offset, region_result.best_inten, color = "red", label = _log_abund_str(region_result.best_abund))
    legend_labels.append(lbl_best[0])
    
    # Plot the observed spectrum, of obs_last is True
    if obs_last:
        lbl_obs, rwav = _plot_obs(region_result, obs_pad, ax)
    legend_labels.append(lbl_obs[0])
    
    # Set the formatter
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))
    
    # Make sure there are always limits
    if None == xlim:
        xlim = (min(wav[0], rwav[0]), max(wav[-1], rwav[-1]))
    if None == ylim:
        ylim = (0.0, 1.1)
    
    # Adjust the x ticks
    _adjust_xyticks(ax, xticks, yticks, xlimits = xlim, ylimits = ylim)

    # Make sure entire y scale is shown
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if show_legend:
        ax.legend(handles = legend_labels, loc = legend_pos, fontsize = legend_font_size, frameon = False)
    if figure_axes == None:
        _plt.tight_layout()
        _plt.show()

def plot_vs(func, xlabel = None, ylabel = None, xticks = None, yticks = None, xlim = None, ylim = None, xfmt = "%0.2f"):
    """
    Creates a function that plots two quantities derived from a region result object, such as an instance of
    ChiRegionResult or EWRegionResult, against each other. The created function is returned. The required
    argument is
    
        func : A function that extracts the x and y values from a region result object. As such it takes one
               required argument and returns two lists, or arrays, of equal length. The first list is the
               x values and the second is the y values.

    The optional arguments are
    
        xlabel : The label of the x axis. If this argument is None, the y axis will not have a label.
                 Default is None.
    
        ylabel : The label of the y axis. If this argument is None, the y axis will not have a label.
                 Default is None.
        
        xticks : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                 ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                 array of filtered ticks. And if None is used, nothing will happen.
                 Default is None.
        
        yticks : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                 ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                 array of filtered ticks. And if None is used, nothing will happen.
                 Default is None.
        
        xlim   : Sets the limits of the x axis. If this is None, no limit is set.
                 Default is None.
        
        ylim   : Sets the limits of the y axis. If this is None, no limit is set.
                 Default is None.
        
        xfmt   : If not None, sets the formatter for the x axis. For information about how this works, look at the documentation
                 for matplotlib.ticker.FormatStrFormatter.
                 Default is "%0.2f".
    
    The returned function has a single required argument, namely

        region_result  : The region result object.
    
    It also has the following optional argument
        
        figure_axes : Sets the axes object. If this is None, then the result of
                      matplotlib.pyplot.gca() will be used. And if this is not None
                      then it will be used to plot the abundance. Also note that
                      if this is not None, the plot will not be shown implicitly.
                      Thereby this can be used to have several plots in the same figure.
                      Default is None.
    """
    
    def plotting_func(region_result, figure_axes = None):
        # Get the axes object
        ax = _get_figure_axes(figure_axes)
        
        x, y = func(region_result)
        ax.plot(x, y)
        
        if xfmt != None:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(xfmt))
        
        # Adjust the ticks for the x and y axes
        _adjust_xyticks(ax, xticks, yticks)
        
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)
        if xlabel != None:
            ax.set_xlabel(xlabel, fontsize = plot_font_size)
        if ylabel != None:
            ax.set_ylabel(ylabel, fontsize = plot_font_size)
        if figure_axes == None:
            _plt.show()
    return plotting_func

def plot_vs_abund(abund, values, ylabel = None, xticks = None, yticks = None, figure_axes = None):
    """
    Plots a quantity against the abundance. The required arguments are

        abund  : An iterable over abundances. An individual abundance is expected to be in the same
                 form as if it was created with the function abundutils.abund.

        values : An iterable of values.
    
    The optional argument is
    
        ylabel      : The label of the y axis. If this argument is None, the y axis will not have a label.
                      Default is None.
        
        xticks      : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                      ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                      array of filtered ticks. And if None is used, nothing will happen.
                      Default is None.
        
        yticks      : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                      ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                      array of filtered ticks. And if None is used, nothing will happen.
                      Default is None.
        
        figure_axes : Sets the axes object. If this is None, then the result of
                      matplotlib.pyplot.gca() will be used. And if this is not None
                      then it will be used to plot the abundance. Also note that
                      if this is not None, the plot will not be shown implicitly.
                      Thereby this can be used to have several plots in the same figure.
                      Default is None.
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    ax.plot(_abund(abund), values)
    
    # Adjust the x ticks
    _adjust_xyticks(ax, xticks, yticks)
    
    ax.set_xlabel("Fe abundance", fontsize = plot_font_size)
    if ylabel != None:
        ax.set_ylabel(ylabel, fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_chisq(region_result, xticks = None, yticks = None, figure_axes = None):
    """
    Plots chi squared for the given region result vs the abundance. The required argument is

        region_result : The region result from which the chi squared should be plotted. This will work
                        for an instance of ChiRegionResult, but not EWRegionResult.
    
    The optional argument is
    
        xticks      : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                      ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                      array of filtered ticks. And if None is used, nothing will happen.
                      Default is None.
    
        yticks      : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                      ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                      array of filtered ticks. And if None is used, nothing will happen.
                      Default is None.
    
        figure_axes : Sets the axes object. If this is None, then the result of
                      matplotlib.pyplot.gca() will be used. And if this is not None
                      then it will be used to plot the abundance. Also note that
                      if this is not None, the plot will not be shown implicitly.
                      Thereby this can be used to have several plots in the same figure.
                      Default is None.
    """
        
    plot_vs_abund(region_result.abund, region_result.chisq, ylabel = "$\\chi^2$", xticks = xticks, yticks = yticks, figure_axes = figure_axes)

def plot_bisect(region_result, offset = 0.0, show_observed = True, show_synth = False, show_labels = True, abund_filter = None, num = 50, xticks = None, yticks = None, xlim = None, ylim = None, figure_axes = None):
    """
    Plots the bisector of the given region result. It is possible to plot this for both synthetic and observed data. By default both are shown.

        region_result : The region result for the region where the bisector is plotted.
        
    The optional arguments are

        offset        : Offsets the synthetic spectrum. Positive values offsets it to the right while negative to the left.
                        Default is 0.
        
        show_observed : Determines if the observed spectrum and its bisector should be shown.
                        Default is True.
        
        show_synth    : Determines if the synthetic spectrum and its bisector should be shown.
                        Default is False.
        
        show_labels   : Determines if the axis labels should be shown.
                        Default is True.
        
        abund_filter  : A filter used to select the abundances to be shown. If None only the synthetic line corresponding to the best fit
                        abundance is used. If it is an integer, numpy array, list or any other collection, it contains the index or indices
                        of the abundances to show.
                        Default is True.
        
        num           : The amount of points for which the bisector should be calculated.
                        Default is 50.
        
        xticks        : Sets the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of x values and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
                        Default is None.
    
        yticks        : Sets the ticks of the y axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of y values and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
                        Default is None.
        
        xlim          : Sets the limits of the x axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                        the first element in the tuple is the minimum x value shown and the second element is the maximum x value shown.
                        Default is None.
        
        ylim          : Sets the limits of the y axis. Should either be a 2 element tuple or None. If None, no limit will be placed. Otherwise
                        the first element in the tuple is the minimum y value shown and the second element is the maximum y value shown.
                        Default is None.
        
        figure_axes   : Sets the axes object. If this is None, then the result of
                        matplotlib.pyplot.gca() will be used. And if this is not None
                        then it will be used to plot the abundance. Also note that
                        if this is not None, the plot will not be shown implicitly.
                        Thereby this can be used to have several plots in the same figure.
                        Default is None.
    
    Note that at least one of plot_observed or plot_synth must be true. Otherwise an exception is raised.
    """
    
    if not (show_observed or show_synth):
        print("Must plot something")

    ax = _get_figure_axes(figure_axes)
    
    #
    min_inten_val = np.inf

    # Plot the bisector of the synthetic data    
    if show_synth:
        # Get the wavelengths
        rwav = region_result.wav
        if None == abund_filter:
            rinten_all = [region_result.best_inten]
        elif isinstance(abund_filter, (int, long)):
            rinten_all = [region_result.inten[abund_filter]]
        else:
            rinten_all = [inten for i, inten in enumerate(region_result.inten) if i in abund_filter]
        
        # Plot the bisectors
        for a, rinten in enumerate(rinten_all):
            bwav, binten = bisect.get_bisector(rwav, rinten, num = num)
            ax.plot(rwav - offset, rinten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.4, linestyle = "--")
            ax.plot(bwav - offset, binten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.8)
    
    # Plot the bisector of the observed data
    if show_observed:
        rwav = region_result.region.wav
        rinten = region_result.region.inten
        bwav, binten = bisect.get_bisector(rwav, rinten, num = num)
        ax.plot(rwav, rinten, color = "blue", alpha = 0.75, linestyle = "--")
        ax.plot(bwav, binten, color = "blue")

    # Set the formatter
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))

    # Make sure there are always limits
    if None == xlim:
        xlim = (min(region_result.wav[0], region_result.region.wav[0]), max(region_result.wav[-1], region_result.region.wav[-1]))
    if None == ylim:
        ylim = (0.0, 1.1)

    # Adjust the x ticks
    _adjust_xyticks(ax, xticks, yticks, xlimits = xlim, ylimits = ylim)

    # Make sure everything is properly shown
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_macroturb(region_result, abund_index = None, show_obs = True, alpha_obs = 1.0, xticks = None, yticks = None, xticks_fmt = "%0.2f", linestyle_obs = "--", legend_pos = 4, figsize = None):
    """
    Plots the intensity that handles macroturbulence and the corresponding intensity that does not handle macroturbulence against
    the wavelength, for an abundance. If no abundance is specified, the best fit abundance is used. The required argument is
    
        region_result : The region result object, containing the results of the calculations for a
                        specific region.
    
    The optional arguments are
        
        abund_index   : The abundance index to show the macroturbulence for. If set to None the best
                        abundance will be used.
                        Default is None.
        
        show_obs      : Determines if the observed spectrum should be shown as well.
                        Default is True.
        
        alpha_obs     : Determines the alpha value of the observed spectrum, if it where to be plotted.
                        Essentially this sets how transperant the observed spectrum is. A value of 0 is
                        invisible and a value of 1 is fully visible.
                        Default is 1.
        
        xticks        : If not None, this sets the x ticks of the plot. It can be a positive integer, in
                        which case it determines how many x ticks should be placed between the minimum and
                        maximum wavelengths (including bounderies). Otherwise it is an array over x ticks.
                        Default is None.
        
        yticks        : If not None, this sets the y ticks of the plot. It can be a positive integer, in
                        which case it determines how many y ticks should be placed between the minimum and
                        maximum wavelengths (including bounderies). Otherwise it is an array over x ticks.
                        Default is None.
        
        xticks_fmt    : If not None, it sets the format string for the x ticks.
                        Default is "%0.2f".
        
        linestyle_obs : Determines the style with which the observed spectrum should be drawn. For example,
                        a value of "-" means it is drawn as a solid line, while a value of "--" means it is
                        drawn as a dashed line.
                        Default is "--".
        
        legend_pos    : Determines the position of the legend. Valid values are
                        
                            0  : best
                            1  : upper right
                            2  : upper left
                            3  : lower left
                            4  : lower right
                            5  : right
                            6  : center left
                            7  : center right
                            8  : lower center
                            9  : upper center
                            10 : center
                        
                        Alternatively a 2 element tuple can be used to specify the x and y position (first element
                        is x, second element is y) of the lower left corner of the legend. This position has to be
                        in the coordinates of the plot, so x is wavelength and y is the normalized intensity.
                        Default is 4.
        
        figsize       : If not None, this sets the size of the figure.
                        Default is None.

    """
    
    # Set the figure size
    if None != figsize:
        _plt.figure(figsize = figsize)

    # If the abundance index is None, set it to the index of the best fit abundance    
    if None == abund_index:
        abund_index = region_result.best_index
    
    # Get the intensities
    inten_nm = region_result.inten_no_macroturb[abund_index]
    inten_wm = region_result.inten[abund_index]
    inten_obs = region_result.region.inten
    
    # Plot the intensities without macroturbulence and then with macroturbulence against the wavelength
    lbl_nm = _plt.plot(region_result.wav, inten_nm, color = "red", label = "Original")
    lbl = _plt.plot(region_result.wav, inten_wm, color = "green", label = "Convolved")
    labels = [lbl_nm[0], lbl[0]]
    
    # Plot the observed spectrum if show_obs is true
    if show_obs and alpha_obs != 0.0:
        lbl_obs = _plt.plot(region_result.region.wav, inten_obs, color = "blue", label = "FTS atlas", alpha = alpha_obs, linestyle = linestyle_obs)
        labels.append(lbl_obs[0])
    
    # Add the legend showing which curve is which, as well as the x-axis and y-axis labels... then show the plot
    _plt.legend(handles = labels, loc = legend_pos, fontsize = legend_font_size, frameon = False)
    _plt.xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    _plt.ylabel("Normalized intensity", fontsize = plot_font_size)

    # Adjust the y limits
    min_inten = min([min(inten_nm), min(inten_wm), min(inten_obs)])
    _plt.ylim([min_inten - 0.05, 1.1])
    
    # Adjust the x ticks
    if isinstance(xticks, int) and xticks > 0:
        xticks = np.linspace(region_result.wav[0], region_result.wav[-1], num = xticks)
        _plt.xticks(xticks)
    elif None != xticks:
        _plt.xticks(xticks)
    
    # Adjust the y ticks
    if isinstance(yticks, int) and yticks > 0:
        yticks = np.linspace(min_inten - 0.05, 1.1, num = yticks)
        _plt.yticks(yticks)
    elif None != xticks:
        _plt.yticks(yticks)
    
    # Set the format of the x axis
    if xticks_fmt != None:
        _plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter(xticks_fmt))
    
    # Show the plot
    _plt.tight_layout()
    _plt.show()

def _estimate_line_wavelength(region):
    tck = si.splrep(region.wav, region.inten)
    min_wav = region.wav[region.inten == min(region.inten)][0]
    wav = np.linspace(min_wav - 2*region.dlambda, min_wav + 2*region.dlambda)
    inten = si.splev(wav, tck, der = 0)
    return wav[inten == min(inten)][0]

def plot_abund(region_results, figure_axes = None):
    """
    Plots the best iron abundances against characteristic wavelengths of
    the regions. The argument is:
    
        region_results : The list of region results.
    
    The optional argument is
        
        figure_axes  : Sets the axes object. If this is None, then the result of
                       matplotlib.pyplot.gca() will be used. And if this is not None
                       then it will be used to plot the abundance. Also note that
                       if this is not None, the plot will not be shown implicitly.
                       Thereby this can be used to have several plots in the same figure.
                       Default is None.
    """
    
    if figure_axes == None:
        ax = _plt.gca()
    else:
        ax = figure_axes
    
    wav = np.array([_estimate_line_wavelength(r.region) for r in region_results])
    abund = np.array([_abund(r.best_abund) for r in region_results])
    
    ax.plot(abund, wav, ".")
    ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    ax.set_ylabel("Fe abundance", fontsize = plot_font_size)
    
    if figure_axes == None:
        _plt.show()

def abund_histogram(region_results, bins = 5, figure_axes = None):
    """
    Plots a histogram of the abundances, using the given amount of bins.
    The required argument is
    
        region_results : A list of region results. Their best abundances will
                         be used to plot the histogram.
    
    The optional arguments are
    
        bins         : The amount of bins of the histogram.
                       Default is 5.
        
        figure_axes  : Sets the axes object. If this is None, then the result of
                       matplotlib.pyplot.gca() will be used. And if this is not None
                       then it will be used to plot the abundance. Also note that
                       if this is not None, the plot will not be shown implicitly.
                       Thereby this can be used to have several plots in the same figure.
                       Default is None.
    """
    
    if figure_axes == None:
        ax = _plt.gca()
    else:
        ax = figure_axes
    
    abundances = np.array([_abund(r.best_abund) for r in region_results])
    ax.hist(abundances, bins = bins)
    ax.set_xlabel("Fe abundance", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def dual_abund_histogram(result_chi, result_ew, bins = 5, xticks = None, yticks = None):
    """
    Plots two histograms of the abundances, using the given amount of bins. The first
    histogram is for the abundances aquired using chi squared fitting, while the second
    is for the abundances aquired by comparing equivalent widths. The required arguments
    are
    
        result_chi : A list of region results from a chi squred fit. Their best abundances will
                     be used to plot the first histogram.
        
        result_ew  : A list of region results from aquired by comparing equivalent widths. Their
                     best abundances will be used to plot the second histogram.
    
    The optional arguments are
    
        bins         : The amount of bins of the histograms.
                       Default is 5.
        
        xticks       : Sets the xticks. If this is None, nothing is done.
                       Default is None.
        
        yticks       : Sets the yticks. If this is None, nothing is done.
                       Default is None.
    """
    
    fig, axes = _plt.subplots(nrows = 1, ncols = 2)
    results = [result_chi, result_ew]
    titles = ["Result from $\\chi^2$", "Result from EW"]

    # Plot the histograms
    for ax, r, t in zip(axes, results, titles):
        ax.set_title(t, fontsize = title_font_size)
        if xticks != None:
            ax.set_xticks(xticks)
        if yticks != None:
            ax.set_yticks(yticks)
        abund_histogram(r, bins = bins, figure_axes = ax)
    
#    # Plot the histogram for the equivalent width results
#    axes[1].set_title("Result from EW", fontsize = plot_font_size)
#    if xticks != None:
#        axes[1].set_xticks(xticks)
#    abund_histogram(result_ew, bins = bins, figure_axes = axes[1])

    # Show the histograms   
    fig.tight_layout()
    _plt.show()

def plot_scaled(region):
    """
    Plots the spectrum scaled after the local maximum and the spectrum scaled after
    the continuum level for the data points, together in the same plot. The argument is
    
        region : The region object. This should be an instance of the Region class.
    """
    
    _plt.plot(region.wav, region.inten, "b")
    _plt.plot(region.wav, region.inten*region.inten_scale_factor/region.cont, "r")
    _plt.xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    _plt.ylabel("Normalized intensity", fontsize = plot_font_size)
    _plt.show()

def plot_in(lambda0, lambda_end, *args, **kwargs):
    """
    Plots the observed spectrum in the given interval. The required arguments are

        lambda0    : The starting wavelength of the interval.
        
        lambda_end : The final wavelength of the interval.
    
    An optional argument is
    
        normalize : Normalizes the intensity if set to True.
                    Default is True.
        
    Any additional arguments are passed on to the plotting function.
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
    _plt.xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    if normalize:
        _plt.ylabel("Normalized intensity", fontsize = plot_font_size)
    else:
        _plt.ylabel("Intensity", fontsize = plot_font_size)
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

def plot_delta(y, x = None, xlabel = None, ylabel = None, *args, **kwargs):
    """
    Plots the changes in y between data points. Specifically, it plots the differences between the
    individual elements in y. The required argument is
    
        y : The data points for which to plot the differences between individual elements.
        
    There is an optional argument
        
        x : The points at which to plot the differences. If set to None, an interval from 0 to the
            length of y minus 1 with steps of 1 will be used.
    
    Any additional parameters are passed on to the plotting function.
    """
    
    dy = y[1:] - y[:-1]
    if x == None:
        x = np.arange(len(dy))
    _plt.plot(x, dy, ".", *args, **kwargs)
    if xlabel != None:
        _plt.xlabel(xlabel, fontsize = plot_font_size)
    if ylabel != None:
        _plt.ylabel(ylabel, fontsize = plot_font_size)
    _plt.show()

def plot_row(obj, plot_funcs, titles = None, figsize = None):
    """
    """

    if figsize == None:
        fig, ax = _plt.subplots(nrows = 1, ncols = len(plot_funcs))
    else:
        fig, ax = _plt.subplots(nrows = 1, ncols = len(plot_funcs), figsize = figsize)
    for i, f in enumerate(plot_funcs):
        if titles != None:
            ax[i].set_title(titles[i], fontsize = title_font_size)
        f(obj, figure_axes = ax[i])
    
    _plt.tight_layout()
    _plt.show()

def _hide(ax):
    """
    Hides an axes object.
    """
    invisible = matplotlib.colors.colorConverter.to_rgba("#FFFFFF", alpha = 0.0)
    ax.set_axis_bgcolor(invisible)
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor = invisible, top = "off", bottom = "off", left = "off", right = "off")
    
def plot_grid(objects, rows, columns, plot_func, xlabel = None, ylabel = None, figsize = None):
    """
    Plots the given list of objects in a grid with the given amount of rows and columns, using the given plotting
    function. The required arguments are
    
        objects   : An list of objects of some type.
        
        rows      : The number of rows.
        
        columns   : The number of columns.
        
        plot_func : The function used to plot an object. It takes two required arguments. The first is the axis object
                    and the second is the object to be plotted.
    
    The optional arguments are
        
        xlabel  : The global label for the y axis. If set to None, no such label is shown.
                  Default is None.
        
        ylabel  : The global label for the y axis. If set to None, no such label is shown.
                  Default is None.
        
        figsize : Sets the size of the figure. If None, this does nothing.
                  Default is None.
    """
    
    # Make sure there are enough "cells"
    if rows*columns < len(objects):
#        raise Exception("Each object must have a cell. There where only " + str(rows*columns) + " cells while there was " + str(len(objects)) + " objects.")
        print("WARNING: All objects does not fit in the grid, so some will be ignored.")
    
    # Create the figure
    if None != figsize:
        fig = _plt.figure(figsize = figsize)
    else:
        fig = _plt.figure()
    
    # Create the main axis
    main_ax = fig.add_subplot(1,1,1)

    # Create the grid
    for i, obj in enumerate(objects):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plot_func(ax, obj)

    # Hide the "main subplot", except for the labels on the x and y axes
    _hide(main_ax)
    
    # Set the x and y labels
    if xlabel != None:
        main_ax.set_xlabel(xlabel, fontsize = plot_font_size)
    if ylabel != None:
        main_ax.set_ylabel(ylabel, fontsize = plot_font_size)
    
    # Show the plot
    fig.tight_layout()
    _plt.show()

def plot_mosaic(objects, rows, columns, plot_func, *args, **kwargs):
    """
    Plots the given list of objects in a mosaic with the given amount of rows and columns, using the given plotting
    function. The required arguments are
    
        objects   : An iterable of objects of some type.
        
        rows      : The number of rows.
        
        columns   : The number of columns.
        
        plot_func : The function used to plot an object. It takes at least one required argument, which is the
                    individual elements in "objects", as well at an optional argument "figure_axes".
    
    The optional arguments are
    
         titles : A list of titles for each cell. If set to None, no titles will be set.
                  Default is None.
        
         sharex : Sets if the x axis should be shared for cells that overlap vertically.
                  Default is False.
        
         sharey : Sets if the x axis should be shared for cells that overlap horizontically.
                  Default is False.
        
         xticks : Sets the xticks. If this is None, nothing is done.
                  Default is None.
        
         yticks : Sets the yticks. If this is None, nothing is done.
                  Default is None.
        
         xlim   : Sets the limits of the x axis. This should be a 2 element tuple, where the
                  first element is the minimum and the second element is the maximum. If this
                  is None, no limit is used.
                  Default is None.
        
         ylim   : Sets the limits of the y axis. This should be a 2 element tuple, where the
                  first element is the minimum and the second element is the maximum. If this
                  is None, no limit is used.
                  Default is None.
        
        figsize : Sets the figure size, unless None is given.
                  Default is None.
    
    Any other arguments are passed on to "plot_func" when each cell is plotted. This includes keyword arguments.
    """

    # Get the keyword relevant arguments
    titles = kwargs.pop("titles", None)
    sharex = kwargs.pop("sharex", False)
    sharey = kwargs.pop("sharey", False)
    xticks = kwargs.pop("xticks", None)
    yticks = kwargs.pop("yticks", None)
    ylim = kwargs.pop("ylim", None)
    xlim = kwargs.pop("xlim", None)
    figsize = kwargs.pop("figsize", None)
    
    # Plot the objects
    if figsize == None:
        fig, axes = _plt.subplots(nrows = rows, ncols = columns, sharex = sharex, sharey = sharey)
    else:
        fig, axes = _plt.subplots(nrows = rows, ncols = columns, sharex = sharex, sharey = sharey, figsize = figsize)
    
    # Reshape the axes array to a 1 dimensional array, if it is an array
    # Otherwise turn it into a 1 dimensional array
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(rows*columns)
    else:
        axes = np.array([axes])
    
    # Plot everything
    for i, (obj, ax) in enumerate(zip(objects, axes)):
        if titles != None:
            ax.set_title(titles[i], fontsize = title_font_size)
        if xticks != None:
            ax.set_xticks(xticks)
        if yticks != None:
            ax.set_yticks(yticks)
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)
        plot_func(obj, *args, figure_axes = ax, **kwargs)
    
    # If now all cells where used, hide the rest
    if i < len(axes) - 1:
        for a in axes[i+1:]:
            _hide(a)

    # Show the plot
    fig.tight_layout()
    _plt.show()

def plot_region_mosaic(regions, rows, columns, figsize = None):
    """
    Plots the given regions in a mosaic with the given amount of rows and columns. The required arguments are
    
        regions : An iterable of Region objects.
        
        rows    : The number of rows.
        
        columns : The number of columns.
    
    The optional argument is
    
        figsize : Sets the size of the figure. If None, this does nothing.
                  Default is None.
    """
    
    # Function used to plot a single cell
    def plot_cell(ax, r):
        ax.set_ylim([0,1.02])
        ax.plot(r.wav, r.inten, color = "blue")
    
    plot_grid(regions, rows, columns, plot_cell, xlabel = u"Wavelength $\\lambda$ [Å]", ylabel = u"Normalized intensity", figsize = figsize)

def plot_result_mosaic(region_results, rows, columns, figsize = None):
    """
    Plots the given region results in a mosaic with the given amount of rows and columns. The required arguments are
    
        region_results : An iterable of RegionResult objects.
        
        rows           : The number of rows.
        
        columns        : The number of columns.
    
    The optional argument is
    
        figsize : Sets the size of the figure. If None, this does nothing.
                  Default is None.
    """
    
    # Function used to plot a single cell
    def plot_cell(ax, reg_result):
        reg = reg_result.region
        ax.set_ylim([0,1.02])
        ax.plot(reg_result.wav, reg_result.best_inten, color = "red")
        ax.plot(reg.wav, reg.inten, color = "blue")

    plot_grid(region_results, rows, columns, plot_cell, xlabel = u"Wavelength $\\lambda$ [Å]", ylabel = u"Normalized intensity", figsize = figsize)
