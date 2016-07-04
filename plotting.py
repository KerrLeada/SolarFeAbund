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

# Get the atlas
_at = _sa.satlas()

# A list of colors to use in plots
plot_color_list = ["#FF0000", "#00FF00", "#FF00FF",
                   "#11FF0A", "#AAAA00", "#00AAAA",
                   "#AF009F", "#0F3FF0", "#F0FA0F",
                   "#A98765", "#0152A3", "#0FFFF0",
                   "#C0110C", "#0B0D0E", "#BDC0EB"]

# Set the font to the default
_plt.rc("font", **{"family": u"sans-serif", u"sans-serif": [u"Helvetica"]})

# Set the font size of the plots
plot_font_size = 11

# The figure size (in inches) of some functions
plot_figure_size = (7, 7)

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

def _adjust_xticks(ax, adjustment):
    # Adjust the x ticks
    if isinstance(adjustment, int) and adjustment > 0:
        xticks = ax.get_xticks()
        ax.set_xticks(np.linspace(xticks[0], xticks[-1], num = adjustment))
    elif hasattr(adjustment, "__call__"):
        xticks = ax.get_xticks()
        ax.set_xticks(adjustment(xticks))
    elif adjustment != None:
        try:
            ax.set_xticks(adjustment)
        except:
            raise Exception("Illegal value for xticks_adjust. It must be None, a positive integer, a function or a list of xticks to use, but it had type " + type(adjustment).__name__ + " and value " + str(adjustment))

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
    return inten

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
        ax.legend(handles = [lbl_comp[0], lbl_real[0], lbl_obs[0]], loc = legend_pos, fontsize = plot_font_size)
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
        ax.legend(handles = [lbl_comp[0], lbl_real[0], lbl_obs[0]], loc = legend_pos, fontsize = plot_font_size)
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_shifted(region_result, show_labels = True, show_legend = True, legend_pos = 4, obs_pad = 0.0, abund_filter = None, figure_axes = None, xticks_adjust = None):
    """
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    # Get the wavelengths
    wav = region_result.wav
    
    # Make sure entire y scale is shown
    ax.set_xlim([wav[0], wav[-1]])
    ax.set_ylim([0, 1.1])
    
    # Plot the unshifted, shifted and observed spectrums
    lbl_best = ax.plot(wav, region_result.best_inten, color = "red", label = "Shifted")
    lbl_shifted = ax.plot(wav + region_result.best_shift, region_result.best_inten, color = "red", linestyle = "--", label = "Not shifted")
    lbl_obs = ax.plot(region_result.region.wav, region_result.region.inten, color = "blue", label = "FTS atlas")

    # Set the formatter
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))
    
    # Adjust the x ticks
#    if isinstance(xticks_adjust, int) and xticks_adjust > 0:
#        xticks = ax.get_xticks()
#        ax.set_xticks(np.linspace(xticks[0], xticks[-1], num = xticks_adjust))
#    elif hasattr(xticks_adjust, "__call__"):
#        xticks = ax.get_xticks()
#        ax.set_xticks(xticks_adjust(xticks))
#    elif xticks_adjust != None:
#        raise Exception("Illegal value for xticks_adjust. It must be None, a positive integer or a function, but it had type " + type(xticks_adjust) + " and value " + str(xticks_adjust))
    _adjust_xticks(ax, xticks_adjust)
    
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if show_legend:
        ax.legend(handles = [lbl_best[0], lbl_shifted[0], lbl_obs[0]], loc = legend_pos, fontsize = 8)
    if figure_axes == None:
        _plt.tight_layout()
        _plt.show()

def plot_region(region_result, offset = 0.0, show_abunds = False, show_labels = True, show_legend = True, legend_pos = 4, obs_pad = 0.0, abund_filter = None, figure_axes = None, xticks_adjust = None):
    """
    Plots the given region result.

        region_result : The result of an abundance fit for a region. This is an instance of ChiRegionResult or EWRegionResult
                        from the synther module.
                        
    The optional arguments are

        offset        : Determines the offset of the synthetic region. Specifically, it shifts the synthetic data.
                        Default is 0.
        
        show_abunds   : Determines if the synthetic data for other abundances then the best abundance should be shown. If set to True,
                        the abundance filter can be used to select specific abundances to show or not. The abundance filter is given
                        through abund_filter.
                        Default is False.
        
        show_labels   : Determines if the labels for the axes should be shown or not.
                        Default is True.
        
        show_legend   : Determines if the legend should be shown.
                        Default is True.
        
        legend_pos    : Determines the position of the legend, if it is shown. Valid values are
                        
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
                        
        obs_pad       : The padding of the observed data. Specifically, it can be used to show observed data from outside of the
                        region. A positive value expands the observed region shown while a negative value decreases it.
                        Default is 0.
        
        abund_filter  : A filter that determines which abundances should be shown. It can be None, a function of anything that a numpy
                        array can be indexed or sliced with. If it is None, nothing is filtered out. If it is a function, it is expected
                        to take 3 arguments. The first is the abundance index, the second is the abundance and the third is the synthetic
                        intensities for that abundance.
                        Default is None.
        
        figure_axes   : Sets the axes object. If this is None, then the result of
                        matplotlib.pyplot.gca() will be used. And if this is not None
                        then it will be used to plot the abundance. Also note that
                        if this is not None, the plot will not be shown implicitly.
                        Thereby this can be used to have several plots in the same figure.
                        Default is None.
        
        xticks_adjust : Adjusts the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of ticks and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
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
    
    # Make sure entire y scale is shown
    ax.set_xlim([wav[0], wav[-1]])
    ax.set_ylim([0, 1.1])
    
    # List of the legend labels
    legend_labels = []

    # Plot the synthetic spectrum
    if show_abunds:
    
        # Get the intensities for the filtered (or unfiltered) abundances
        abund, inten = _filter_abund(abund_filter, region_result.abund, region_result.inten)
        
        # Plot the intensities for the chosen abundances       
        for a in range(inten.shape[0]):
            # Plot everything but the best abundance
            if not np.all(inten[a] == region_result.best_inten):
                lbl = ax.plot(wav - offset, inten[a], color = plot_color_list[a % len(plot_color_list)], alpha = 0.75, label = "$\\log A_{Fe} = " + str(abund[a]) + "$")
                legend_labels.append(lbl[0])
    
    # Plot the best abundance
    lbl_best = ax.plot(wav - offset, region_result.best_inten, color = "red", label = "$\\log A_{Fe} = " + str(region_result.best_abund) + "$")
    legend_labels.append(lbl_best[0])

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
    lbl_obs = ax.plot(rwav, rinten, color = "blue", label = "FTS atlas")
    legend_labels.append(lbl_obs[0])
    
    # Set the formatter
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))
    
    # Adjust the x ticks
#    if isinstance(xticks_adjust, int) and xticks_adjust > 0:
#        xticks = ax.get_xticks()
#        ax.set_xticks(np.linspace(xticks[0], xticks[-1], num = xticks_adjust))
#    elif hasattr(xticks_adjust, "__call__"):
#        xticks = ax.get_xticks()
#        ax.set_xticks(xticks_adjust(xticks))
#    elif xticks_adjust != None:
#        raise Exception("Illegal value for xticks_adjust. It must be None, a positive integer or a function, but it had type " + type(xticks_adjust) + " and value " + str(xticks_adjust))
    _adjust_xticks(ax, xticks_adjust)
    
    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if show_legend:
        ax.legend(handles = legend_labels, loc = legend_pos, fontsize = 8)
    if figure_axes == None:
        _plt.tight_layout()
        _plt.show()

def plot_spec(region_results, show_observed = True, show_continuum = False, show_unshifted = False, padding = 0.0, cgs = True):
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
        
        cgs            : Determines is cgs units should be used.
                         Default is True.
    """

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
    
    _plt.xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    _plt.ylabel("Normalized intensity", fontsize = plot_font_size)
    _plt.show()

def plot_vs(func, xlabel = None, ylabel = None, xticks_adjust = None, xlim = None, ylim = None):
    """
    Creates a function that plots two quantities derived from a region result object, such as an instance of
    ChiRegionResult or EWRegionResult, against each other. The created function is returned. The required
    argument is
    
        func : A function that extracts the x and y values from a region result object. As such it takes one
               required argument and returns two lists, or arrays, of equal length. The first list is the
               x values and the second is the y values.

    The optional arguments are
    
        xlabel        : The label of the x axis. If this argument is None, the y axis will not have a label.
                        Default is None.
    
        ylabel        : The label of the y axis. If this argument is None, the y axis will not have a label.
                        Default is None.
        
        xticks_adjust : Adjusts the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of ticks and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
                        Default is None.
        
        xlim          : Sets the limits of the x axis. If this is None, no limit is set.
                        Default is None.
        
        ylim          : Sets the limits of the y axis. If this is None, no limit is set.
                        Default is None.
    
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
        
        # Adjust the x ticks
        _adjust_xticks(ax, xticks_adjust)
        
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

def plot_vs_abund(abund, values, ylabel = None, figure_axes = None, xticks_adjust = None):
    """
    Plots a quantity against the abundance. The required arguments are

        abund  : An iterable over abundances. An individual abundance is expected to be in the same
                 form as if it was created with the function abundutils.abund.

        values : An iterable of values.
    
    The optional argument is
    
        ylabel      : The label of the y axis. If this argument is None, the y axis will not have a label.
                      Default is None.
        
        figure_axes : Sets the axes object. If this is None, then the result of
                      matplotlib.pyplot.gca() will be used. And if this is not None
                      then it will be used to plot the abundance. Also note that
                      if this is not None, the plot will not be shown implicitly.
                      Thereby this can be used to have several plots in the same figure.
                      Default is None.
        
        xticks_adjust : Adjusts the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of ticks and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
                        Default is None.
    """
    
    # Get the axes object
    ax = _get_figure_axes(figure_axes)
    
    ax.plot(abund, values)
    
    # Adjust the x ticks
    _adjust_xticks(ax, xticks_adjust)
    
    ax.set_xlabel("Fe abundance", fontsize = plot_font_size)
    if ylabel != None:
        ax.set_ylabel(ylabel, fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_chisq(region_result, figure_axes = None, xticks_adjust = None):
    """
    Plots chi squared for the given region result vs the abundance. The required argument is

        region_result : The region result from which the chi squared should be plotted. This will work
                        for an instance of ChiRegionResult, but not EWRegionResult.
    
    The optional argument is
    
        figure_axes   : Sets the axes object. If this is None, then the result of
                        matplotlib.pyplot.gca() will be used. And if this is not None
                        then it will be used to plot the abundance. Also note that
                        if this is not None, the plot will not be shown implicitly.
                        Thereby this can be used to have several plots in the same figure.
                        Default is None.
        
        xticks_adjust : Adjusts the ticks of the x axis. It can be None, an integer or a function. If this is an integer, it specifies how many
                        ticks should be used. If, on the other hand, this is a function then it will take the array of ticks and return a new
                        array of filtered ticks. And if None is used, nothing will happen.
                        Default is None.
    """
        
    plot_vs_abund(region_result.abund, region_result.chisq, ylabel = "$\\chi^2$", figure_axes = figure_axes, xticks_adjust = xticks_adjust)

def plot_bisect(region_result, offset = 0.0, plot_observed = True, plot_synth = True, show_observed = True, show_synth = True, show_labels = True, only_best_synth = False, num = 50, figure_axes = None):
    """
    Plots the bisector of the given region result. It is possible to plot this for both synthetic and observed data. By default both are shown.

        region_result : The region result for the region where the bisector is plotted.
        
    The optional arguments are

        offset          : Offsets the synthetic spectrum. Positive values offsets it to the right while negative to the left.
                          Default is 0.
        
        plot_observed   : Determines if the bisector of the observed spectrum should be shown.
                          Default is True.
        
        plot_synth      : Determines if the bisector of the synthetic spectrum should be shown.
                          Default is True.
        
        show_observed   : Determines if the observed spectrum should be shown.
                          Default is True.
        
        show_synth      : Determines if the synthetic spectrum should be shown.
                          Default is True.
        
        show_labels     : Determines if the axis labels should be shown.
                          Default is True.
        
        only_best_synth : Determines if only the best fit synthetic spectrum should be shown.
                          Default is False.
        
        num             : The amount of points for which the bisector should be calculated.
                          Default is 50.
        
        figure_axes     : Sets the axes object. If this is None, then the result of
                          matplotlib.pyplot.gca() will be used. And if this is not None
                          then it will be used to plot the abundance. Also note that
                          if this is not None, the plot will not be shown implicitly.
                          Thereby this can be used to have several plots in the same figure.
                          Default is None.
    
    Note that at least one of plot_observed or plot_synth must be true. Otherwise an exception is raised.
    """
    
    if not (plot_observed or plot_synth):
        print("Must plot something")

    ax = _get_figure_axes(figure_axes)

    # Plot the bisector of the synthetic data    
    if plot_synth:
        # Get the wavelengths
        rwav = region_result.wav
        if only_best_synth:
            rinten_all = [region_result.best_inten]
        else:
            rinten_all = region_result.inten
        
        # Plot the bisectors
        for a, rinten in enumerate(rinten_all):
            bwav, binten = bisect.get_bisector(rwav, rinten, num = num)
            if show_synth:
                ax.plot(rwav - offset, rinten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.4, linestyle = "--")
            ax.plot(bwav - offset, binten, color = plot_color_list[a % len(plot_color_list)], alpha = 0.8)
    
    # Plot the bisector of the observed data
    if plot_observed:
        rwav = region_result.region.wav
        rinten = region_result.region.inten
        bwav, binten = bisect.get_bisector(rwav, rinten, num = num)
        if show_observed:
            ax.plot(rwav, rinten, color = "blue", alpha = 0.75, linestyle = "--")
        ax.plot(bwav, binten, color = "blue")

    if show_labels:
        ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
        ax.set_ylabel("Normalized intensity", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def plot_macroturb(region_result, abund_index = None, show_obs = True, alpha_obs = 1.0, linestyle_obs = "-", legend_pos = 4):
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
        
        linestyle_obs : Determines the style with which the observed spectrum should be drawn. A value of
                        "-" means it is drawn as a solid line.
                        Default is "-".
        
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

    """

    # If the abundance index is None, set it to the index of the best fit abundance    
    if abund_index == None:
        abund_index = region_result.best_index
        
    # Plot the intensities without macroturbulence and then with macroturbulence against the wavelength
    lbl_nm = _plt.plot(region_result.wav, region_result.inten_no_macroturb[abund_index], color = "red", label = "Not convolved")
    lbl = _plt.plot(region_result.wav, region_result.inten[abund_index], color = "green", label = "Convolved")
    labels = [lbl_nm[0], lbl[0]]
    
    # Plot the observed spectrum if show_obs is true
    if show_obs and alpha_obs != 0.0:
        lbl_obs = _plt.plot(region_result.region.wav, region_result.region.inten, color = "blue", label = "Observed", alpha = alpha_obs, linestyle = linestyle_obs)
        labels.append(lbl_obs[0])
    
    # Add the legend showing which curve is which, as well as the x-axis and y-axis labels... then show the plot
    _plt.legend(handles = labels, loc = legend_pos, fontsize = plot_font_size)
    _plt.xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    _plt.ylabel("Normalized intensity", fontsize = plot_font_size)
    _plt.show()

def _estimate_line_wavelength(region):
    tck = si.splrep(region.wav, region.inten)
    min_wav = region.wav[region.inten == min(region.inten)][0]
    wav = np.linspace(min_wav - 2*region.dlambda, min_wav + 2*region.dlambda)
    inten = si.splev(wav, tck, der = 0)
    return wav[inten == min(inten)][0]

def plot_abund(region_results, with_H_as_12 = False, figure_axes = None):
    """
    Plots the best iron abundances against characteristic wavelengths of
    the regions. The argument is:
    
        region_results : The list of region results.
    
    The optional argument is
        
        with_H_as_12 : Sets which abundance convention should be used.
                       Default is False.
        
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
    abund = np.array([r.best_abund for r in region_results])
    if with_H_as_12:
        y += 12.0
    
    ax.plot(abund, wav, ".")
    ax.set_xlabel(u"Wavelength $\\lambda$ [Å]", fontsize = plot_font_size)
    ax.set_ylabel("Fe abundance", fontsize = plot_font_size)
    
    if figure_axes == None:
        _plt.show()

def abund_histogram(region_results, bins = 5, with_H_as_12 = False, figure_axes = None):
    """
    Plots a histogram of the abundances, using the given amount of bins.
    The required argument is
    
        region_results : A list of region results. Their best abundances will
                         be used to plot the histogram.
    
    The optional arguments are
    
        bins         : The amount of bins of the histogram.
                       Default is 5.
        
        with_H_as_12 : Sets which abundance convention should be used.
                       Default is False.
        
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
    
    abundances = np.array([r.best_abund for r in region_results])
    if with_H_as_12:
        abundances += 12.0
    ax.hist(abundances, bins = bins)
    ax.set_xlabel("Fe abundance", fontsize = plot_font_size)
    if figure_axes == None:
        _plt.show()

def dual_abund_histogram(result_chi, result_ew, bins = 5, with_H_as_12 = False, xticks = None, yticks = None):
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
        
        with_H_as_12 : Sets which abundance convention should be used.
                       Default is False.
        
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
        ax.set_title(t, fontsize = plot_font_size)
        if xticks != None:
            ax.set_xticks(xticks)
        if yticks != None:
            ax.set_yticks(yticks)
        abund_histogram(r, bins = bins, with_H_as_12 = with_H_as_12, figure_axes = ax)
    
#    # Plot the histogram for the equivalent width results
#    axes[1].set_title("Result from EW", fontsize = plot_font_size)
#    if xticks != None:
#        axes[1].set_xticks(xticks)
#    abund_histogram(result_ew, bins = bins, with_H_as_12 = with_H_as_12, figure_axes = axes[1])

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

def plot_row(obj, plot_funcs, titles = None, figsize = None, args = None):
    """
    """

    if args == None:
        args = [options()]*len(plot_funcs)
    elif len(args) != len(plot_funcs):
        raise Exception("plot_options must be either None or list of the same length as plot_funcs")
    else:
        args = [a if a != None else options() for a in args]

    if figsize == None:
        fig, ax = _plt.subplots(nrows = 1, ncols = len(plot_funcs))
    else:
        fig, ax = _plt.subplots(nrows = 1, ncols = len(plot_funcs), figsize = figsize)
    for i, f in enumerate(plot_funcs):
        if titles != None:
            ax[i].set_title(titles[i], fontsize = plot_font_size)
        f(obj, *args[i][0], figure_axes = ax[i], **args[i][1])
    
    _plt.tight_layout()
    _plt.show()

def plot_grid(objects, rows, columns, plot_func, xlabel = None, ylabel = None):
    """
    Plots the given list of objects in a grid with the given amount of rows and columns, using the given plotting
    function. The required arguments are
    
        objects   : An list of objects of some type.
        
        rows      : The number of rows.
        
        columns   : The number of columns.
        
        plot_func : The function used to plot an object. It takes two required arguments. The first is the axis object
                    and the second is the object to be plotted.
    
    The optional arguments are
        
        xlabel   : The global label for the y axis. If set to None, no such label is shown.
                   Default is None.
        
        ylabel   : The global label for the y axis. If set to None, no such label is shown.
                   Default is None.
    """
    
    # Make sure there are enough "cells"
    if rows*columns < len(objects):
#        raise Exception("Each object must have a cell. There where only " + str(rows*columns) + " cells while there was " + str(len(objects)) + " objects.")
        print("WARNING: All objects does not fit in the grid, so some will be ignored.")
    
    # Create the figure
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
    invisible = matplotlib.colors.colorConverter.to_rgba("#FFFFFF", alpha = 0.0)
    main_ax.set_axis_bgcolor(invisible)
    main_ax.spines["top"].set_color("none")
    main_ax.spines["bottom"].set_color("none")
    main_ax.spines["left"].set_color("none")
    main_ax.spines["right"].set_color("none")
    main_ax.tick_params(labelcolor = invisible, top = "off", bottom = "off", left = "off", right = "off")
    
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
    axes = axes.reshape(rows*columns)
    for i, (obj, ax) in enumerate(zip(objects, axes)):
        if titles != None:
            ax.set_title(titles[i], fontsize = plot_font_size)
        if xticks != None:
            ax.set_xticks(xticks)
        if yticks != None:
            ax.set_yticks(yticks)
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)
        plot_func(obj, *args, figure_axes = ax, **kwargs)

    # Show the plot
    fig.tight_layout()
    _plt.show()

def plot_region_mosaic(regions, rows, columns):
    """
    Plots the given regions in a mosaic with the given amount of rows and columns. The arguments are
    
        regions : An iterable of Region objects.
        
        rows    : The number of rows.
        
        columns : The number of columns.
    """
    
    # Function used to plot a single cell
    def plot_cell(ax, r):
        ax.set_ylim([0,1.02])
        ax.plot(r.wav, r.inten, color = "blue")
    
    plot_grid(regions, rows, columns, plot_cell, xlabel = u"Wavelength $\\lambda$ [Å]", ylabel = u"Normalized intensity")

def plot_result_mosaic(region_results, rows, columns):
    """
    Plots the given region results in a mosaic with the given amount of rows and columns. The arguments are
    
        region_results : An iterable of RegionResult objects.
        
        rows           : The number of rows.
        
        columns        : The number of columns.
    """
    
    # Function used to plot a single cell
    def plot_cell(ax, reg_result):
        reg = reg_result.region
        ax.set_ylim([0,1.02])
        ax.plot(reg_result.wav, reg_result.best_inten, color = "red")
        ax.plot(reg.wav, reg.inten, color = "blue")

    plot_grid(region_results, rows, columns, plot_cell, xlabel = u"Wavelength $\\lambda$ [Å]", ylabel = u"Normalized intensity")
