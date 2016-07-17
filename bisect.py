"""
This module contains functions that handles bisectors. Specifically one function.
"""

from __future__ import print_function
from __future__ import division

import scipy.interpolate as si
import numpy as np

def _estimate_minimum(x, y):
    dx = np.max(x[1:] - x[:-1])
    ymin = np.min(y)
    xmin = x[y == ymin][0]
    tck = si.splrep(x, y)
    xvals = np.linspace(xmin - 3*dx, xmin + 3*dx, num = 500)
    yvals = si.splev(xvals, tck)
    ymin = np.min(yvals)
    return xvals[yvals == ymin][0], ymin

def get_bisector(x, y, ylim = None, num = 50):
    """
    Gets the bisector of the curve given by x and y. The required arguments are
    
        x : The x values.
        
        y : The y values corresponding to the x values.
    
    The optional arguments is
    
        ylim : Specifies an upper limit, above which the bisector is not calculated. If set to None, no
               such limit is used. Otherwise, ylim has to be a number or a function. If it is a function
               it will accept the y values and return a number, which will be the y limit.
               Default is None.
    
        num  : Specifies how many points the bisector should be calculated in (at most).
    
    Note that if there are more or less then two x coordinates for a given y coordinate, it will be ignored.
    """

    # Make sure x and y have the same length
    if len(x) != len(y):
        raise Exception("x and y must have equal length, but x had length " + str(len(x)) + " while y had length " + str(len(y)) + ".")
    
    # If x and y had 0 length, return empty arrays
    if len(x) == 0:
        return np.array([]), np.array([])
    
    # Initialize the lists that stores the bisector points
    x_pts = []
    y_pts = []
    
    # Estimate the minimum
    xmin, ymin = _estimate_minimum(x, y)
    
    # If no y limit was given, set it to the maximum y value
    if ylim == None:
        ylim = max(y)
    elif hasattr(ylim, "__call__"):
        ylim = ylim(y)
    
    # Find the bisector points
    for curr_y in np.linspace(min(y), ylim, num = num):
        # Find the root at y = curr_y by using quadratic interpolation
        tck = si.splrep(x, y - curr_y)
        roots = np.sort(si.sproot(tck))
        
        # If there are exactly 2 roots, there is a bisector point
        if len(roots) == 2:
            # Make sure both roots are on either side of the minimum
            if roots[0] < xmin and roots[1] > xmin:
                x_pts.append(np.mean(roots))
                y_pts.append(curr_y)
        elif len(roots) > 2:
            print("Number of roots:", len(roots))
            # Determine the x on either side of xmin by subtracting xmin and taking the x values directly on the left
            # and right side of the transition between negative and positive.
            x_left = None
            x_right = None
            for xval in (roots - xmin):
                if xval < 0:
                    x_left = xval
                else:
                    x_right = xval
                    break
            
            # If either x_left or x_right is None, then all roots are on one side of xmin and nothing should be added
            # to the list of bisector x values
            if x_left != None and x_right != None:
                x_pts.append(np.mean([x_left + xmin, x_right + xmin]))
                y_pts.append(curr_y)
    
    # Return the bisector points as an array over the x values and an array over the y values
    return np.array(x_pts), np.array(y_pts)

