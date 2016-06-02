"""
This module contains functions that handles bisections. Specifically one function.
"""

from __future__ import print_function

import scipy.optimize as so
import scipy.interpolate as si
import numpy as np

def get_bisection(x, y, num = 50):
    """
    Gets the bisection of the curve given by x and y. The argument num specifies in how many points the bisection is calculated for, at most.
    """

    # Make sure x and y have the same length
    if len(x) != len(y):
        raise Exception("x and y must have equal length, but x had length " + str(len(x)) + " while y had length " + str(len(y)) + ".")
    
    # If x and y had 0 length, return empty arrays
    if len(x) == 0:
        return np.array([]), np.array([])
    
    # Initialize the lists that stores the bisection points
    x_pts = []
    y_pts = []
    
    # Find the bisection points
    for curr_y in np.linspace(min(y), max(y), num = num):
        # Find the root at y = curr_y by using quadratic interpolation
        tck = si.splrep(x, y - curr_y)
        roots = si.sproot(tck)
        
        # If there are exactly 2 roots, there is a bisection point
        if len(roots) == 2:
            x_pts.append(np.mean(roots))
            y_pts.append(curr_y)
    
    # Return the bisection points as an array over the x values and an array over the y values
    return np.array(x_pts), np.array(y_pts)
