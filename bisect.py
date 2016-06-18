"""
This module contains functions that handles bisectors. Specifically one function.
"""

from __future__ import print_function

import scipy.optimize as so
import scipy.interpolate as si
import numpy as np

def get_bisector(x, y, num = 50):
    """
    Gets the bisector of the curve given by x and y. The required arguments are
    
        x : The x values.
        
        y : The y values corresponding to the x values.
    
    The optional argument is
    
        num : Specifies how many points the bisector should be calculated in (at most).
    
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
    
    # Find the bisector points
    for curr_y in np.linspace(min(y), max(y), num = num):
        # Find the root at y = curr_y by using quadratic interpolation
        tck = si.splrep(x, y - curr_y)
        roots = si.sproot(tck)
        
        # If there are exactly 2 roots, there is a bisector point
        if len(roots) == 2:
            x_pts.append(np.mean(roots))
            y_pts.append(curr_y)
    
    # Return the bisector points as an array over the x values and an array over the y values
    return np.array(x_pts), np.array(y_pts)

