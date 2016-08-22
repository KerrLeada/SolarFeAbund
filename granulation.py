# -*- coding: utf8 -*-

"""
This module handles the fits file containing the granulation image.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Determines if the entire image should be shown, or just the area determined by
# the variables x, y, width and height.
PLOT_SELECTED = False

# Determines the selection area
x, y = 317, 350
width, height = 180, 180

# Plot the granulation pattern
print("Opening fits file")
hdulist = fits.open("data/granulation.fits")
try:
    print("Plotting fits file")
    image_data = hdulist[0].data
    if PLOT_SELECTED:
        image_data = image_data[x:(x+width), y:(y+height)]
    plt.imshow(image_data, cmap = "gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()
finally:
    hdulist.close()
    print("Closed fits file")
