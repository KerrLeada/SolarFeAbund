# -*- coding: utf8 -*-

"""
This module is used to study the model.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import satlas as sa
import sparsetools as sp
import astropy.units as u
from synther import DEFAULT_MODEL_FILE as model_file
from plotting import plot_font_size, legend_font_size, title_font_size, init_plot_font

# Setup the plot font
init_plot_font()

# Get the temperature gradient. The variables are
#   z:    the height in the atmosphere in km
#   temp: the temperature at a height z
m = sp.model(model_file)
z = (m.z.squeeze() * u.cm).to(u.km).value
temp = m.temp.squeeze()

# Plot the temperature gradient
plt.figure(figsize = (5,3))
plt.plot(z, temp)
plt.xlabel("Height $z$ [km]", fontsize = plot_font_size)
plt.ylabel("Temperature [K]", fontsize = plot_font_size)
plt.xlim(0, 1000)
plt.ylim(4300, 6700)
plt.tight_layout()
plt.show()
