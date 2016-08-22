# -*- coding: utf8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import pyLTE as p
import matplotlib.pyplot as plt
import time
import satlas as sa
import sparsetools as sp
import regions as regs
import astropy.constants
import astropy.units
import synther
import regfile
import plotting
import latexgen
import dispresult
import os
import plotting

_DIRECTORY = os.path.join("data", "testloggf")
_FILENAME_PREFIX = os.path.join("data", "testloggf", "linesnr")
_REGION_LAB_WAV = [6219.2801, 6173.3339, 6498.9379]

def gen_cfg_files(pertubations, decimals = 5):
    # Create the filenames
    files = [_FILENAME_PREFIX + str(i) + ".cfg" for i in range(len(pertubations))]

    # Set the operation
    op = lambda x, y: x + y
    
    # Create the directly, and set the permissions so only the current user can do stuff... and
    # restrict those stuff to reading and writing to files. No execution of stuff.
    if not os.path.isdir(_DIRECTORY):
        os.makedirs(_DIRECTORY, mode = 0300)
    
    # Create the cfg files containing the pertubed atomic data
    for p, f in zip(pertubations, files):
        with open(f, "w") as curr_file:
            curr_file.write("FeI_6219    Fe    1    26    6219.2801  " + str(op(-2.433, np.round(p, decimals = decimals))) + "   2.0    2.0    1.820   1.500   2.1979   8.290    -6.160    278.264   1.0\n")
            curr_file.write("FeI_6173    Fe    1    26    6173.3339  " + str(op(-2.880, np.round(p, decimals = decimals))) + "   1.0    0.0    2.500   0.000   2.2227   8.310    -6.160    281.266   1.0\n")
            curr_file.write("FeI_6498    Fe    1    26    6498.9379  " + str(op(-4.699, np.round(p, decimals = decimals))) + "   3.0    3.0    1.250   1.510   0.9582   4.430    -6.210    226.253   1.0\n")
    
    # Return the pertubations as well as the corresponding cfg file names
    return pertubations, files

pertubations, files = gen_cfg_files(np.arange(-0.20, 0.25, step = 0.05))
#pertubations, files = gen_cfg_files(np.arange(0.95, 1.05, step = 0.01))

at = sa.satlas()
abunds = -np.arange(4.1, 4.8, step = 0.001)
#abunds = -np.arange(3.3, 4.6, step = 0.005)
region_list = [r for r in regfile.get_regions() if r.lab_wav in _REGION_LAB_WAV]

results = []
for p, f in zip(pertubations, files):
    print("")
    print("*** Starting synth, with pertubation:", p, "***")
    r = synther.fit_spectrum(f, at, region_list, abunds, verbose = False)
    results.append(r)

def plot_pert():
    plotting.plot_pert(pertubations, results)
