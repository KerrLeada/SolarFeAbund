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

_FILENAME_PREFIX = os.path.join("data", "testloggf", "linesnr")
_REGION_LAB_WAV = [6219.2801, 6173.3339, 6498.9379]

def gen_cfg_files(pertubations, decimals = 5):
    files = [_FILENAME_PREFIX + str(i) + ".cfg" for i in range(len(pertubations))]
    for p, f in zip(pertubations, files):
        with open(f, "w") as curr_file:
            curr_file.write("FeI_6219    Fe    1    26    6219.2801  " + str(-2.433 + np.round(p, decimals = decimals)) + "   2.0    2.0    1.820   1.500   2.1979   8.290    -6.160    278.264   1.0\n")
            curr_file.write("FeI_6173    Fe    1    26    6173.3339  " + str(-2.880 + np.round(p, decimals = decimals)) + "   1.0    0.0    2.500   0.000   2.2227   8.310    -6.160    281.266   1.0\n")
            curr_file.write("FeI_6498    Fe    1    26    6498.9379  " + str(-4.699 + np.round(p, decimals = decimals)) + "   3.0    3.0    1.250   1.510   0.9582   4.430    -6.210    226.253   1.0\n")
    return pertubations, files

pertubations, files = gen_cfg_files(np.arange(-0.2, 0.25, step = 0.05))

at = sa.satlas()
abunds = -np.arange(4.1, 4.8, step = 0.001)
region_list = [r for r in regfile.get_regions() if r.lab_wav in _REGION_LAB_WAV]

results = []
for p, f in zip(pertubations, files):
    print("Starting synth, with pertubation:", p)
    r = synther.fit_spectrum(f, at, region_list, abunds, verbose = False)
    results.append(r)

def plot_pert():
    plotting.plot_pert(pertubations, results)
