# -*- coding: utf8 -*-

# ************************
# NOTE ABOUT UNITS:
# Whenever a unit labeled Å is used, it stands for "Ångström". In English this is sometimes
# written "Angstom" and labeled "A" instead of "Å", since English lacks the latters "Å" and "ö".
# ************************

from __future__ import print_function

import numpy as np
import regions as regs
import satlas as _sa

# Get the atlas
_at = _sa.satlas()

# Create the regions
# Note that this list should (mostly) not be used directly
# A copy is retrieved through the get_regions function
_regions = [
    # Line at: 6301.5 Å
    regs.new_region_in(_at, 6301.18, 6301.82, 6301.4999),

    # Line at: 6302.5 Å
    # POSSIBLY (BUT DOUBTFULLY) BETTER:
    #    6302.25 to 6302.75, gives shift of 0.004, cannot see much difference compared to the current
    #    interval 6302.25 to 6302.75. The chi squared is however higher for all abundencies, so it might
    #    just be worse.
    regs.new_region_in(_at, 6302.3, 6302.7, 6302.4935),

    # Line at: [6219.27] or 6219.28
    # CANDIDATES:
    #    dlambda = lambda w: 0.98*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.002... looks like a bit higher shift could be good
    #    dlambda = lambda w: 0.975*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.004... not sure if it fits or not...
    regs.new_region_in(_at, 6219.01, 6219.53, 6219.2801, dlambda = lambda w: 0.972*np.max(w[1:] - w[:-1])),

    # Line at: 6481.85 or 6481.86 or [6481.87] (maybe: 6481.8 to 6481.9)
    # A SHIFT OF 0.004 SEEMS MAYBE BEST, GET IT WITH: dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])
    # IF A SHIFT OF 0.006 IS BETTER, USE: dlambda = lambda w: 0.98*np.max(w[1:] - w[:-1])
    regs.new_region_in(_at, 6481.87 - 0.25, 6481.87 + 0.25, 6481.8694, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

    # Line at: 6498.93 or 6498.94 (maybe: 6498.9 to 6498.97)
    # Currently we get a shift of about: 0.004 Å
    regs.new_region_in(_at, 6498.938 - 0.3, 6498.938 + 0.3, 6498.9379, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

    # Line at: 5778.44 or 5778.45
    regs.new_region_in(_at, 5778.45 - 0.2, 5778.45 + 0.2, 5778.4526, dlambda = lambda w: np.mean(w[1:] - w[:-1])),
#    regs.new_region_in(_at, 5778.45 - 0.18, 5778.45 + 0.18, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # Line at: 5701.53 or 5701.54 or 5701.55
    regs.new_region_in(_at, 5701.54 - 0.3, 5701.54 + 0.3, 5701.5435, dlambda = lambda w: np.max(w[1:] - w[:-1])),

    # Line at: 6836.99 or 6837 or 6837.01
    # REFINE
    regs.new_region_in(_at, 6837 - 0.2, 6837 + 0.2, 6837.0051, dlambda = lambda w: 0.967*np.max(w[1:] - w[:-1])),

    # Line at: 5253.4617
    # REFINE
    regs.new_region_in(_at, 5253.4617 - 0.25, 5253.4617 + 0.25, 5253.4617, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),

    # Line at: 5412.7833
    # REFINE
    regs.new_region_in(_at, 5412.7833 - 0.2, 5412.7833 + 0.2, 5412.7833, dlambda = lambda w: 0.95*np.max(w[1:] - w[:-1])),

    # Line at: 6750.1510
    regs.new_region_in(_at, 6750.1510 - 0.3, 6750.1510 + 0.3, 6750.1510),

    # Line at: 5705.4639
    # REFINE: WOULD BE BETTER IF IT COULD BE PUSHED 0.002 TO THE LEFT!!! (TO 0.008 FROM 0.006)
    regs.new_region_in(_at, 5705.4639 - 0.25, 5705.4639 + 0.25, 5705.4639, dlambda = lambda w: 0.991*np.max(w[1:] - w[:-1])),

    # Line at: 5956.6933
    regs.new_region_in(_at, 5956.6933 - 0.3, 5956.6933 + 0.3, 5956.6933, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # **** STRONG LINES **** 6173.3339 6173.3347
#    regs.new_region_in(_at, 6173.3339 - 0.25, 6173.3339 + 0.25, dlambda = lambda w: 1.01*np.mean(w[1:] - w[:-1])),
    regs.new_region_in(_at, 6173.3339 - 0.25, 6173.3339 + 0.22, 6173.3339, dlambda = lambda w: 1.01*np.mean(w[1:] - w[:-1])),
    
    # Line at: 5232.9397
    regs.new_region_in(_at, 5232.9397 - 1.0, 5232.9397 + 1.0, 5232.9397, dlambda = lambda w: 0.9585*np.max(w[1:] - w[:-1])),
]

_EXTRA_WIDTH = 0.15
_wide_regions = [
    # Line at: 6301.5 Å
    regs.new_region_in(_at, 6301.18 - _EXTRA_WIDTH, 6301.82 + _EXTRA_WIDTH, 6301.4999, dlambda = lambda w: 0.992*np.max(w[1:] - w[:-1])),

    # Line at: 6302.5 Å
    # POSSIBLY (BUT DOUBTFULLY) BETTER:
    #    6302.25 to 6302.75, gives shift of 0.004, cannot see much difference compared to the current
    #    interval 6302.25 to 6302.75. The chi squared is however higher for all abundencies, so it might
    #    just be worse.
    regs.new_region_in(_at, 6302.3 - _EXTRA_WIDTH, 6302.7 + _EXTRA_WIDTH, 6302.4935),

    # Line at: [6219.27] or 6219.28
    # CANDIDATES:
    #    dlambda = lambda w: 0.98*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.002... looks like a bit higher shift could be good
    #    dlambda = lambda w: 0.975*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.004... not sure if it fits or not...
    regs.new_region_in(_at, 6219.01 - _EXTRA_WIDTH, 6219.53 + _EXTRA_WIDTH, 6219.2801, dlambda = lambda w: 0.972*np.max(w[1:] - w[:-1])),

    # Line at: 6481.85 or 6481.86 or [6481.87] (maybe: 6481.8 to 6481.9)
    # A SHIFT OF 0.004 SEEMS MAYBE BEST, GET IT WITH: dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])
    # IF A SHIFT OF 0.006 IS BETTER, USE: dlambda = lambda w: 0.98*np.max(w[1:] - w[:-1])
    regs.new_region_in(_at, 6481.87 - 0.25 - _EXTRA_WIDTH, 6481.87 + 0.25 + _EXTRA_WIDTH, 6481.8694, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

    # Line at: 6498.93 or 6498.94 (maybe: 6498.9 to 6498.97)
    # Currently we get a shift of about: 0.004 Å
    regs.new_region_in(_at, 6498.938 - 0.3 - _EXTRA_WIDTH, 6498.938 + 0.3 + _EXTRA_WIDTH, 6498.9379, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

    # Line at: 5778.44 or 5778.45
    regs.new_region_in(_at, 5778.45 - 0.2 - _EXTRA_WIDTH, 5778.45 + 0.2 + _EXTRA_WIDTH, 5778.4526, dlambda = lambda w: np.mean(w[1:] - w[:-1])),
#    regs.new_region_in(_at, 5778.45 - 0.18, 5778.45 + 0.18, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # Line at: 5701.53 or 5701.54 or 5701.55
    regs.new_region_in(_at, 5701.54 - 0.3 - _EXTRA_WIDTH, 5701.54 + 0.3 + _EXTRA_WIDTH, 5701.5435, dlambda = lambda w: np.max(w[1:] - w[:-1])),

    # Line at: 6836.99 or 6837 or 6837.01
    # REFINE
    regs.new_region_in(_at, 6837 - 0.2 - _EXTRA_WIDTH, 6837 + 0.2 + _EXTRA_WIDTH, 6837.0051, dlambda = lambda w: 0.967*np.max(w[1:] - w[:-1])),

    # Line at: 5253.4617
    # REFINE
    regs.new_region_in(_at, 5253.4617 - 0.25 - _EXTRA_WIDTH, 5253.4617 + 0.25 + _EXTRA_WIDTH, 5253.4617, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),

    # Line at: 5412.7833
    # REFINE
    regs.new_region_in(_at, 5412.7833 - 0.2 - _EXTRA_WIDTH, 5412.7833 + 0.2 + _EXTRA_WIDTH, 5412.7833, dlambda = lambda w: 0.95*np.max(w[1:] - w[:-1])),

    # Line at: 6750.1510
    regs.new_region_in(_at, 6750.1510 - 0.3 - _EXTRA_WIDTH, 6750.1510 + 0.3 + _EXTRA_WIDTH, 6750.1510),

    # Line at: 5705.4639
    # REFINE: WOULD BE BETTER IF IT COULD BE PUSHED 0.002 TO THE LEFT!!! (TO 0.008 FROM 0.006)
    regs.new_region_in(_at, 5705.4639 - 0.25 - _EXTRA_WIDTH, 5705.4639 + 0.25 + _EXTRA_WIDTH, 5705.4639, dlambda = lambda w: 0.991*np.max(w[1:] - w[:-1])),

    # Line at: 5956.6933
    regs.new_region_in(_at, 5956.6933 - 0.3 - _EXTRA_WIDTH, 5956.6933 + 0.3 + _EXTRA_WIDTH, 5956.6933, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # **** STRONG LINES **** 6173.3339 6173.3347
#    regs.new_region_in(_at, 6173.3339 - 0.25, 6173.3339 + 0.25, dlambda = lambda w: 1.01*np.mean(w[1:] - w[:-1])),
    regs.new_region_in(_at, 6173.3339 - 0.25 - _EXTRA_WIDTH, 6173.3339 + 0.22 + _EXTRA_WIDTH, 6173.3339, dlambda = lambda w: 1.01*np.mean(w[1:] - w[:-1])),
    
    # Line at: 5232.9397
    regs.new_region_in(_at, 5232.9397 - 1.0 - _EXTRA_WIDTH, 5232.9397 + 1.0 + _EXTRA_WIDTH, 5232.9397, dlambda = lambda w: 0.9585*np.max(w[1:] - w[:-1])),
]

def get_regions():
    """
    Returns the regions
    """
    
    return list(_regions)

def get_wide_regions():
    """
    Returns the wide regions
    """
    
    return list(_wide_regions)

def region_at(wavelength, region_list = _regions):
    """
    Gets the index of the region and the region at the given wavelength
    """
    found = None
    for i, r in enumerate(region_list):
        if r.lambda0 <= wavelength and wavelength <= r.lambda_end:
            found = i
            break
    if found == None:
        raise Exception("No region found for wavelength " + str(wavelength))
    return found, region_list[i]

def refine(region_list, wavelength, left, right, dlambda = None, nlambda = None):
    i, r = region_at(wavelength, region_list = region_list)
    region_list[i] = r.refine(left, right, dlambda = dlambda, nlambda = nlambda)

