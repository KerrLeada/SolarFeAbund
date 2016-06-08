from __future__ import print_function

# Get the atlas
import satlas as _sa
_at = _sa.satlas()

# Create the regions
regions = [
    # Line at: 6301.5 Å
    regs.new_region_in(_at, 6301.18, 6301.82),

    # Line at: 6302.5 Å
    # POSSIBLY (BUT DOUBTFULLY) BETTER:
    #    6302.25 to 6302.75, gives shift of 0.004, cannot see much difference compared to the current
    #    interval 6302.25 to 6302.75. The chi squared is however higher for all abundencies, so it might
    #    just be worse.
    regs.new_region_in(_at, 6302.3, 6302.7),

    # Line at: [6219.27] or 6219.28
    # CANDIDATES:
    #    dlambda = lambda w: 0.98*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.002... looks like a bit higher shift could be good
    #    dlambda = lambda w: 0.975*np.max(w[1:]-w[:-1])  <---- Gives a shift of 0.004... not sure if it fits or not...
    regs.new_region_in(_at, 6219.01, 6219.53, dlambda = lambda w: 0.972*np.max(w[1:] - w[:-1])),

    # Line at: 6481.85 or 6481.86 or [6481.87] (maybe: 6481.8 to 6481.9)
    # SHOULD POSSIBLY TRY AND PUSH THE SHIFT TO 0.004, WHICH MIGHT FIT BETTER! (KEYWORD IS MIGHT!)
    regs.new_region_in(_at, 6481.87 - 0.25, 6481.87 + 0.25, dlambda = lambda w: 0.98*np.max(w[1:] - w[:-1])),

    # Line at: 6498.93 or 6498.94 (maybe: 6498.9 to 6498.97)
    # Currently we get a shift of about: 0.004 Å
    regs.new_region_in(_at, 6498.938 - 0.3, 6498.938 + 0.3, dlambda = lambda w: 0.982*np.max(w[1:] - w[:-1])),

    # Line at: 5778.44 or 5778.45            
    regs.new_region_in(_at, 5778.45 - 0.2, 5778.45 + 0.2, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # Line at: 5701.53 or 5701.54 or 5701.55
    regs.new_region_in(_at, 5701.54 - 0.3, 5701.54 + 0.3, dlambda = lambda w: np.max(w[1:] - w[:-1])),

    # Line at: 6836.99 or 6837 or 6837.01
    # REFINE
    regs.new_region_in(_at, 6837 - 0.2, 6837 + 0.2, dlambda = lambda w: 0.967*np.max(w[1:] - w[:-1])),

    # Line at: 5253.4617
    # REFINE
    regs.new_region_in(_at, 5253.4617 - 0.25, 5253.4617 + 0.25, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),

    # Line at: 5412.7833
    # REFINE
    regs.new_region_in(_at, 5412.7833 - 0.2, 5412.7833 + 0.2, dlambda = lambda w: 0.95*np.max(w[1:] - w[:-1])),

    # Line at: 6750.1510
    regs.new_region_in(_at, 6750.1510 - 0.3, 6750.1510 + 0.3),

    # Line at: 5705.4639
    # REFINE: WOULD BE BETTER IF IT COULD BE PUSHED 0.002 TO THE LEFT!!! (TO 0.008 FROM 0.006)
    regs.new_region_in(_at, 5705.4639 - 0.25, 5705.4639 + 0.25, dlambda = lambda w: 0.991*np.max(w[1:] - w[:-1])),

    # Line at: 5956.6933
    regs.new_region_in(_at, 5956.6933 - 0.3, 5956.6933 + 0.3, dlambda = lambda w: np.mean(w[1:] - w[:-1])),

    # **** STRONG LINES **** 6173.3339 6173.3347
    #            regs.new_region_in(_at, 6173.3339 - 0.3, 6173.3339 + 0.3),
    #            regs.new_region_in(_at, 6173.3347 - 0.3, 6173.3347 + 0.3),
    # Line at: 5232.93 or 5232.94 or 5232.95
    # CANDIDATES:
    #    dlambda = lambda w: 0.96*np.max(w[1:] - w[:-1])      <---- shift 0.002
    #    dlambda = lambda w: np.mean(w[1:] - w[:-1])          <---- shift 0.004   (best chi squared)
    #            regs.new_region_in(_at, 5232.94 - 1.5, 5232.94 + 1.5, dlambda = lambda w: 0.955*np.max(w[1:] - w[:-1])),

    # Line at: 4957.29 or 4957.3 or 4957.31
    #            regs.new_region_in(_at, 4957.3 - 1.5, 4957.3 + 1.5),

    # Line at: 4890.74 or 4890.75 or 4890.76 or 4890.77
    # Regarding dlambda: The synthetic line should be shifted to the left, towards lower wavelengths. If dlambda is the mean wavelength difference
    # or higher the synthetic line is shifted towards the right (higher wavelengths). Meanwhile, if dlambda is the minimum wavelength difference
    # it is shifted too much to the left. As such, we can conclude that dlambda should be between the minimum and the mean wavelength difference.
    #            regs.new_region_in(_at, 4890.75 - 1.5, 4890.75 + 1.5, dlambda = lambda w: 0.97*np.max(w[1:] - w[:-1])),
]
