# -*- coding: utf8 -*-

from __future__ import print_function
from __future__ import division

import plotting
import latexgen
import numpy as np
import matplotlib.pyplot as plt

def display(result_pair):
    """
    Displays everything that should be displayed. This function may change on an arbitrary basis.
    """
    # Get the results
    result_chi = result_pair.result_chi
    result_ew = result_pair.result_ew
    
    # Get the regions and the lines
    regions = [r.region for r in result_chi.region_result]
    lines = [r.lab_wav for r in regions]
    
    # Function that helps with creating number columns in latex tables
    numbers = latexgen.numbers
    
    # Generating the table for the results obtained using chi squared
    print("Latex table: chi²")
    best_shifts = [r.best_shift for r in result_chi.region_result]
#    doppler_vels = [_calc_vel(r.best_shift, r.wav[np.argmin(r.inten[r.best_index])]) for r in result_chi.region_result]
    doppler_vels = [_calc_vel(r.best_shift, r.region.lab_wav) for r in result_chi.region_result]
    chi2 = result_chi.minimized_quantity
    print(latexgen.gen_table([lines, numbers(_abund(result_chi.best_abunds), fmt = "{:0.3f}"), numbers(best_shifts, fmt = "{:0.3f}"), doppler_vels, chi2], number_fmt = "{:0.4f}"))
    
    # Generating the table for the results obtained using equivalent widths
    print("\n\nLatex table: equivalent widths")
    ew_obs = [r.obs_eq_width for r in result_ew.region_result]
    ew_synth = [r.best_eq_width for r in result_ew.region_result]
    ew_diff = [abs(r.best_diff) for r in result_ew.region_result]
    print(latexgen.gen_table([lines, numbers(_abund(result_ew.best_abunds), fmt = "{:0.3f}"), ew_obs, ew_synth, ew_diff], number_fmt = "{:0.4f}"))
    
    # Calculate the differences in abundance between the best synthetic lines obtained by chi squared and equivalent widths
    abund_diffs = [r_chi.best_abund - r_ew.best_abund for r_chi, r_ew in zip(result_chi.region_result, result_ew.region_result)]
    
    # Generate a table over the difference in abundance between the best synthetic lines obtained by chi squared and equivalent widths
    print("\n\nLatex table: differences between abundance derived using chi² and equivalent widths")
    print(latexgen.gen_table([numbers(lines, fmt = "{:0.4f}"), _abund(result_chi.best_abunds), _abund(result_ew.best_abunds), abund_diffs], number_fmt = "{:0.3f}"))
    
    # Print the mean difference in abundance between the best synthetic lines obtained by chi squared and equivalent widths, as well as the standard deviation
    print("\n\nAbund diff:", np.mean(abund_diffs), "+-", np.std(abund_diffs), "    <---- This is the mean and standard deviation")

def print_best(result_pair):
    """
    Prints the best result from the given result pair.
    """
    
    # Get the results
    result_chi = result_pair.result_chi
    result_ew = result_pair.result_ew

    # Print the results for each region
    for i, (r_chi, r_ew) in enumerate(zip(result_chi.region_result, result_ew.region_result)):
#        lambda_em = r_chi.wav[np.argmin(r_chi.inten[r_chi.best_index])]
        lambda_em = r_chi.region.lab_wav
        print("Region nr ", i,": ", r_chi.region, sep = "")
        print("    Chi squared method:")
        print("        Best chisq:", r_chi.best_chisq)
        print("        Best shift:", r_chi.best_shift)
        print("        Best abund:", _abund(r_chi.best_abund), "     or:", r_chi.best_abund, "(as -12 + best abund)")
        print("        Velocity: ~", _calc_vel(r_chi.best_shift, lambda_em), "     ( Using delta_lambda =", r_chi.best_shift, "lambda_em =", lambda_em, ")")
        print("    Equivalent widths:")
        print("        Best eq width:", r_ew.best_eq_width)
        print("        Obs eq width: ", r_ew.obs_eq_width)
        print("        Best diff:    ", r_ew.best_diff)
        print("        Best abund:   ", _abund(r_ew.best_abund), "     or:", r_ew.best_abund, "(as -12 + best abund)")
        print("")
    
    # Print the final results
    for result, method in [(result_chi, "chi squared"), (result_ew, "equivalent widths")]:
        print("Result using", method, "yields:")
        print("    Best abunds:", _abund(result.best_abunds))
        print("    Min abund: ", min(_abund(result.best_abunds)), "     or:", min(result.best_abunds), "(as -12 + min abund)")
        print("    Max abund: ", max(_abund(result.best_abunds)), "     or:", max(result.best_abunds), "(as -12 + max abund)")
        print("    Mean abund:", _abund(result.abund), "+-", result.error_abund, "     or:", result.abund, "+-", result.error_abund, "(as -12 + mean abund)")

def list_best(result, show_together = True):
    # Get the results
    result_chi = result.result_chi
    result_ew = result.result_ew
    
    # Create the rows for the result from chi squared
    regres_chi = [(i, rr.region.lab_wav, abs(rr.best_abund - result_chi.abund)) for i, rr in enumerate(result_chi.region_result)]
    regres_chi = sorted(regres_chi, key = lambda rr: rr[2])
    table_chi = []
    for i, lab_wav, diffabund in regres_chi:
        i_str = str(i) if i >= 10 else " " + str(i)
        table_chi.append("| " + i_str + "  |  {:0.4f}  |            {:0.3f} |".format(lab_wav, diffabund))
    
    # Create the rows for the result from equivalent widths
    regres_ew = [(i, rr.region.lab_wav, abs(rr.best_abund - result_ew.abund)) for i, rr in enumerate(result_ew.region_result)]
    regres_ew = sorted(regres_ew, key = lambda rr: rr[2])
    table_ew = []
    for i, lab_wav, diffabund in regres_ew:
        i_str = str(i) if i >= 10 else " " + str(i)
        table_ew.append("| " + i_str + "  |  {:0.4f}  |           {:0.4f} |".format(lab_wav, diffabund))
    
    # Create the horizontal bars, titles and column names
    hbar = "----------------------------------------"
    title_chi = "|             Chi squared              |"
    title_ew = "|          Equivalent widths           |"
    colnames = "| Nr  |  wav        |  abs(abund diff) |"
    
    # Determine if the results should be shown side by side or one after another, and print the results in the
    # appropriate way
    if show_together:
        # Print the results from chi squared to the left and the results from equivalent widths to the right
#        spaces = " "*(len(colnames) + 11 - len(table_chi[0]))
        print(hbar + "           " + hbar)
        print(title_chi + "           " + title_ew)
        print(hbar + "           " + hbar)
        print(colnames + "           " + colnames)
        print(hbar + "           " + hbar)
        for rc, re in zip(table_chi, table_ew):
#            print(rc + spaces + re)
            print(rc + "           " + re)
        print(hbar + "           " + hbar)
    else:
        # Print the chi squared result
        print(hbar)
        print(title_chi)
        print(hbar)
        print(colnames)
        print(hbar)
        for rc in table_chi:
            print(rc)
        print(hbar)
        
        # Print the equivalent width result
        print("")
        print(hbar)
        print(title_ew)
        print(hbar)
        print(colnames)
        print(hbar)
        for re in table_ew:
            print(re)
        print(hbar)

def _abund(abund):
    """
    Converts the abundance to the correct convention
    """
    
    if isinstance(abund, list):
        return [12.0 + a for a in abund]
    return 12.0 + abund

def _calc_vel(delta_lambda, lambda_em):
    """
    Calculates the velocity that corresponds to a Doppler shift
    with a given shift delta_lambda and an emitted wavelength lambda_em.
    """
    
    return delta_lambda*300000.0/lambda_em
