"""
This module contains the functions that handles cfg files.
"""

from __future__ import print_function

import astropy.units as units
import astropy.constants as consts

_CFG_COLUMNS = ["Label", "Elem", "ion", "anum", "wav", "loggf", "j_low", "j_up", "g_low", "g_up", "e_low", "gam_rad", "gam_strk", "vdW/Bark", "wid"]
_MAX_NAME_LENGTH = max(map(len, _CFG_COLUMNS))

def get_column(cfg_data, column, dtype = str):
    """
    Returns a list of the data in the column with the given name. Optionally
    takes a datatype to convert the individual data elements into.
    """
    column_index = _CFG_COLUMNS.index(column)
    return [dtype(r[column_index]) for r in cfg_data]

def read_cfg(filename):
    """
    Reads the given cfg file and returns the data as a list of lists.
    """

    # Read the entire file and get the content
    with open(filename, "r") as cfg_file:
        content_str = cfg_file.read()
    lines = content_str.split("\n")
    content = []

    # Loop through each line, and then split it into the columns
    for l in lines:
        l = l.strip().replace("\t", " ")
        if len(l) > 0 and not l.startswith("#"):
            data = [x for x in l.split(" ") if len(x) > 0]
            if len(data) != len(_CFG_COLUMNS):
                raise Exception("Row had length " + str(len(data)) + " but should have had length " + str(len(_CFG_COLUMNS)))
            
            # Add the column data to the content
            content.append(data)
    return content

def print_cfg(cfg_data):
    """
    Prints the given cfg data. If cfg_data doesn't have the right amount of columns, an exception is thrown.
    """

    # Check row size (doing this now to get fast feedback, even if it means I have to loop through the rows twice)
    for row in cfg_data:
        if len(row) != len(_CFG_COLUMNS):
            raise Exception("Row had length " + str(len(row)) + " but should have had length " + str(len(_CFG_COLUMNS)))

    #
    for row in cfg_data:
        print("\n ************* \n")
        for i in range(len(_CFG_COLUMNS)):
            spaces = " "*(_MAX_NAME_LENGTH - len(_CFG_COLUMNS[i]))
            print(_CFG_COLUMNS[i] + spaces + " :", row[i])

def print_cfg_file(filename):
    """
    Prints the given cfg file.
    """

    # Not too concerned with efficiency, since the amount of data is small
    print_cfg(read_cfg(filename))

def _split_at(sep, string):
    """
    Splits the given string using the given separator. Empty strings and strings
    starting with # are not included in the result.
    """
    
    result = string.split(sep)
    return [x for x in map(str.strip, result) if len(x) > 0 and not x.startswith("#")]

def format_read_cfg(source_filename, energy_eV = False):
    """
    KIND OF IMPLEMENTED, BUT NOT DOCUMENTED YET!
    """
    
    with open(source_filename, "r") as source_file:
        content = source_file.read()
    
    # Remove all ( number)
    for i in range(10):
        content = content.replace("( " + str(i) + ")", "")

    # Replace tabs with spaces
    content = content.replace("\t", " ")

    # Split the content into lines and filter out any empty lines
    lines = _split_at("\n", content)

    # Make sure there is an even number of lines remaining
    if len(lines) % 2 != 0:
        raise Exception("Wrong format")

    data_order = ["wav", "loggf", "j_low", "j_up", "g_low", "g_up", "e_low", "gam_rad", "gam_strk", "vdW/Bark", "wid"]
    src_order1 = ["wav", "Elem", "ion", "loggf", "e_low", "j_low", "e_up", "j_up"]
    src_order2 = ["?", "g_low", "g_up", "gam_rad", "gam_strk", "vdW/Bark"]
    src1 = {src_order1[i]: i for i in range(len(src_order1))}
    src2 = {src_order2[i]: i for i in range(len(src_order2))}
#    data = {quantity: [] for quantity in data_order}
    data = ""
    wav = []
    for i in xrange(len(lines)/2):
        dl1 = lines[2*i]
        dl2 = lines[2*i + 1]
        
        cols1 = _split_at(" ", lines[2*i])
        cols2 = _split_at(" ", lines[2*i + 1])

        data += "FeI_" + str(int(float(cols1[src1["wav"]])))
        data += "    Fe    1    26    "
        data += cols1[src1["wav"]] + "    "
        data += cols1[src1["loggf"]] + "    "
        data += cols1[src1["j_low"]] + "    "
        data += cols1[src1["j_up"]] + "    "
        data += cols2[src2["g_low"]] + "    "
        data += cols2[src2["g_up"]] + "    "
        if energy_eV:
            data += cols1[src1["e_low"]]
        else:
            data += str((consts.h*consts.c*(float(cols1[src1["e_low"]]) * (1/units.cm))).to(units.eV).value)
        data += "    "
        data += cols2[src2["gam_rad"]] + "    "
        data += cols2[src2["gam_strk"]] + "    "
        data += cols2[src2["vdW/Bark"]] + "    "
        data += "1.0" # <---- SHOULD PROBABLY NOT HARDCODE THIS... FIX IT!!!
        data += "\n"
        print("FIX wid!!! Right now it's hardcoded to 1.0!!! FIX IT!!!")

        wav.append(float(cols1[src1["wav"]]))
    return data, wav
#    raise Exception("Not implemented...")

def create_cfg_file(source_filename, cfg_filename, reg_filename = None, energy_eV = False):
    """
    Creates a new cfg file from the given source file. The source file is handled by format_read_cfg.
    """
    
    data, wav = format_read_cfg(source_filename, energy_eV = energy_eV)
    with open(cfg_filename, "w") as cfg_file:
        cfg_file.write(data)
    if reg_filename != None:
        reg_content = "wav = [\n"
        for w in wav:
            reg_content += "    " + str(w) + ",\n"
        reg_content += "]\n"
        reg_content += "def create_regions(offset, dlambda, nlambda, cont):\n"
        reg_content += "    return [(w - offset, dlambda, nlambda, cont) for w in wavelengths]\n"
        with open(reg_filename, "w") as reg_file:
            reg_file.write(reg_content)

