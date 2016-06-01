"""
This module contains helper functions for dealing with abundencies.
"""

import numpy as np

empty_abund = [[]]

def abund(elem, abundency):
    """
    Returns an new abundance for the element. The data is in the form
    of [[elem, abundency]].
    """
    return [[elem, abundency]]

def list_abund(abund, default_val = None):
    """
    Returns a list of the abundence values, assuming that only one element is updated every time.
    Ignores the default abundance.
    """
    abund_vals = np.zeros(len(abund), dtype = np.float64)
    for i, a in enumerate(abund):
        if len(a) == 0:
            if default_val == None:
                raise Exception("The default abundence was found but no default value to use was given")
            abund_vals[i] = default_val
        elif len(a) == 1:
            abund_vals[i] = a[0][1]
        else:
            raise Exception("There must only be one updated abundence")
    return abund_vals

def get_value(abund):
    """
    Gets the first abundence value. Any other are ignored.
    """
    return abund[0][1]

def unpack_abund(abund):
    """
    Unpacks the abundencies in abund into a list of element names and a list of corresponding abundency values.
    Returns both lists, first the element names and then the abundency values.
    """
    return zip(*abund)

def check_abund(abund):
    """
    Checks so the elements in the abundencies will not cause a buffer overflow. An exception is thrown
    if an element would cause a buffer overflow.
    
    If the element name is more then 2 bytes this will cause a buffer overflow in the underlying
    code, which is beyond my control. Since no element to my knowledge will be too long, this
    shouldn't be a problem. However, typos happen, and I also prefer to avoid buffer overflows out
    of principle.
    """
    
    # Prevent possible buffer overflows that occur when the name of an element is too long
    for a in abund:
        for e in a:
            _check_element(e)

def _check_element(elem):
    """
    Checks so an element name won't cause a buffer overflow. Throws an exception if it would.
    
    If the element name is more then 2 bytes this will cause a buffer overflow in the underlying
    code, which is beyond my control. Since no element to my knowledge will be too long, this
    shouldn't be a problem. However, typos happen, and I also prefer to avoid buffer overflows out
    of principle.
    """
    
    # Check so the name of the element is not too long.
    # NOTE: Not sure if this check is done correctly. The name should be at most 2 bytes long,
    #       so if len(e[0]) doesn't return the number of bytes in the string this might not
    #       prevent every buffer overflow.
    if len(elem[0]) > 2:
        raise Exception("Element name cannot have a length greater then 2.")
