"""
This module contains helper functions for dealing with abundencies.
"""

import numpy as np

# An empty abundance
EMPTY_ABUND = [[]]

def abund(elem, abundency):
    """
    Returns an new abundance for the element. The arguments are
    
        elem      : The element.
        
        abundance : The abundance of the element.
    
    The returned data is in the form of [[elem, abundency]].
    """
    
    return [[elem, abundency]]

def list_abund(abund, default_val = None):
    """
    Returns a list of the abundance values, assuming that only one element is updated every time.
    Substritutes the default abundance with the given value. The required argument is
        
        abund : A list of abundances.
        
    The optional argument is
    
        default_val : The value to use in place of the default abundance.
    
    Returns a list of the abundance values.
    """
    
    abund_vals = np.zeros(len(abund), dtype = np.float64)
    for i, a in enumerate(abund):
        if len(a) == 0:
            if default_val == None:
                raise Exception("The default abundance was found but no default value to use was given")
            abund_vals[i] = default_val
        elif len(a) == 1:
            abund_vals[i] = a[0][1]
        else:
            raise Exception("There must only be one updated abundance")
    return abund_vals

def get_value(abund):
    """
    Gets the first abundance value. Any other are ignored. The argument is
    
        abund :  A list of abundances.
    """
    
    return abund[0][1]

def check_abund(abund):
    """
    Checks so the elements in the abundencies will not cause a buffer overflow. An exception is thrown
    if an element would cause a buffer overflow. The argument is
    
        abund : A list of abundances.
    
    Returns nothing.
    
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
    The argument is
    
        elem : The element abundance to check the name for. It has the form: [element name, abundance].
    
    Returns nothing.
    
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
