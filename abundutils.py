"""
This module contains helper functions for dealing with abundencies.
"""

def abund(elem, abundency):
    """
    Returns an new abundance for the element. The data is in the form
    of [[elem, abundency]].
    """
    return [[elem, abundency]]

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
