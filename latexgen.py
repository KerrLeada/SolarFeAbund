from __future__ import print_function
from __future__ import division

class LatexNumber(object):
    def __init__(self, number, fmt = None):
        """
        Constructor of the LatexNumber class. Creates a LatexNumber object, which
        contains a number that should be written as a latex string, if converted
        to a string. The required argument is
        
            number : The latex number.
        
        The optional argument is
        
            fmt : A formatting option that sets how the number should be displayed.
        """
        
        self.number = number
        if fmt == None:
            self.formatter = str
        elif hasattr(fmt, "__call__"):
            self.formatter = fmt
        else:
            self.formatter = lambda x: fmt.format(x)
    
    def __cmp__(self, other):
        if isinstance(other, LatexNumber):
            other = other.number
        return cmp(self.number, other)
    
    def __str__(self):
        return "$" + self.formatter(self.number) + "$"
    
    def __repr__(self):
        return "LatexNumber(" + str(self.number) + ")"

def numbers(column_elements, fmt = None):
    """
    Creates a list of LatexNumbers.
    """
    
    return [LatexNumber(e, fmt = fmt) for e in column_elements]

def _auto_conv_elem(elem, number_fmt):
    """
    Converts to a LatexNumber instance if elem is an int, a long or a float. Otherwise the original object is returned.
    """
    
    if isinstance(elem, (int, long, float)):
        elem = LatexNumber(elem, fmt = number_fmt)
    return elem

def _auto_conv_list(column, number_fmt):
    """
    Automatically converts the elements in the given list to the correct format. As in converts them to LatexNumber if
    they are instances of int, long or float while leaving the rest of the elements alone.
    """
    
    return [_auto_conv_elem(e, number_fmt) for e in column]

def gen_table(columns, sort_after = 0, auto_latex = True, number_fmt = None):
    """
    Generates the content of a latex table, using the goven columns. The required argument is
    
        columns : A list over the column data.
    
    The optional arguments are
    
        sort_after : The columns to sort the generated table after.
        
        auto_latex : Determines if the type of the elements should determine how they are printed.
                     For example, instances of int and float can be printed as $element$. If set to
                     False this is not done.
                     Default is True.
        
        number_fmt : Formats the numbers that doesn't already have a format associated with them, if
                     auto_latex is True. If set to None, nothing happens.
                     Default is None.
    """
    
    # Generate the table
    if sort_after == None:
        table = zip(*columns)
    else:
        table = sorted(zip(*columns), key = lambda row: row[sort_after])
    
    # Auto convert if needed
    if auto_latex:
        table = [_auto_conv_list(row, number_fmt) for row in table]
    
    # Generate the latex table
    row_count = len(table)
    col_count = len(columns)
    table_text = ""
    for r in range(row_count):
        table_text += str(table[r][0])
        for c in range(1, col_count):
            table_text += " & " + str(table[r][c])
        table_text += " \\\\\n"
    return table_text
