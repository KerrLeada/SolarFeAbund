

class NumericColumn(object):
    def __init__(self, column_data, fmt = None):
        """
        Constructor of the NumericColumn class. Creates a NumericColumn object, which
        is an list that represents a numeric column in a latex table. The required
        argument is
        
            column_data : The list of numbers in the column
        
        The optional argument is
        
            fmt : A formatting option that sets how numbers should be displayed.
        """
        
        self.column_data = column_data
        if fmt == None:
            self.formatter = str
        else:
            self.formatter = lambda x: fmt.format(x)
    
    def __len__(self):
        return len(self.column_data)
    
    def __getitem__(self, key):
        return "$" + self.formatter(self.column_data[key]) + "$"
    
    def __iter__(self):
        for c in self.column_data:
            yield "$" + self.formatter(c) + "$"

def numbers(column_elements, fmt = None):
    return NumericColumn(column_elements, fmt = fmt)

def gen_table(columns, sort_after = 0):
    # Generate the table
    if sort_after == None:
        table = zip(*columns)
    else:
        table = sorted(zip(*columns), key = lambda row: row[sort_after])
    
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
