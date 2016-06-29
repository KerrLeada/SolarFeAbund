

class Number(object):
    def __init__(self, column_data):
        self.column_data = column_data
    
    def __len__(self):
        return len(self.column_data)
    
    def __getitem__(self, key):
        return "$" + str(self.column_data[key]) + "$"

def gen_table(first_column, *columns):
    
    table = [first_column].extend(columns)
    row_count = len(first_column)
    col_count = len(table)
    table_text = ""
    for r in range(row_count):
        table_text += str(table[0][r])
        for c in range(1, col_count):
            table_text += " & " + str(table[c][r])
        table_text += r"\\"
    return table_text
