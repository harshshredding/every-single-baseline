from preamble import *

def replace_text(old_text, new_text, file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    data = data.replace(old_text, new_text)
    with open(file_path, 'w') as file:
        file.write(data)

# Personalize pudb
replace_text('theme = classic', 'theme = monokai', '~/.config/pudb/pudb.cfg')
replace_text('shell = internal', 'shell = classic', '~/.config/pudb/pudb.cfg')
