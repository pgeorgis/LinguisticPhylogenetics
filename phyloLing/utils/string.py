import re
from unidecode import unidecode

# MISC STRING MANIPULATION FUNCTIONS

def strip_ch(string, to_remove):
    """Removes a set of characters from strings"""
    to_remove_regex = '|'.join(to_remove)
    string = re.sub(to_remove_regex, '', string)
    return string

def format_as_variable(string):
    variable = unidecode(string)
    variable = re.sub("[-\s'\(\)]", '', variable)
    return variable