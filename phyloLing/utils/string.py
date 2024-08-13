"""MISC STRING MANIPULATION FUNCTIONS"""

import re

from asjp import ipa2asjp
from unidecode import unidecode


def strip_ch(string, to_remove):
    """Removes a set of characters from strings"""
    to_remove_regex = '|'.join(to_remove)
    string = re.sub(to_remove_regex, '', string)
    return string


def format_as_variable(string):
    variable = unidecode(string)
    variable = re.sub(r"[-\s'\(\)]", '', variable)
    return variable


def asjp_in_ipa(ipa_string):
    """Convert some non-IPA ASJP characters to IPA equivalents. 
    Preserves set of ASJP characters/mapping, but keeps IPA compatibility."""
    ipa_string = ipa2asjp(ipa_string)
    ipa_string = re.sub('~', '', ipa_string)
    ipa_string = re.sub('4', 'n̪', ipa_string)
    ipa_string = re.sub('5', 'ɲ', ipa_string)
    ipa_string = re.sub('N', 'ŋ', ipa_string)
    ipa_string = re.sub('L', 'ʎ', ipa_string)
    ipa_string = re.sub('c', 'ʦ', ipa_string)
    ipa_string = re.sub('T', 'c', ipa_string)
    ipa_string = re.sub('g', 'ɡ', ipa_string)
    ipa_string = re.sub('G', 'ɢ', ipa_string)
    ipa_string = re.sub('7', 'ʔ', ipa_string)
    ipa_string = re.sub('C', 'ʧ', ipa_string)
    ipa_string = re.sub('j', 'ʤ', ipa_string)
    ipa_string = re.sub('8', 'θ', ipa_string)
    ipa_string = re.sub('S', 'ʃ', ipa_string)
    ipa_string = re.sub('Z', 'ʒ', ipa_string)
    ipa_string = re.sub('X', 'χ', ipa_string)
    ipa_string = re.sub('y', 'j', ipa_string)
    ipa_string = re.sub('E', 'ɛ', ipa_string)
    ipa_string = re.sub('3', 'ə', ipa_string)
    ipa_string = re.sub(r'\*', '̃', ipa_string)
    return ipa_string
