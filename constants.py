TRANSCRIPTION_PARAM_DEFAULTS = {
    'asjp':False,
    'ignore_stress':False,
    'combine_diphthongs':True,
    'normalize_geminates':False,
    'preaspiration':True,
    'ch_to_remove':{' '}, # TODO add syllabic diacritics here
    'suprasegmentals':None,
    'level_suprasegmentals':None,
    }

ALIGNMENT_PARAM_DEFAULTS = {
    'gap_ch':'-', # alternative null character <∅> looks too similar to IPA character <ø>
    'pad_ch':'#',
}

# SPECIAL CHARACTERS FOR JOINING AND/OR DELIMITING PHON CORRS
SEG_JOIN_CH = '_'
PHON_ENV_JOIN_CH = ';'
START_PAD_CH = '<'
END_PAD_CH = '>'
SPECIAL_JOIN_CHS = [
    SEG_JOIN_CH,
    PHON_ENV_JOIN_CH,
    START_PAD_CH,
    END_PAD_CH,
]