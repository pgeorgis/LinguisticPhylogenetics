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