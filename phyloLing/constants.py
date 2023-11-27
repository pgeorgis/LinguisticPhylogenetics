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

# GAP CHARACTER: DENOTES GAPS IN ALIGNMENTS OR NULL SEGMENTS IN CORRESPONDENCES
GAP_CH_DEFAULT = '-'
# Note: alternative null character <∅> looks too similar to IPA character <ø>
# but is used in "visual alignments" where "-" delimits aligned units
NULL_CH_DEFAULT = '∅' 

# PAD CHARACTER: USED FOR PADDING EDGES OF ALIGNMENTS
PAD_CH_DEFAULT = '#'

# DEFAULT PARAMETERs
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
    'gap_ch':GAP_CH_DEFAULT, 
    'pad_ch':PAD_CH_DEFAULT,
}