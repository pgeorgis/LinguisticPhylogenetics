# Initialize logger
import logging
import re
from phonUtils.initPhoneData import pre_diacritics, post_diacritics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

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

# ALIGNMENT DELIMITERS
ALIGNMENT_DELIMITER = "-------------------"
ALIGNMENT_POSITION_DELIMITER = " / "
ALIGNED_PAIR_DELIMITER = "-"

# ALIGNMENT KEY REGEX
ALIGNMENT_KEY_REGEX = re.compile(r"/(.+)/ - /(.+)/")

# DEFAULT PARAMETERS
TRANSCRIPTION_PARAM_DEFAULTS = {
    'asjp': False,
    'ignore_stress': False,
    'combine_diphthongs': True,
    'normalize_geminates': False,
    'preaspiration': True,
    'ch_to_remove': {' '},  # TODO add syllabic diacritics here
    'autonomous_diacritics': None,
    'min_phone_instances': 2,  # minimum instances of a phone in a doculect to be considered valid, else issue a warning
}
STRESS_DIACRITICS = {'ˌ', 'ˈ'}
PRE_DIACRITICS = set(pre_diacritics)
POST_DIACRITICS = set(post_diacritics)

ALIGNMENT_PARAM_DEFAULTS = {
    'gap_ch': GAP_CH_DEFAULT,
    'pad_ch': PAD_CH_DEFAULT,
}

# NON-IPA CHARACTER
# Used for finding OOV values
NON_IPA_CH_DEFAULT = '<NON-IPA>'

# COLUMN NAMES IN DATA FILES
ID_COLUMN_LABEL = 'ID'
LANGUAGE_NAME_LABEL = 'Language_ID'
CONCEPT_LABEL = 'Parameter_ID'
ORTHOGRAPHY_LABEL = 'Value'
PHONETIC_FORM_LABEL = 'Form'
SEGMENTS_LABEL = 'Segments'
COGNATE_CLASS_LABEL = 'Cognate_ID'
LOAN_LABEL = 'Loan'
GLOTTOCODE_LABEL = 'Glottocode'
ISO_CODE_LABEL = 'ISO 639-3'

# FAMILY INDEX KEYS
DOCULECT_INDEX_KEY = "doculects"
PHONE_CORRELATORS_INDEX_KEY = "phone_correlators"
