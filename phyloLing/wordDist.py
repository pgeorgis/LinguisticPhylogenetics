import logging
import re
from math import sqrt
from statistics import mean

from asjp import ipa2asjp
from constants import (DOCULECT_INDEX_KEY, PAD_CH_DEFAULT,
                       PHONE_CORRELATORS_INDEX_KEY)
from nltk import edit_distance
from phonAlign import (Alignment, Gap, get_align_key, get_alignment_iter,
                       visual_align)
from phonUtils.initPhoneData import (alveolopalatal, nasals, palatal,
                                     postalveolar)
from phonUtils.phonSim import phone_sim
from phonUtils.segment import _toSegment
from utils.phoneme_map import PhonemeMap
from utils.distance import Distance, dist_to_sim, sim_to_dist
from utils.information import adaptation_surprisal
from utils.sequence import Ngram
from utils.string import preprocess_ipa_for_asjp_conversion, strip_ch
from utils.word import Word

logger = logging.getLogger(__name__)

# Designate maximum sonority as sonority of a toneme
MAX_SONORITY = _toSegment('˧').sonority
# Designate maximum prosodic environment strength
MAX_PROSODIC_ENV_STRENGTH = 7


class WordDistance(Distance):
    def eval(self, x, y, **kwargs):
        if isinstance(x, Word) and isinstance(y, Word):
            key = ((x.doculect_key, x.ipa), (y.doculect_key, y.ipa), self.hashable_kwargs)
            if key in self.measured:
                return self.measured[key]

        if (x, y, self.hashable_kwargs) in self.measured:
            return self.measured[(x, y, self.hashable_kwargs)]
        else:
            for arg, val in kwargs.items():
                self.set(arg, val)
            result = self.func(x, y, **self.kwargs)
            self.measured[(x, y, self.hashable_kwargs)] = result
            return result


def get_doculects_from_word_pair(word1, word2, family_index):
    lang1 = word1.get_doculect(family_index[DOCULECT_INDEX_KEY])
    lang2 = word2.get_doculect(family_index[DOCULECT_INDEX_KEY])
    return lang1, lang2


def get_phoneme_surprisal(lang1, lang2, family_index, ngram_size=1, **kwargs):
    """Calculate phoneme surprisal if not already done."""

    phone_correlators_index = family_index[PHONE_CORRELATORS_INDEX_KEY]
    from phonCorr import get_phone_correlator
    correlator1, family_index[PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
        lang1,
        lang2,
        phone_correlators_index=phone_correlators_index,
    )
    correlator2, _ = correlator1.get_twin(phone_correlators_index)
    if len(correlator1.surprisal_results[ngram_size]) == 0:
        correlator1.compute_phone_corrs(ngram_size=ngram_size, family_index=family_index, **kwargs)
    if len(correlator2.surprisal_results[ngram_size]) == 0:
        correlator2.compute_phone_corrs(ngram_size=ngram_size, family_index=family_index, **kwargs)
    return correlator1, correlator2


def get_pmi_dict(lang1,
                 lang2,
                 family_index,
                 **kwargs) -> PhonemeMap:
    """Calculate phoneme PMI if not already done and return PMI dict."""
    from phonCorr import get_phone_correlator
    correlator, family_index[PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
        lang1,
        lang2,
        phone_correlators_index=family_index[PHONE_CORRELATORS_INDEX_KEY],
    )
    if len(correlator.pmi_results) == 0:
        correlator.compute_phone_corrs(family_index, **kwargs)
    return correlator.pmi_results


def retrieve_saved_alignment(align_key, lang1, lang2, family_index):
    # Retrieve phone correlator and check whether this word pair has already been aligned
    # If so, return this saved alignment
    from phonCorr import get_phone_correlator
    correlator, family_index[PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
        lang1,
        lang2,
        phone_correlators_index=family_index[PHONE_CORRELATORS_INDEX_KEY],
    )
    align_log = correlator.align_log
    if align_key in align_log:
        return align_log[align_key], correlator, family_index[PHONE_CORRELATORS_INDEX_KEY]
    return None, correlator, family_index[PHONE_CORRELATORS_INDEX_KEY]


def prepare_alignment(word1, word2, family_index, **kwargs):
    """Prepares pairwise alignment of two Word objects.
    If the language of both words is specified, phoneme PMI is additionally used for the alignment, else only phonetic similarity.
    If phonetic PMI has not been previously calculated, it will be calculated automatically here.

    Args:
        word1 (Word): first Word object to align
        word2 (Word): second Word object to align

    Returns:
        Alignment: aligned object of the two words
    """
    # If language is specified for both words, incorporate their phoneme PMI for the alignment
    lang1, lang2 = get_doculects_from_word_pair(word1, word2, family_index)
    if lang1 is not None and lang2 is not None:
        align_key = get_align_key(word1.ipa, word2.ipa)
        saved_alignment, correlator, family_index[PHONE_CORRELATORS_INDEX_KEY] = retrieve_saved_alignment(
            align_key, lang1, lang2, family_index,
        )
        if saved_alignment is not None:
            return saved_alignment

        # Align the phonetic sequences
        alignment = correlator.align_wordlist(
            [(word1, word2)],
            align_costs=correlator.pmi_results,
            **kwargs
        )[0]

        reverse_alignment = alignment.reverse()
        reverse_align_key = reverse_alignment.key
        # Add new alignment to alignment log
        correlator.align_log[align_key] = alignment
        twin, family_index[PHONE_CORRELATORS_INDEX_KEY] = correlator.get_twin(family_index[PHONE_CORRELATORS_INDEX_KEY])
        twin.align_log[reverse_align_key] = reverse_alignment

    # Perform phonetic alignment without PMI support
    else:
        alignment = Alignment(word1, word2, **kwargs)

    return alignment


def handle_word_pair_input(input1, input2, family_index):  # TODO remove this function, used only for phonetic_dist
    """Check if a pair of word inputs or if already aligned word pair; align if not done already."""
    if isinstance(input1, Alignment):
        alignment = input1
        word1 = alignment.word1
        word2 = alignment.word2
    else:
        word1, word2 = input1, input2
        alignment = prepare_alignment(word1, word2, family_index)
    return word1, word2, alignment


def phonetic_dist(word1, family_index, word2=None, phone_sim_func=phone_sim, **kwargs):
    """Calculates phonetic distance of an alignment without weighting by
    segment type, position, etc.

    Args:
        word1 (phyloLing.Word or phonAlign.Alignment): first Word object, or an Alignment object
        word2 (phyloLing.Word): second Word object. Defaults to None.
        phone_sim_func (phone_sim, optional): Phonetic similarity function. Defaults to phone_sim.
    """

    # Calculate or retrieve the alignment
    _, _, alignment = handle_word_pair_input(word1, word2, family_index=family_index)

    # Calculate the phonetic similarity of each aligned segment
    # Gap alignments receive a score of 0
    phone_sims = [phone_sim_func(pair[0], pair[1], **kwargs)
                  if alignment.gap_ch not in pair else 0
                  for pair in alignment.alignment]

    # Return the mean distance
    return 1 - mean(phone_sims)


def prosodic_environment_strength(segments, i):
    """Returns the relative prosodic environment strength of a segment within a word, based on List (2012)"""

    # Map of prosodic environment strengths
    try:
        assert MAX_PROSODIC_ENV_STRENGTH == 7
    except AssertionError as exc:
        raise AssertionError(f"Unexpected value of constant MAX_PROSODIC_ENV_STRENGTH ({MAX_PROSODIC_ENV_STRENGTH})") from exc
    env_strengths = {
        "#C": 7,  # word-initial consonant
        "#V": 6,  # word-initial vowel
        "<S<": 5, # ascending sonority
        ">S>": 4, # descending sonority
        "<S>": 3,  # sonority peak
        "C#": 2,  # word-final consonant
        "V#": 1,  # word-final vowel
        "T": 0,
    }
    # TODO: all equal sonority

    seg = _toSegment(segments[i])
    
    # Minimum strength for tonemes/suprasegmentals
    if seg.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
        return env_strengths["T"]

    # Word-initial segments
    if i == 0:
        # Word-initial consonants: weight 7
        if seg.phone_class in ('CONSONANT', 'GLIDE'):
            return env_strengths["#C"]

        # Word-initial vowels: weight 6
        elif seg.phone_class in ('VOWEL', 'DIPHTHONG'):
            return env_strengths["#V"]

    # Word-final segments
    elif i == len(segments) - 1:

        # Word-final consonants: weight 2
        if seg.phone_class in ('CONSONANT', 'GLIDE'):
            return env_strengths["C#"]

        # Word-final vowels: weight 1
        elif seg.phone_class in ('VOWEL', 'DIPHTHONG'):
            return env_strengths["V#"]

    # Word-medial segments
    else:
        prev_segment, segment_i, next_segment = segments[i - 1], segments[i], segments[i + 1]
        prev_segment, segment_i, next_segment = map(_toSegment, [prev_segment, segment_i, next_segment])
        prev_sonority = prev_segment.sonority
        sonority_i = segment_i.sonority
        next_sonority = next_segment.sonority

        # Sonority peak: weight 3
        if prev_sonority <= sonority_i >= next_sonority:
            return env_strengths["<S>"]

        # Descending sonority: weight 4
        elif prev_sonority >= sonority_i >= next_sonority:
            return env_strengths[">S>"]

        # Ascending sonority: weight 5
        else:
            return env_strengths["<S<"]


def scale_deletion_penalty_by_prosodic_env_strength(penalty: float | int,
                                           gap: Gap,
                                           alignment: list[tuple],
                                           index: int
                                           ):
    """Scales a deletion penalty down according to the prosodic environment strength of the segment.

    Args:
        penalty (float | int): Penalty value.
        gap (Gap): Aligned Gap object.
        alignment (list): Aligned list of IPA string tuples.
        index (int): Alignment position index.

    Returns:
        (penalty, discount): Tuple of reduced penalty and discount factor.
    """
    deleted_index = gap.seg_i
    deleted_i = sum([1 for j in range(index + 1) if alignment[j][deleted_index] != gap.gap_ch]) - 1
    segment_list = [
        pos[deleted_index]
        for pos in alignment
        if pos[deleted_index] != gap.gap_ch
    ]
    prosodic_env_weight = prosodic_environment_strength(segment_list, deleted_i)
    discount = sqrt(abs(prosodic_env_weight - MAX_PROSODIC_ENV_STRENGTH) + 1)
    penalty /= discount
    return penalty, discount


def reduce_phon_deletion_penalty_by_phon_context(penalty: float,
                                                 gap: Gap,
                                                 alignment: list,
                                                 i: int,
                                                 penalty_discount: int | float = 2):
    """Reduces deletion penalty in certain phonological contexts.

    Args:
        penalty (float): Deletion penalty.
        gap (Gap): Gap (AlignedPair) object.
        alignment (list): List of aligned segments.
        i (int): Alignment index.
        penalty_discount (int | float, optional): Value by which deletion penalties are divided if reduction conditions are met. Defaults to 2.

    Returns:
        penalty: Reduced deletion penalty.
    """
    gap_ch = gap.gap_ch
    gap_index = gap.gap_i
    deleted_segment = _toSegment(gap.segment)
    previous_seg, next_seg = None, None
    previous_pos =  i > 0 and alignment[i - 1][gap_index] != gap_ch
    previous_seg = _toSegment(alignment[i - 1][gap_index]) if previous_pos else None
    next_pos = i < len(alignment) -1 and alignment[i + 1][gap_index] != gap_ch
    next_seg = _toSegment(alignment[i + 1][gap_index]) if next_pos else None

    # 1) If the deleted segment is a nasal and the corresponding
    # precending/following segment was (pre)nasal(ized)
    if deleted_segment.stripped in nasals:
        if previous_seg and previous_seg.features['nasal'] == 1:
            penalty /= penalty_discount
        elif next_seg and next_seg.features['nasal'] == 1:
            penalty /= penalty_discount

    # 2) If the deleted segment is a palatal glide (j, ɥ) or high front vowel (i, ɪ, y, ʏ),
    # and the corresponding preceding/following segment was palatalized
    # or is a palatal, alveolopalatal, or postalveolar consonant
    elif re.search(r'[jɥ]|([iɪyʏ]̯)', deleted_segment.segment):
        if previous_seg and previous_seg.base in palatal.union(alveolopalatal).union(postalveolar):
            penalty /= penalty_discount
        elif previous_seg and re.search(r'[ʲᶣ]', previous_seg.segment):
            penalty /= penalty_discount
        elif next_seg and next_seg.base in palatal.union(alveolopalatal).union(postalveolar): # TODO set this as a constant
            penalty /= penalty_discount
        elif next_seg and re.search(r'[ʲᶣ]', next_seg.segment):
            penalty /= penalty_discount

    # 3) If the deleted segment is a high rounded/labial glide
    # and the corresponding preceding/following segment was labialized
    elif re.search(r'[wʋ]|([uʊyʏ]̯)', deleted_segment.segment):
        if previous_seg and re.search(r'[ʷᶣ]', previous_seg.segment):
            penalty /= penalty_discount
        elif next_seg and re.search(r'[ʷᶣ]', next_seg.segment):
            penalty /= penalty_discount

    # 4) If the deleted segment is /h, ɦ/ and the corresponding
    # preceding/following segment was (pre-)aspirated or breathy
    elif re.search(r'[hɦ]', deleted_segment.base):
        if previous_seg and re.search(r'[ʰʱ̤]', previous_seg.segment):
            penalty /= penalty_discount
        elif next_seg and re.search(r'[ʰʱ̤]', next_seg.segment):
            penalty /= penalty_discount

    # 5) If the deleted segment is a rhotic approximant /ɹ, ɻ/
    # and the corresponding preceding/following segment was rhoticized
    elif re.search(r'[ɹɻ]', deleted_segment.base):
        if previous_seg and re.search(r'[ɚɝ˞]', previous_seg.segment):
            penalty /= penalty_discount
        if next_seg and re.search(r'[ɚɝ˞]', next_seg.segment):
            penalty /= penalty_discount

    # 6) If the deleted segment is a glottal stop and the corresponding
    # preceding segment was glottalized or creaky
    elif deleted_segment.base == 'ʔ':
        if previous_seg and re.search(r'[ˀ̰]', previous_seg.segment):
            penalty /= penalty_discount
        elif next_seg and re.search(r'[ˀ̰]', next_seg.segment):
            penalty /= penalty_discount

    # TODO add handling with glides and diphthongs, same principle as labialized/palatalized; same sound in 2 vs. in 1 segment transcription
    return penalty


def phonological_dist(word1: Word,
                      word2: Word,
                      family_index,
                      alignment=None,
                      sim_func=phone_sim,
                      penalize_sonority=True,
                      context_reduction=False,
                      env_strength_reduction=True,
                      total_dist=False,
                      **kwargs):
    f"""Calculates phonological distance between two words on the basis of the phonetic similarity of aligned segments and phonological deletion penalties.

    Args:
        word1 (Word): first Word object
        word2 (Word): second Word object
        alignment (Alignment): optional Alignment object
        family_index (dict): Family index dict for retrieving Doculect and PhonCorrelator objects.
        sim_func (_type_, optional): Phonetic similarity function. Defaults to {sim_func}.
        penalize_sonority (bool, optional): Penalizes deletions according to sonority of the deleted segment. Defaults to {penalize_sonority}.
        context_reduction (bool, optional): Reduces deletion penalties if certain phonological context conditions are met. Defaults to {context_reduction}.
        env_strength_reduction (bool, optional): Reduces deletion penalties according to prosodic environment strength (List, 2012). Defaults to {env_strength_reduction}.
        total_dist (bool, optional): Computes phonological distance as the sum of all penalties rather than as an average. Defaults to {total_dist}.

    Returns:
        float: phonological distance value
    """
    if alignment is None:
        alignment = prepare_alignment(word1, word2, family_index)
    gap_ch = alignment.gap_ch
    pad_ch = alignment.pad_ch
    alignment_obj = alignment
    alignment = alignment.alignment

    def _remove_boundaries(segment):
        """Remove boundaries from complex ngrams and convert simple boundary ngrams to gaps."""
        ngram = Ngram(segment)
        if ngram.is_boundary(pad_ch):
            no_boundary_ngram = ngram.remove_boundaries(pad_ch).undo()
            if len(no_boundary_ngram) == 0:
                return gap_ch
            return no_boundary_ngram
        return segment

    # Remove boundaries or convert to gaps
    alignment = [
        (_remove_boundaries(left), _remove_boundaries(right))
        for left, right in alignment
    ]
    # Remove any resulting (gap_ch, gap_ch) pairs
    alignment = [pos for pos in alignment if pos != (gap_ch, gap_ch)]

    # Simplify complex ngram alignments to unigrams
    alignment = alignment_obj.get_unigram_alignment(alignment)
    length = len(alignment)

    # Get list of penalties
    penalties = []
    for i, pair in enumerate(alignment):
        seg1, seg2 = pair

        # If the pair is a gap-aligned segment, assign the penalty
        # based on the sonority and information content (if available) of the deleted segment
        if gap_ch in pair:
            penalty = 1
            gap = Gap(alignment, i)
            deleted_segment = _toSegment(gap.segment)
            deleted_index = gap.seg_i

            if penalize_sonority:
                sonority = deleted_segment.sonority
                sonority_penalty = 1 - (sonority / (MAX_SONORITY + 1))
                penalty *= sonority_penalty

            # Adjust for geminates vs. long consonants (always performed)
            # If the deleted segment is part of a long/geminate segment transcribed as double (e.g. /tt/ rather than /tː/),
            # where at least one part of the geminate has been aligned
            # Method: check if the preceding or following pair contained the deleted segment at deleted_index, aligned to something other than the gap character
            # Check following pair
            double = False
            if i < length - 1:
                nxt_pair = alignment[i + 1]
                if gap_ch not in nxt_pair and nxt_pair[deleted_index] == deleted_segment.segment:
                    penalty = 0  # eliminate the current penalty altogether
                    alignment[i + 1] = list(alignment[i + 1])
                    # Adjust transcription of next segment to include gemination/length
                    # Penalty for next pair will take it into account
                    alignment[i + 1][deleted_index] = f'{deleted_segment.segment}ː'
                    double = True

            # Check preceding pair
            if i > 0 and not double:
                prev_pair = alignment[i - 1]
                if gap_ch not in prev_pair and prev_pair[deleted_index] == deleted_segment.segment:
                    penalty = 0  # eliminate the current penalty altogether
                    alignment[i - 1] = list(alignment[i - 1])
                    # Adjust previous penalty to include the length/gemination
                    alignment[i - 1][deleted_index] = f'{deleted_segment.segment}ː'
                    s1, s2 = alignment[i - 1]
                    penalties[-1] = 1 - sim_func(s1, s2, **kwargs)
            if penalty == 0:
                continue

            # Lessen the penalty if certain phonological conditions are met
            if context_reduction:
                penalty = reduce_phon_deletion_penalty_by_phon_context(
                    penalty, gap, alignment, i
                )

            # Discount deletion penalty according to strength of prosodic environment (based on List, 2012)
            if env_strength_reduction:
                penalty, _ = scale_deletion_penalty_by_prosodic_env_strength(
                    penalty, gap, alignment, i
                )

            # Add the final penalty to penalty list
            penalties.append(penalty)

        # Otherwise take the penalty as the phonetic distance between the aligned segments
        else:
            distance = 1 - sim_func(seg1, seg2, **kwargs)
            penalties.append(distance)

    if total_dist:
        word_dist = sum(penalties)
    else:
        # Euclidean distance of all penalties (= distance per dimension of the word)
        # normalized by square root of number of dimensions
        #word_dist = euclidean_dist(penalties) / sqrt(len(penalties))
        word_dist = mean(penalties)

    return word_dist


def mutual_surprisal(word1, word2, family_index, ngram_size=1, phon_env=True, normalize=False, pad_ch=PAD_CH_DEFAULT, **kwargs):
    lang1, lang2 = get_doculects_from_word_pair(word1, word2, family_index)

    # Check whether phoneme PMI has been calculated for this language pair
    # Otherwise calculate from scratch
    get_pmi_dict(lang1, lang2, family_index=family_index, **kwargs)

    # Calculate phoneme surprisal if not already done
    correlator1, correlator2 = get_phoneme_surprisal(
        lang1,
        lang2,
        ngram_size=ngram_size,
        family_index=family_index,
        **kwargs
    )

    # Generate alignments in each direction
    alignment = prepare_alignment(word1, word2, family_index, phon_env=phon_env)
    alignment.remove_padding()
    # Add phon env
    if phon_env:
        alignment.phon_env_alignment = alignment.add_phon_env()
    # Finally, reverse alignment
    rev_alignment = alignment.reverse()

    # Calculate the word-adaptation surprisal in each direction
    # (note: alignment needs to be reversed to run in second direction)
    if phon_env:
        sur_dict1 = correlator1.phon_env_surprisal_results
        sur_dict2 = correlator2.phon_env_surprisal_results
    else:
        sur_dict1 = correlator1.surprisal_results[ngram_size]
        sur_dict2 = correlator2.surprisal_results[ngram_size]

    WAS_l1l2 = adaptation_surprisal(
        alignment,
        surprisal_map=sur_dict1,
        ngram_size=ngram_size,
        phon_env=phon_env,
        normalize=False,
        pad_ch=lang1.alignment_params['pad_ch'],
        gap_ch=lang1.alignment_params['gap_ch'],
    )
    if ngram_size > 1:
        # TODO issue is possibly that the ngram size of 2 is not actually in the dict keys also including phon env, just has phon_env OR 2gram in separate dicts...
        # the way to get around this is:
        # calculate the 2gram, then get the 2gram's phon_env equivalent
        # interpolate the probability/surprisal of the 2gram with that of the phon_env equivalent
        raise NotImplementedError
    WAS_l2l1 = adaptation_surprisal(rev_alignment,
                                    surprisal_map=sur_dict2,
                                    ngram_size=ngram_size,
                                    phon_env=phon_env,
                                    normalize=False,
                                    pad_ch=lang2.alignment_params['pad_ch'],
                                    gap_ch=lang2.alignment_params['gap_ch'],
                                    )

    # Calculate self-surprisal values in each direction
    self_surprisal1 = lang1.self_surprisal(word1, normalize=False)
    self_surprisal2 = lang2.self_surprisal(word2, normalize=False)

    # Weight surprisal values by self-surprisal/information content value of corresponding segment
    # Segments with greater information content weighted more heavily
    # Normalize by phoneme entropy
    def weight_by_self_surprisal(alignment, WAS, self_surprisal, normalize_by, sur_dict, phon_env):
        self_info = sum([self_surprisal[j][-1] for j in self_surprisal])
        weighted_WAS = []
        seq_map1 = alignment.seq_map[0]
        align_iter = get_alignment_iter(alignment, phon_env=phon_env)
        for i, pair in enumerate(align_iter):

            # Skip pairs with aligned suprasegmental features with a gap
            # when the paired language (of the gap) does not have phonemic tones/suprasegmental features
            # Such gaps skew linguistic distances since tones/suprasegmental features occur on most or all words
            # and never have any equivalent
            # Also don't double-penalize deletion for shifted accent
            # if alignment.gap_ch in pair:
            #     gap_index = pair.index(alignment.gap_ch)
            #     seg = pair[gap_index - 1]
            #     if gap_index == 0:
            #         seg_lang, gap_lang = alignment.word2.language, alignment.word1.language
            #     else:
            #         seg_lang, gap_lang = alignment.word1.language, alignment.word2.language
            #     if seg in seg_lang.tonemes:
            #         if gap_lang.tonal is False:
            #             continue

            if seq_map1[i] is not None:
                weight = sum([self_surprisal[index][-1] for j, index in enumerate(seq_map1[i])]) / self_info
                normalized = WAS[i] / normalize_by
                weighted = weight * normalized
                weighted_WAS.append(weighted)

        return weighted_WAS

    weighted_WAS_l1l2 = weight_by_self_surprisal(
        alignment,
        WAS_l1l2,
        self_surprisal1,
        normalize_by=lang2.phoneme_entropy,
        sur_dict=sur_dict1,
        phon_env=phon_env
    )
    weighted_WAS_l2l1 = weight_by_self_surprisal(
        rev_alignment,
        WAS_l2l1,
        self_surprisal2,
        normalize_by=lang1.phoneme_entropy,
        sur_dict=sur_dict2,
        phon_env=phon_env
    )
    # Return and save the average of these two values
    if normalize:
        score = mean([mean(weighted_WAS_l1l2), mean(weighted_WAS_l2l1)])
    # TODO Treat surprisal values as distances and compute euclidean distance over these, then take average
    # score = mean([euclidean_dist(weighted_WAS_l1l2), euclidean_dist(weighted_WAS_l2l1)])
    else:
        score = mean([sum(weighted_WAS_l1l2), sum(weighted_WAS_l2l1)])

    return score


def pmi_dist(word1, word2, family_index, normalize=True, sim2dist=True, alpha=0.5, pad_ch=PAD_CH_DEFAULT, **kwargs):
    lang1, lang2 = get_doculects_from_word_pair(word1, word2, family_index)

    # Check whether phoneme PMI has been calculated for this language pair
    # Otherwise calculate from scratch
    pmi_dict = get_pmi_dict(
        lang1,
        lang2,
        family_index=family_index,
        **kwargs
    )

    # Fetch already computed alignments
    alignment = prepare_alignment(word1, word2, family_index, **kwargs)
    alignment.remove_padding()

    # Calculate PMI scores for each aligned pair
    PMI_values = [
        pmi_dict.get_value(Ngram(pair_left).undo(), Ngram(pair_right).undo())
        for pair_left, pair_right in alignment.alignment
    ]

    # Weight by information content per segment
    def weight_by_info_content(alignment, PMI_vals):
        info_content1 = word1.getInfoContent(doculect=lang1)
        info_content2 = word2.getInfoContent(doculect=lang2)
        total_info1 = word1.total_info_content
        total_info2 = word2.total_info_content
        seq_map1, seq_map2 = alignment.seq_map
        weighted_PMI = []
        for i, pair in enumerate(alignment.alignment):
            # Take the information content value of each segment within the respective word
            # Divide this by the total info content of the word to calculate the proportion of info content constituted by the segment
            if seq_map1[i] is not None:
                weight1 = sum([info_content1[index][-1] for j, index in enumerate(seq_map1[i])]) / total_info1

            else:
                weight1 = None

            if seq_map2[i] is not None:
                weight2 = sum([info_content2[index][-1] for j, index in enumerate(seq_map2[i])]) / total_info2
            else:
                weight2 = None

            # Average together the info contents of each aligned segment
            # Don't double-penalize deletion in case of shifted stress/accent: skip adding value if shifted later
            if weight1 is None:
                weight = weight2

            elif weight2 is None:
                weight = weight1

            else:
                weight = mean([weight1, weight2])

            # Weight by the averaged values
            weighted = weight * PMI_vals[i]
            weighted_PMI.append(weighted)

        return weighted_PMI

    PMI_values = weight_by_info_content(alignment, PMI_values)

    if normalize:
        PMI_score = mean(PMI_values)
    else:
        PMI_score = sum(PMI_values)

    if sim2dist:
        return sim_to_dist(PMI_score, alpha)

    else:
        return PMI_score


def levenshtein_dist(word1, word2, normalize=True, asjp=True):
    word1 = word1.ipa
    word2 = word2.ipa

    if asjp:
        word1 = strip_ch(ipa2asjp(preprocess_ipa_for_asjp_conversion(word1)), ["~"])
        word2 = strip_ch(ipa2asjp(preprocess_ipa_for_asjp_conversion(word2)), ["~"])

    LevDist = edit_distance(word1, word2)
    if normalize:
        LevDist /= max(len(word1), len(word2))

    return LevDist


def hybrid_dist(word1, word2, funcs: dict, weights=None, normalize_weights=False, family_index=None) -> float:
    """Calculates a hybrid distance of multiple distance or similarity functions

    Args:
        word1 (phyloLing.Word): first Word object
        word2 (phyloLing.Word): second Word object
        funcs (iterable): iterable of Distance class objects
        weights (list): list of weights (floats)
        normalize_weights (bool): if True, normalize weights such that they sum to 1.0
    Returns:
        float: hybrid similarity measure
    """
    scores = []
    if weights is None:
        # Uniform weighting
        weights = [1 / len(funcs) for i in range(len(funcs))]
    elif normalize_weights:
        weight_sum = sum(weights)
        weights = [weight/weight_sum for weight in weights]
        assert round(sum(weights)) == 1.0
    for func, weight in zip(funcs, weights):
        if weight == 0:
            scores.append(0)
            continue
        func_sim = func.sim
        score = func.eval(word1, word2)
        if func_sim:
            score = 1 - score

        # Distance weighting concept: if a distance is weighted with a higher coefficient relative to another distance,
        # it is as if that dimension is more impactful
        scores.append(score * weight)

        # Record word scores # TODO into Distance class object?
        log_word_score(word1, word2, score, key=func.name, family_index=family_index)

    # score = euclidean_dist(scores)
    # score = sum(scores)
    # TODO temp implementation: make more robust by checking that each function is as expected
    pmi_score, surprisal_score, phon_score = scores
    score = pmi_score + (surprisal_score * phon_score)
    log_word_score(word1, word2, score, key=HYBRID_DIST_KEY, family_index=family_index)
    log_word_score(word1, word2, dist_to_sim(score), key=HYBRID_SIM_KEY, family_index=family_index)

    return score


def log_alignment(word1, word2, lang1, lang2, family_index):
    if "alignment" not in lang1.lexical_comparison[lang2.name][(word1, word2)]:
        align_key = get_align_key(word1, word2)
        alignment, correlator, family_index[PHONE_CORRELATORS_INDEX_KEY] = retrieve_saved_alignment(
            align_key, lang1, lang2, family_index
        )
        if alignment is None:
            logger.debug(f"No saved alignment found for key {align_key}")
        else:
            alignment_str = visual_align(alignment)
            lang1.lexical_comparison[lang2.name][(word1, word2)]["alignment"] = alignment_str


def log_word_score(word1, word2, score, key, family_index):
    lang1, lang2 = get_doculects_from_word_pair(word1, word2, family_index)
    lang1.lexical_comparison_measures.add(key)
    lang2.lexical_comparison_measures.add(key)
    lang1.lexical_comparison[lang2.name][(word1, word2)][key] = score
    lang2.lexical_comparison[lang1.name][(word2, word1)][key] = score
    log_alignment(word1, word2, lang1, lang2, family_index)
    log_alignment(word2, word1, lang2, lang1, family_index)


# Initialize distance functions as Distance objects
# NB: Hybrid distance(s) need to be defined in classifyLangs.py or else we can't set the parameters of the component functions based on config settings
LEVENSHTEIN_DIST_KEY = 'LevenshteinDist'
PHONETIC_DIST_KEY = 'PhoneticDist'
PHONOLOGICAL_DIST_KEY = 'PhonDist'
PMI_DIST_KEY = 'PMIDist'
SURPRISAL_DIST_KEY = 'SurprisalDist'
HYBRID_DIST_KEY = 'HybridDist'
HYBRID_SIM_KEY = 'HybridSimilarity'
LevenshteinDist = WordDistance(func=levenshtein_dist, name=LEVENSHTEIN_DIST_KEY)
# PhoneticDist = WordDistance(func=phonetic_dist, name=PHONETIC_DIST_KEY)
PhonDist = WordDistance(func=phonological_dist, name=PHONOLOGICAL_DIST_KEY)
PMIDist = WordDistance(func=pmi_dist, name=PMI_DIST_KEY)
SurprisalDist = WordDistance(func=mutual_surprisal, name=SURPRISAL_DIST_KEY, ngram_size=1)
