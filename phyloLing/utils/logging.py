import os

from constants import ALIGNMENT_DELIMITER
from phonAlign import visual_align
from utils.information import prune_oov_surprisal, surprisal_to_prob
from utils.sequence import Ngram
from utils.utils import dict_tuplelist
from utils.wordlist import sort_wordlist


def make_outdir(outfile):
    outdir = os.path.abspath(os.path.dirname(outfile))
    os.makedirs(outdir, exist_ok=True)


def ngram2log_format(ngram, phon_env=False):
    if phon_env:
        ngram, phon_env = ngram[:-1], ngram[-1]
        return (Ngram(ngram).string, phon_env)
    else:
        return Ngram(ngram).string


def write_sample_log(sample_logs, log_file):
    make_outdir(log_file)
    content = []
    for _, (same_meaning_log, diff_meaning_log) in sample_logs.items():
        sample_n_log = "\n\n".join([same_meaning_log, diff_meaning_log])
        content.append(sample_n_log)
    content = "\n\n".join(content)
    with open(log_file, 'w') as f:
        f.write(content)


def write_alignments_log(alignment_log, log_file):
    """Write an alignment log."""
    sorted_alignment_keys = sorted(alignment_log.keys())
    n_alignments = len(sorted_alignment_keys)
    make_outdir(log_file)
    with open(log_file, 'w') as f:
        for i, key in enumerate(sorted_alignment_keys):
            f.write(f'{key}\n')
            alignment = alignment_log[key]
            align_str = visual_align(alignment.alignment, gap_ch=alignment.gap_ch)
            align_cost = round(alignment.cost, 3)
            seq_map1, seq_map2 = alignment.seq_map
            f.write(f'{align_str}\n{seq_map1}\n{seq_map2}\nSCORE: {align_cost}\n')
            if i < n_alignments - 1:
                f.write(f'\n{ALIGNMENT_DELIMITER}\n\n')


def log_phon_corr_iteration(iteration,
                            qualifying_words,
                            disqualified_words,
                            method=None,
                            same_meaning_alignments=None
                            ):
    iter_log = []
    qualifying = qualifying_words[iteration].word_pairs
    prev_qualifying = qualifying_words[iteration - 1].word_pairs
    disqualified = disqualified_words[iteration]
    prev_disqualified = disqualified_words[iteration - 1]
    iter_log.append(f'Iteration {iteration}')
    iter_log.append(f'\tQualified: {len(qualifying)}')
    iter_log.append(f'\tDisqualified: {len(disqualified)}')
    added = set(qualifying) - set(prev_qualifying)
    iter_log.append(f'\tAdded: {len(added)}')
    for word1, word2 in sort_wordlist(added):
        iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')
    removed = set(disqualified) - set(prev_disqualified)
    iter_log.append(f'\tRemoved: {len(removed)}')
    for word1, word2 in sort_wordlist(removed):
        iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')

    iter_log = '\n'.join(iter_log)

    return iter_log


def write_phon_corr_iteration_log(iter_logs, log_file, n_same_meaning_pairs):
    make_outdir(log_file)
    with open(log_file, 'w') as f:
        f.write(f'Same meaning pairs: {n_same_meaning_pairs}\n')
        for n in iter_logs:
            iter_log = '\n\n'.join(iter_logs[n][:-1])
            f.write(f'****SAMPLE {n+1}****\n')
            f.write(iter_log)
            final_qualifying, final_disqualified = iter_logs[n][-1]
            f.write('\n\nFinal qualifying:\n')
            for word1, word2 in sort_wordlist(final_qualifying.word_pairs):
                f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
            f.write('\nFinal disqualified:\n')
            for word1, word2 in sort_wordlist(final_disqualified):
                f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
            f.write('\n\n-------------------\n\n')


def write_phoneme_pmi_report(pmi_results, outfile, threshold=0.0001, sep='\t'):
    make_outdir(outfile)
    # Save all segment pairs with non-zero PMI values to file
    # Skip extremely small decimals that are close to zero
    lines = []
    for seg1 in pmi_results.get_primary_keys():
        for seg2 in pmi_results.get_secondary_keys(seg1):
            pmi_val = round(pmi_results.get_value(seg1, seg2), 3)
            if abs(pmi_val) > threshold:
                line = [ngram2log_format(seg1), ngram2log_format(seg2), str(pmi_val)]
                lines.append(line)
    # Sort PMI in descending order, then by phone pair
    lines = sorted(lines, key=lambda line: (float(line[-1]), line[0], line[1]), reverse=True)
    lines = '\n'.join([sep.join(line) for line in lines])

    with open(outfile, 'w') as f:
        header = sep.join(['Phone1', 'Phone2', 'PMI'])
        f.write(f'{header}\n{lines}')


def write_phoneme_surprisal_report(surprisal_results, outfile, phon_env=True, ngram_size=1, sep='\t'):
    make_outdir(outfile)
    lines = []
    surprisal_results, oov_value = prune_oov_surprisal(surprisal_results)
    oov_value = round(oov_value, 3)
    for seg1 in surprisal_results:
        for seg2 in surprisal_results[seg1]:
            if ngram_size > 1:
                raise NotImplementedError  # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
            if phon_env:
                seg1_str, phon_env = ngram2log_format(seg1, phon_env=True)
            else:
                seg1_str = ngram2log_format(seg1, phon_env=False)
            lines.append(
                [
                    seg1_str,
                    ngram2log_format(seg2, phon_env=False),  # phon_env only on seg1
                    str(abs(round(surprisal_results[seg1][seg2], 3))),
                    str(oov_value)
                ]
            )
            if phon_env:
                lines[-1].insert(1, phon_env)

    # Sort by phone1 (by phon env if relevant) and then by surprisal in ascending order
    if phon_env:
        lines = sorted(lines, key=lambda x: (x[0], x[1], float(x[3]), x[2]), reverse=False)
    else:
        lines = sorted(lines, key=lambda x: (x[0], float(x[2]), x[1]), reverse=False)
    lines = '\n'.join([sep.join(line) for line in lines])
    with open(outfile, 'w') as f:
        header = ['Phone1', 'Phone2', 'Surprisal', 'OOV_Smoothed']
        if phon_env:
            header.insert(1, "PhonEnv")
        header = sep.join(header)
        f.write(f'{header}\n{lines}')


def write_phon_corr_report(corr, lang1_name, lang2_name, gap_ch, outfile, corr_type, min_prob=0.05):
    make_outdir(outfile)
    lines = []
    corr, _ = prune_oov_surprisal(corr)
    l1_phons = sorted([p for p in corr if gap_ch not in p], key=lambda x: Ngram(x).string)
    for p1 in l1_phons:
        p2_candidates = corr[p1]
        if len(p2_candidates) > 0:
            p2_candidates = dict_tuplelist(p2_candidates, reverse=True)
            for p2, score in p2_candidates:
                if corr_type == 'surprisal':
                    prob = surprisal_to_prob(score)  # turn surprisal value into probability
                    if prob >= min_prob:
                        p1 = Ngram(p1).string
                        p2 = Ngram(p2).string
                        line = [p1, p2, str(round(prob, 3))]
                        lines.append(line)
                else:
                    raise NotImplementedError  # not implemented for PMI
    # Sort by corr value, then by phone string if values are equal
    lines.sort(key=lambda x: (float(x[-1]), x[0], x[1]), reverse=True)
    lines = ['\t'.join(line) for line in lines]
    header = '\t'.join([lang1_name, lang2_name, 'probability'])
    lines = '\n'.join(lines)
    content = '\n'.join([header, lines])
    with open(outfile, 'w') as f:
        f.write(f'{content}')
