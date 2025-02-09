import os

from constants import ALIGNMENT_DELIMITER
from phonAlign import visual_align
from utils.information import prune_oov_surprisal
from utils.sequence import Ngram
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
            f.write(f'{align_str}\n{seq_map1}\n{seq_map2}\nCOST: {align_cost}\n')
            if i < n_alignments - 1:
                f.write(f'\n{ALIGNMENT_DELIMITER}\n\n')


def write_phon_corr_iteration_log(iter_logs, log_file, n_same_meaning_pairs):
    make_outdir(log_file)
    with open(log_file, 'w') as f:
        f.write(f'Same meaning pairs: {len(n_same_meaning_pairs)}\n')
        for n in iter_logs:
            iter_log = '\n\n'.join(iter_logs[n][:-1])
            f.write(f'****SAMPLE {n+1}****\n')
            f.write(iter_log)
            final_qualifying, final_disqualified = iter_logs[n][-1]
            f.write('\n\nFinal qualifying:\n')
            for word1, word2 in sort_wordlist(final_qualifying):
                f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
            f.write('\nFinal disqualified:\n')
            for word1, word2 in sort_wordlist(final_disqualified):
                f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
            f.write('\n\n-------------------\n\n')


def log_phoneme_pmi(pmi_results, outfile, threshold=0.0001, sep='\t'):
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


def log_phoneme_surprisal(self, outfile, sep='\t', phon_env=True, ngram_size=1):
    make_outdir(outfile)
    if phon_env:
        surprisal_dict = self.phon_env_surprisal_results
    else:
        surprisal_dict = self.surprisal_results[ngram_size]

    lines = []
    surprisal_dict, oov_value = prune_oov_surprisal(surprisal_dict)
    oov_value = round(oov_value, 3)
    for seg1 in surprisal_dict:
        for seg2 in surprisal_dict[seg1]:
            if ngram_size > 1:
                raise NotImplementedError  # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
            if phon_env:
                seg1_str, phon_env = ngram2log_format(seg1, phon_env=True)
            else:
                seg1_str = ngram2log_format(seg1, phon_env=False)
            lines.append([
                seg1_str,
                ngram2log_format(seg2, phon_env=False),  # phon_env only on seg1
                str(abs(round(surprisal_dict[seg1][seg2], 3))),
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
