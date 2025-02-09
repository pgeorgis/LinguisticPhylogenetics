import os

from constants import ALIGNMENT_DELIMITER
from phonAlign import visual_align
from utils.wordlist import sort_wordlist


def write_alignments_log(alignment_log, log_file):
    """Write an alignment log."""
    sorted_alignment_keys = sorted(alignment_log.keys())
    n_alignments = len(sorted_alignment_keys)
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
    log_dir = os.path.abspath(os.path.dirname(log_file))
    os.makedirs(log_dir, exist_ok=True)
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
