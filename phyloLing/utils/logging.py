from constants import ALIGNMENT_DELIMITER
from phonAlign import visual_align


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
