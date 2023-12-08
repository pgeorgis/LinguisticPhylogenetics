from math import log
from statistics import mean
from constants import PAD_CH_DEFAULT
from utils.sequence import Ngram

def pointwise_mutual_info(p_joint, p_x, p_y):
    return log(p_joint/(p_x*p_y)) # TODO should it be log base 2?

def bayes_pmi(pX_given_Y, pX):  # TODO should it be log base 2?
    # via Bayes Theorem : pmi = log( p(x,y) / ( p(x) * p(y) ) ) = log( p(x|y) / p(x) )
    return log(pX_given_Y/pX)

def surprisal(p):
    try:
        return -log(p, 2)
    except ValueError:
        raise ValueError(f'Math Domain Error: cannot take the log of {p}')

def surprisal_to_prob(s):
    return 2**-s

def adaptation_surprisal(alignment, surprisal_dict, ngram_size=1, phon_env=False, normalize=True, pad_ch=PAD_CH_DEFAULT):
    """Calculates the surprisal of an aligned sequence, given a dictionary of 
    surprisal values for the sequence corresponcences"""

    # if type(alignment) is Alignment:
    #     length = alignment.length
    #     alignment = alignment.alignment
    # elif type(alignment) is list:
    #     length = len(alignment)
    # else:
    #     raise TypeError
    # TODO problem: this function needs to live in this script (auxFuncs.py) to avoid circular imports but importing Alignment object from phonAlign.py would also cause a circular import
    # Temporary solution: assume that if the alignment is not a list, it is an Alignment class object
    if isinstance(alignment, list):
        length = len(alignment)
    else:
        length = alignment.length
        if phon_env:
            if alignment.phon_env_alignment is None:
                alignment.phon_env_alignment = alignment.add_phon_env()
            alignment = alignment.phon_env_alignment
            
        else:
            alignment = alignment.alignment
    
    pad_n = ngram_size - 1
    if ngram_size > 1:
        alignment = [(pad_ch, pad_ch)]*pad_n + alignment + [(pad_ch, pad_ch)]*pad_n
    
    values = []
    for i in range(pad_n, length+pad_n):
        ngram = alignment[i:i+ngram_size]
        segs = list(zip(*ngram))
        seg1, seg2 = segs
        # Convert to form stored in correspondence dictionaries, whereby unigrams are strings and larger ngrams are tuples
        seg1 = Ngram(seg1).undo()
        seg2 = Ngram(seg2).undo()
        values.append(surprisal_dict[seg1][seg2])

    if normalize:
        return mean(values)
    else:
        return values

def entropy(X):
    """X should be a dictionary with absolute counts"""
    total = sum(X.values())
    E = 0
    for i in X:
        p = X[i]/total
        if p > 0:
            E += p * surprisal(p)
    return E