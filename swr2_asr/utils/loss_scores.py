"""Methods for determining the loss and scores of the model."""
import numpy as np


def avg_wer(wer_scores, combined_ref_len) -> float:
    """Calculate the average word error rate (WER).

    Args:
        wer_scores: word error rate scores
        combined_ref_len: combined length of reference sentences

    Returns:
        average word error rate (float)

    Usage:
        >>> avg_wer([0.5, 0.5], 2)
        0.5
    """
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp) -> int:
    """Levenshtein distance.

    Args:
        ref: reference sentence
        hyp: hypothesis sentence

    Returns:
        distance: levenshtein distance between reference and hypothesis

    Usage:
        >>> _levenshtein_distance("hello", "helo")
        2
    """
    len_ref = len(ref)
    len_hyp = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if len_ref == 0:
        return len_hyp
    if len_hyp == 0:
        return len_ref

    if len_ref < len_hyp:
        ref, hyp = hyp, ref
        len_ref, len_hyp = len_hyp, len_ref

    # use O(min(m, n)) space
    distance = np.zeros((2, len_hyp + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, len_hyp + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, len_ref + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, len_hyp + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[len_ref % 2][len_hyp]


def word_errors(
    reference: str, hypothesis: str, ignore_case: bool = False, delimiter: str = " "
) -> tuple[float, int]:
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.

    Args:
        reference: The reference sentence.
        hypothesis: The hypothesis sentence.
        ignore_case: Whether case-sensitive or not.
        delimiter: Delimiter of input sentences.

    Returns:
        Levenshtein distance and length of reference sentence.

    Usage:
        >>> word_errors("hello world", "hello")
        1, 2
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    remove_space: bool = False,
) -> tuple[float, int]:
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    Args:
        reference: The reference sentence.
        hypothesis: The hypothesis sentence.
        ignore_case: Whether case-sensitive or not.
        remove_space: Whether remove internal space characters

    Returns:
        Levenshtein distance and length of reference sentence.

    Usage:
        >>> char_errors("hello world", "hello")
        1, 10
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = " "
    if remove_space:
        join_char = ""

    reference = join_char.join(filter(None, reference.split(" ")))
    hypothesis = join_char.join(filter(None, hypothesis.split(" ")))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference: str, hypothesis: str, ignore_case=False, delimiter=" ") -> float:
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level.
    WER is defined as:
        WER = (Sw + Dw + Iw) / Nw
    with:
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference

    Args:
        reference: The reference sentence.
        hypothesis: The hypothesis sentence.
        ignore_case: Whether case-sensitive or not.
        delimiter: Delimiter of input sentences.

    Returns:
        Word error rate (float)

    Usage:
        >>> wer("hello world", "hello")
        0.5
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case, delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    word_error_rate = float(edit_distance) / ref_len
    return word_error_rate


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
        CER = (Sc + Dc + Ic) / Nc
    with
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference

    Args:
        reference: The reference sentence.
        hypothesis: The hypothesis sentence.
        ignore_case: Whether case-sensitive or not.
        remove_space: Whether remove internal space characters

    Returns:
        Character error rate (float)

    Usage:
        >>> cer("hello world", "hello")
        0.2727272727272727
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case, remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    char_error_rate = float(edit_distance) / ref_len
    return char_error_rate
