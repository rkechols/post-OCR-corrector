import numpy as np


COST_INDEL = 1
COST_SUB = 1
COST_SWAP = 1


def edit_distance(s1: str, s2: str) -> int:
    """
    Compute the edit distance between two strings using Damerau-Levenshtein distance
        (see https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance).

    Uses the value of `COST_INDEL` for the cost to insert or delete.
    Uses the value of `COST_SUB` for the cost to substitute a character.
    Uses the value of `COST_SWAP` for the cost to swap/transpose adjacent characters.
    Cost of a character match is 0.

    Parameters
    ----------
    s1 : str
        the first string for comparison
    s2 : str
        the second string for comparison

    Returns
    -------
    int
        the edit distance between the two strings
    """
    m = len(s1) + 1
    n = len(s2) + 1
    scores = np.empty((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:  # origin
                scores[i, j] = 0
                continue
            elif i == 0:  # first row, only option is left
                scores[i, j] = scores[i, j - 1] + COST_INDEL
                continue
            elif j == 0:  # far left column, only option is up
                scores[i, j] = scores[i - 1, j] + COST_INDEL
                continue
            # normal spot, compare our options for hypothetical scores
            left = scores[i, j - 1] + COST_INDEL
            up = scores[i - 1, j] + COST_INDEL
            diagonal = scores[i - 1, j - 1]
            if s1[i - 1] != s2[j - 1]:  # not a match means it's a substitution
                diagonal += COST_SUB
            score_options = [left, up, diagonal]
            # what if we swapped (transposed) adjacent characters?
            if (i >= 2 and j >= 2) and (s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]):
                swap = scores[i - 2, j - 2] + COST_SWAP
                score_options.append(swap)
            scores[i, j] = min(score_options)

    return scores[-1, -1]


def generate_edits(to_edit: str, good_chars: str):  # -> string generator
    n = len(to_edit)
    for i in range(n):  # try deleting each char
        yield to_edit[:i] + to_edit[(i + 1):]
    for i in range(n - 1):  # try swapping each adjacent pair
        yield to_edit[:i] + to_edit[i + 1] + to_edit[i] + to_edit[(i + 2):]
    for i in range(n):  # try changing each char to a different one
        this_char = to_edit[i]
        for char in good_chars:
            if char == this_char:
                continue  # don't swap a char for itself
            yield to_edit[:i] + char + to_edit[(i + 1):]
    for i in range(n + 1):  # try inserting a char
        for char in good_chars:
            yield to_edit[:i] + char + to_edit[i:]
