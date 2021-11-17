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


def edit_distance_banded(s1: str, s2: str) -> int:
    m = len(s1)
    n = len(s2)
    # make sure the shorter string is going vertically
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    bottom_shift = m - n
    d = min(abs(bottom_shift) + 10, m)
    width = 2 * d + 1
    height = n + 1

    # make the blank scoring table
    scores = np.empty((height, width), dtype=int)

    # set values in the first row
    scores[0, d] = 0
    for shift_from_center in range(1, d + 1):
        col = d + shift_from_center
        # copy from the left
        scores[0, col] = scores[0, col - 1] + COST_INDEL

    # calculate remaining scores
    for row in range(1, height):
        band_center_col = row
        for shift_from_center in range(-d, d + 1):
            col_theoretical = band_center_col + shift_from_center
            if col_theoretical < 0 or col_theoretical >= m + 1:
                # out of bounds
                continue
            col = d + shift_from_center
            if col_theoretical == 0:  # first column
                # copy from above
                scores[row, col] = scores[row - 1, col + 1] + COST_INDEL
            else:
                # get the three possible scores
                # get from_left
                if col == 0:
                    # left is not an option
                    from_left = float("inf")  # force it to lose when put into 'min'
                else:
                    from_left = scores[row, col - 1] + COST_INDEL
                # get from_above
                if col == width - 1:
                    # above is not an option
                    from_above = float("inf")  # force it to lose when put into 'min'
                else:
                    from_above = scores[row - 1, col + 1] + COST_INDEL
                # get from_diagonal
                if s1[col_theoretical - 1] == s2[row - 1]:
                    from_diagonal = scores[row - 1, col]  # +0 for match
                else:
                    from_diagonal = scores[row - 1, col] + COST_SUB
                # write down our three options
                score_options = [from_left, from_above, from_diagonal]

                # what if we swapped (transposed) adjacent characters?
                if (row >= 2 and col_theoretical >= 2) and (s1[col_theoretical - 1] == s2[row - 2] and s1[col_theoretical - 2] == s2[row - 1]):
                    swap = scores[row - 2, col] + COST_SWAP
                    score_options.append(swap)
                # pick from the three or four options
                scores[row, col] = min(score_options)
    return scores[-1, d + bottom_shift]


def normalized_edit_distance(incorrect: str, correct: str, banded: bool = True) -> float:
    if banded:
        plain_edit_distance = edit_distance_banded(incorrect, correct)
    else:
        plain_edit_distance = edit_distance(incorrect, correct)
    return plain_edit_distance / len(correct)


if __name__ == "__main__":
    score_ = edit_distance_banded("xxabcdefghijklnmop", "abcdefghijklmnop")
    print(score_)
