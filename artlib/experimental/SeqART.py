import numpy as np
from typing import Optional, Callable
from artlib import BaseART
import operator
import re

def compress_dashes(input_string):
    return re.sub('-+', '-', input_string)

def arr2seq(x):
    return "".join([str(i_) for i_ in x])

def needleman_wunsch(seq1, seq2, match_score=1, gap_cost=-1, mismatch_cost=-1):
    m, n = len(seq1), len(seq2)

    # Initialize the scoring matrix
    score_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the gap penalties for the first row and column
    for i in range(1, m + 1):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_cost
    for j in range(1, n + 1):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_cost

    # Fill in the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_cost)
            delete = score_matrix[i-1][j] + gap_cost
            insert = score_matrix[i][j-1] + gap_cost
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to get the alignment
    align1, align2 = '', ''
    i, j = m, n

    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        diagonal_score = score_matrix[i-1][j-1]
        up_score = score_matrix[i-1][j]
        left_score = score_matrix[i][j-1]

        if current_score == diagonal_score + (match_score if seq1[i-1] == seq2[j-1] else mismatch_cost):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif current_score == up_score + gap_cost:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif current_score == left_score + gap_cost:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1

    # Fill the remaining sequence if the above while loop finishes
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1

    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1

    # The alignments are built from the end to the beginning, so we need to reverse them
    align1 = align1[::-1]
    align2 = align2[::-1]
    alignment = ''.join([a if a == b else '-' for a, b in zip(align1, align2)])
    l = max(len(seq1), len(seq2))
    return alignment, float(score_matrix[m][n])/l


def prepare_data(data: np.ndarray) -> np.ndarray:
    return data


class SeqART(BaseART):
    # template for ART module
    def __init__(self, rho: float, metric: Callable = needleman_wunsch):
        """
        Parameters:
        - rho: vigilance parameter
        - metric: allignment function. Should be in the format alignment, score = metric(seq_a, seq_b)

        """
        params = {
            "rho": rho,
        }
        self.metric = metric
        super().__init__(params)
    @staticmethod
    def validate_params(params: dict):
        """
        validate clustering parameters

        Parameters:
        - params: dict containing parameters for the algorithm

        """
        assert "rho" in params
        assert isinstance(params["rho"], float)

    def validate_data(self, X: np.ndarray):
        """
        validates the data prior to clustering

        Parameters:
        - X: data set

        """
        pass

    def check_dimensions(self, X: np.ndarray):
        """
        check the data has the correct dimensions

        Parameters:
        - X: data set

        """
        pass


    def category_choice(self, i: str, w: str, params: dict) -> tuple[float, Optional[dict]]:
        """
        get the activation of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            cluster activation, cache used for later processing

        """
        alignment, score = self.metric(arr2seq(i), w)
        cache = {'alignment': alignment, 'score': score}
        return score, cache

    def match_criterion(self, i: str, w: str, params: dict, cache: Optional[dict] = None) -> tuple[float, dict]:
        """
        get the match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion, cache used for later processing

        """
        # _, M = self.metric(cache['alignment'], w)

        return cache['score'], cache

    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None, op: Callable = operator.ge) -> tuple[bool, dict]:
        """
        get the binary match criterion of the cluster

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            cluster match criterion binary, cache used for later processing

        """
        M, cache = self.match_criterion(arr2seq(i), w, params, cache)
        M_bin = op(M, params["rho"])
        if cache is None:
            cache = dict()
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin

        return M_bin, cache

    def update(self, i: str, w: str, params: dict, cache: Optional[dict] = None) -> str:
        """
        get the updated cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm
        - cache: dict containing values cached from previous calculations

        Returns:
            updated cluster weight, cache used for later processing

        """
        # print(cache['alignment'])
        return compress_dashes(cache['alignment'])

    def new_weight(self, i: str, params: dict) -> str:
        """
        generate a new cluster weight

        Parameters:
        - i: data sample
        - w: cluster weight / info
        - params: dict containing parameters for the algorithm

        Returns:
            updated cluster weight

        """
        return arr2seq(i)