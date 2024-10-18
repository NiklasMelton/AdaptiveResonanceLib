import numpy as np
from typing import Optional, Callable, Tuple, Union, Dict
from artlib import BaseART
import operator
import re


def compress_dashes(input_string: str) -> str:
    """
    Compress consecutive dashes in a string into a single dash.

    Parameters
    ----------
    input_string : str
        The input string containing dashes.

    Returns
    -------
    str
        The string with consecutive dashes compressed into one dash.
    """
    return re.sub("-+", "-", input_string)


def arr2seq(x: np.ndarray) -> str:
    """
    Convert an array of integers to a string.

    Parameters
    ----------
    x : np.ndarray
        Array of integers to be converted.

    Returns
    -------
    str
        The string representation of the array.
    """
    return "".join([str(i_) for i_ in x])


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match_score: int = 1,
    gap_cost: int = -1,
    mismatch_cost: int = -1,
) -> Tuple[str, float]:
    """
    Perform Needleman-Wunsch sequence alignment between two sequences.

    Parameters
    ----------
    seq1 : str
        The first sequence to align.
    seq2 : str
        The second sequence to align.
    match_score : int, optional
        The score for a match (default is 1).
    gap_cost : int, optional
        The penalty for a gap (default is -1).
    mismatch_cost : int, optional
        The penalty for a mismatch (default is -1).

    Returns
    -------
    tuple
        The aligned sequences and the normalized alignment score.
    """
    m, n = len(seq1), len(seq2)

    # Initialize the scoring matrix
    score_matrix = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the gap penalties for the first row and column
    for i in range(1, m + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap_cost
    for j in range(1, n + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap_cost

    # Fill in the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_cost
            )
            delete = score_matrix[i - 1][j] + gap_cost
            insert = score_matrix[i][j - 1] + gap_cost
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to get the alignment
    align1, align2 = "", ""
    i, j = m, n

    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        diagonal_score = score_matrix[i - 1][j - 1]
        up_score = score_matrix[i - 1][j]
        left_score = score_matrix[i][j - 1]

        if current_score == diagonal_score + (
            match_score if seq1[i - 1] == seq2[j - 1] else mismatch_cost
        ):
            align1 += seq1[i - 1]
            align2 += seq2[j - 1]
            i -= 1
            j -= 1
        elif current_score == up_score + gap_cost:
            align1 += seq1[i - 1]
            align2 += "-"
            i -= 1
        elif current_score == left_score + gap_cost:
            align1 += "-"
            align2 += seq2[j - 1]
            j -= 1

    # Fill the remaining sequence if the above while loop finishes
    while i > 0:
        align1 += seq1[i - 1]
        align2 += "-"
        i -= 1

    while j > 0:
        align1 += "-"
        align2 += seq2[j - 1]
        j -= 1

    # The alignments are built from the end to the beginning, so we need to reverse them
    align1 = align1[::-1]
    align2 = align2[::-1]
    alignment = "".join([a if a == b else "-" for a, b in zip(align1, align2)])
    l = max(len(seq1), len(seq2))

    return alignment, float(score_matrix[m][n]) / l


def prepare_data(data: np.ndarray) -> np.ndarray:
    """
    Prepares the data for clustering.

    Parameters
    ----------
    data : np.ndarray
        The input data.

    Returns
    -------
    np.ndarray
        The prepared data.
    """
    return data


class SeqART(BaseART):
    """
    Sequence ART for clustering based on sequence alignment.
    """

    def __init__(self, rho: float, metric: Callable = needleman_wunsch):
        """
        Initialize the SeqART instance.

        Parameters
        ----------
        rho : float
            The vigilance parameter.
        metric : Callable, optional
            The alignment function. Should be in the format: alignment, score = metric(seq_a, seq_b).
        """
        params = {
            "rho": rho,
        }
        self.metric = metric
        super().__init__(params)

    @staticmethod
    def validate_params(params: dict):
        """
        Validate clustering parameters.

        Parameters
        ----------
        params : dict
            The parameters for the algorithm.
        """
        assert "rho" in params
        assert isinstance(params["rho"], float)

    def validate_data(self, X: np.ndarray):
        """
        Validate the input data for clustering.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        """
        pass

    def check_dimensions(self, X: np.ndarray):
        """
        Check that the input data has the correct dimensions.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        """
        pass

    def category_choice(
        self, i: str, w: str, params: dict
    ) -> tuple[float, Optional[dict]]:
        """
        Get the activation of the cluster.

        Parameters
        ----------
        i : str
            The data sample.
        w : str
            The cluster weight/info.
        params : dict
            The algorithm parameters.

        Returns
        -------
        tuple
            Cluster activation and cache used for later processing.
        """
        alignment, score = self.metric(arr2seq(i), w)
        cache = {"alignment": alignment, "score": score}
        return score, cache

    def match_criterion(
        self, i: str, w: str, params: dict, cache: Optional[dict] = None
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get the match criterion of the cluster.

        Parameters
        ----------
        i : str
            The data sample.
        w : str
            The cluster weight/info.
        params : dict
            The algorithm parameters.
        cache : dict, optional
            Cached values from previous calculations.

        Returns
        -------
        tuple
            Cluster match criterion and cache used for later processing.
        """
        # _, M = self.metric(cache['alignment'], w)

        return cache["score"], cache

    def match_criterion_bin(
        self,
        i: np.ndarray,
        w: np.ndarray,
        params: dict,
        cache: Optional[dict] = None,
        op: Callable = operator.ge,
    ) -> tuple[bool, dict]:
        """
        Get the binary match criterion of the cluster.

        Parameters
        ----------
        i : np.ndarray
            The data sample.
        w : np.ndarray
            The cluster weight/info.
        params : dict
            The algorithm parameters.
        cache : dict, optional
            Cached values from previous calculations.
        op : Callable, optional
            Comparison operator for the match criterion (default is operator.ge).

        Returns
        -------
        tuple
            Binary match criterion and cache used for later processing.
        """
        M, cache = self.match_criterion(arr2seq(i), w, params, cache)
        M_bin = op(M, params["rho"])
        if cache is None:
            cache = dict()
        cache["match_criterion"] = M
        cache["match_criterion_bin"] = M_bin

        return M_bin, cache

    def update(
        self, i: str, w: str, params: dict, cache: Optional[dict] = None
    ) -> str:
        """
        Update the cluster weight.

        Parameters
        ----------
        i : str
            The data sample.
        w : str
            The cluster weight/info.
        params : dict
            The algorithm parameters.
        cache : dict, optional
            Cached values from previous calculations.

        Returns
        -------
        str
            Updated cluster weight.
        """
        # print(cache['alignment'])
        return compress_dashes(cache["alignment"])

    def new_weight(self, i: str, params: dict) -> str:
        """
        Generate a new cluster weight.

        Parameters
        ----------
        i : str
            The data sample.
        params : dict
            The algorithm parameters.

        Returns
        -------
        str
            New cluster weight.
        """
        return arr2seq(i)
