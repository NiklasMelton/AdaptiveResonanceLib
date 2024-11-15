"""Some things to consider in the future.

Removing entire labels from the dataset should be possible.

Right now I think its not possible for Fuzzy Art to delete a cluster, so no need to do
so now.

Creating functions to explictly add a sample, remove or switch, instead of requiring the
update call would be nice Or at least change the function names for add, remove, and
switch, since they imply that its done, not that it will update after update is called.

"""
import numpy as np
from typing import Dict


def delta_add_sample_to_average(
    average: float, sample: float, total_samples: int
) -> float:
    """Calculate the new average if a sample is added.

    Parameters
    ----------
    average : float
        Current average.
    sample : float
        New sample to be added.
    total_samples : int
        Total number of samples including the new one.

    Returns
    -------
    float
        Updated average after adding the sample.

    """
    return (sample - average) / total_samples


def delta_remove_sample_from_average(
    average: float, sample: float, total_samples: int
) -> float:
    """Calculate the new average if a sample is removed.

    Parameters
    ----------
    average : float
        Current average.
    sample : float
        Sample to be removed.
    total_samples : int
        Total number of samples before removal.

    Returns
    -------
    float
        Updated average after removing the sample.

    """
    return (average - sample) / (total_samples - 1)


class iCVI_CH:
    """Implementation of the Calinski Harabasz Validity Index in incremental form.

    Expanded implementation of the incremental version of the Calinski Harabasz Cluster
    Validity Index.

    The original matlab code can be found at
    https://github.com/ACIL-Group/iCVI-toolbox/blob/master/classes/CVI_CH.m
     The formulation is available at
    scholarsmine.mst.edu/cgi/viewcontent.cgi?article=3833&context=doctoral_dissertations
    Pages 314-316 and 319-320

    This implementation returns a dictionary of updated parameters when calling
    functions, which can then be passed with the update function to accept the changes.
    This allows for testing changes/additions to the categories without doing a deep
    copy of the object.

    In addition, the calculations for removing a sample, or switching the label of a
    sample from the dataset are included. This allows for very efficient calculations
    on clustering algorithms that would like to prune or adjust the labels of samples
    in the dataset.

    For the Calinski Harabasz validity Index, larger values represent better clusters.

    """

    def __init__(self, x: np.ndarray) -> None:
        """Create the iCVI_CH object.

        Parameters
        ----------
        x : np.ndarray
            A sample from the dataset used for recording data dimensionality.

        """
        self.dim = x.shape[0]  # Dimension of the input data
        self.n_samples: int = 0  # number of samples encountered
        self.mu = np.array([])  # geometric mean of the data
        self.CD: Dict = {}  # Dict for each cluster label containing n,v,CP, and G
        self.WGSS = 0  # within group sum of squares
        self.criterion_value = 0  # calcualted CH index

    def add_sample(self, x: np.ndarray, label: int) -> Dict:
        """Calculate the result of adding a new sample with a given label.

        Parameters
        ----------
        x : np.ndarray
            The sample to add to the current validity index calculation.
        label : int
            The sample category/cluster.

        Returns
        -------
        dict
            A dictionary containing the updated values after the sample is added.

        """
        newP = {"x": x, "label": label}  # New Parameters
        newP["n_samples"] = self.n_samples + 1
        if self.mu.size == 0:  # mu will be size 0 if no samples in dataset.
            newP["mu"] = x
        else:
            newP["mu"] = self.mu + delta_add_sample_to_average(
                self.mu, x, newP["n_samples"]
            )

        CD: Dict = {}
        newP["CD"] = CD
        SEP = []  # separation between each cluster and the mean of the data
        if label not in self.CD:
            n_clusters = len(self.CD) + 1
            CD["n"] = 1
            CD["v"] = x
            CD["CP"] = 0
            CD["G"] = np.zeros(self.dim)
            newP["CP_diff"] = 0
            diff = x - newP["mu"]
            SEP.append(
                sum(diff**2)
            )  # Handle SEP for new clusters now instead of later.
        else:
            n_clusters = len(self.CD)
            Data = self.CD[label]
            CD["n"] = Data["n"] + 1

            # The paper defines deltaV = Vold - Vnew, so I need to switch this sign.
            # Consider changing functions to do this.
            deltaV = -1 * delta_add_sample_to_average(Data["v"], x, CD["n"])
            CD["v"] = Data["v"] - deltaV  # Vnew = Vold - deltaV
            diff_x_v = x - CD["v"]

            newP["CP_diff"] = (
                (diff_x_v @ diff_x_v.T)
                + (CD["n"] - 1) * (deltaV @ deltaV.T)
                + 2 * (deltaV @ Data["G"])
            )
            CD["CP"] = Data["CP"] + newP["CP_diff"]
            CD["G"] = Data["G"] + diff_x_v + (CD["n"] - 1) * deltaV

        if n_clusters < 2:
            newP["criterion_value"] = 0
        else:
            for i in self.CD:  # A new label won't be processed, but handled earlier
                if i == label:
                    n = CD["n"]
                    v = CD["v"]
                else:
                    n = self.CD[i]["n"]
                    v = self.CD[i]["v"]
                diff = v - newP["mu"]
                SEP.append(n * sum(diff**2))

            WGSS = self.WGSS + newP["CP_diff"]
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters,
                # which is a divide by 0 error.
                newP["criterion_value"] = 0
            else:
                newP["criterion_value"] = (
                    (BGSS / WGSS) * (newP["n_samples"] - n_clusters) / (n_clusters - 1)
                )
        return newP

    def update(self, params: dict) -> None:
        """Update the parameters of the object. Takes the updated params from
        adding/removing a sample or switching its label, and updates the object.
        Switching a label needs more updates, so those dicts have an extra set of things
        to update, signified with the 'label2' key existing.

        Parameters
        ----------
        params : dict
            Dictionary containing the updated parameters to be applied.

        """
        self.n_samples = params["n_samples"]
        self.mu = params["mu"]
        self.criterion_value = params["criterion_value"]
        self.CD[params["label"]] = params["CD"]
        self.WGSS += params["CP_diff"]
        if "label2" in params:
            self.CD[params["label2"]] = params["CD2"]
            self.WGSS += params["CP_diff2"]

    def switch_label(self, x: np.ndarray, label_old: int, label_new: int) -> dict:
        """Calculate the parameters when a sample has its label changed.

        This essentially removes a sample with the old label from the clusters, then
        adds it back with the new sample. There are a few optimizations, such as
        keeping mu the same since adding and removing it doesn't affect any calculations
        that are needed.

        Otherwise it should work the same as removing a sample and updating, then
        adding the sample back and updating, without the need to create a deep copy of
        the object if just testing the operation.

        Parameters
        ----------
        x : np.ndarray
            The sample whose label is being changed.
        label_old : int
            The old label of the sample.
        label_new : int
            The new label of the sample.

        Returns
        -------
        dict
            A dictionary containing the updated values after switching the label.

        """
        if label_new == label_old:
            return {
                "n_samples": self.n_samples,
                "mu": self.mu,
                "criterion_value": self.criterion_value,
                "label": label_old,
                "CP_diff": 0,
                "CD": {
                    "n": self.CD[label_old]["n"],
                    "v": self.CD[label_old]["v"],
                    "CP": self.CD[label_old]["CP"],
                    "G": self.CD[label_old]["G"],
                },
            }
        if self.CD[label_old]["n"] <= 1:
            raise Exception("Can't remove a value from a cluster of 1")

        newP = {
            "x": x,
            "label": label_old,
            "label2": label_new,
        }  # New Parameters
        newP["mu"] = self.mu
        newP["n_samples"] = self.n_samples
        params_remove = self.remove_sample(x, label_old)
        params_add = self.add_sample(x, label_new)
        newP["CD"] = params_remove["CD"]
        newP["CP_diff"] = params_remove["CP_diff"]
        newP["CD2"] = params_add["CD"]
        newP["CP_diff2"] = params_add["CP_diff"]

        SEP = []  # separation between each cluster and the mean of the data
        if label_new not in self.CD:
            n_clusters = len(self.CD) + 1
            diff = x - newP["mu"]
            SEP.append(
                sum(diff**2)
            )  # Handle SEP for new clusters now instead of later.
        else:
            n_clusters = len(self.CD)

        if n_clusters < 2:  # I don't think this can ever happen
            newP["criterion_value"] = 0
        else:
            for i in self.CD:
                if i == label_old:
                    n = params_remove["CD"]["n"]
                    v = params_remove["CD"]["v"]
                elif i == label_new:
                    n = params_add["CD"]["n"]
                    v = params_add["CD"]["v"]
                else:
                    n = self.CD[i]["n"]
                    v = self.CD[i]["v"]
                diff = v - newP["mu"]
                SEP.append(n * sum(diff**2))

            WGSS = self.WGSS + newP["CP_diff"]
            WGSS += newP["CP_diff2"]
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters,
                # which is a divide by 0 error.
                newP["criterion_value"] = 0
            else:
                newP["criterion_value"] = (
                    (BGSS / WGSS) * (newP["n_samples"] - n_clusters) / (n_clusters - 1)
                )
        return newP

    def remove_sample(
        self, x: np.ndarray, label: int
    ) -> dict:  # This is left here mostly as an extra, and not really meant to be used.
        """Remove a sample from the clusters.

        Parameters
        ----------
        x : np.ndarray
            The sample to remove from the current validity index calculation.
        label : int
            The sample category/cluster.

        Returns
        -------
        dict
            A dictionary containing the updated values after the sample is removed.

        """
        Data = self.CD[label]
        if Data["n"] <= 1:
            raise Exception(
                "Can't remove a value from a cluster of 1"
            )  # At least for now

        newP = {"x": x, "label": label}  # New Parameters after removal
        newP["mu"] = self.mu - delta_remove_sample_from_average(
            self.mu, x, self.n_samples
        )
        newP["n_samples"] = self.n_samples - 1

        CD = {}
        newP["CD"] = CD
        CD["n"] = Data["n"] - 1

        # We need the delta v from when the sample was added,
        # but the paper defines deltaV = Vold - Vnew, so I need to keep these signs
        deltaVPrior = delta_remove_sample_from_average(Data["v"], x, Data["n"])
        CD["v"] = Data["v"] + deltaVPrior  # Vnew + deltaV = Vold
        diff_x_vPrior = x - Data["v"]

        # We already subtracted 1 from n
        CD["G"] = Data["G"] - (diff_x_vPrior + (CD["n"]) * deltaVPrior)
        # CD's G is the old G, which is what we added before.
        newP["CP_diff"] = -1 * (
            (diff_x_vPrior @ diff_x_vPrior.T)
            + (CD["n"]) * (deltaVPrior @ deltaVPrior.T)
            + 2 * (deltaVPrior @ CD["G"])
        )
        CD["CP"] = Data["CP"] + newP["CP_diff"]

        n_clusters = len(self.CD)  # Move this up if deleting clusters are allowed.
        if n_clusters < 2:
            newP["criterion_value"] = 0
        else:
            SEP = []  # separation between each cluster and the mean of the data
            for i in self.CD:
                if i == label:
                    n = CD["n"]
                    v = CD["v"]
                else:
                    n = self.CD[i]["n"]
                    v = self.CD[i]["v"]
                diff = v - newP["mu"]
                SEP.append(n * sum(diff**2))

            WGSS = self.WGSS + newP["CP_diff"]
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters,
                # which is a divide by 0 error.
                newP["criterion_value"] = 0
            else:
                newP["criterion_value"] = (
                    (BGSS / WGSS) * (newP["n_samples"] - n_clusters) / (n_clusters - 1)
                )
        return newP
