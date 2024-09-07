import numpy as np
from abc import ABC, abstractmethod


def delta_add_sample_to_average(average, sample, totalSamples):
    """Calculate the new average if sample is added"""
    return (sample - average) / totalSamples


def delta_remove_sample_from_average(average, sample, totalSamples):
    """Calculate the new average if sample is removed"""
    return (average - sample) / (totalSamples - 1)


def distance_squared(vector: np.ndarray):
    "Calculate the squared euclidean distance"
    return vector @ vector.T


class Base_iCVI(ABC):
    """Abstract implementation for Incremental Cluster Validity Indexes

    Calculations derived from
    L. E. Brito da Silva, N. Rayapati and D. C. Wunsch,
    "iCVI-ARTMAP: Using Incremental Cluster Validity Indices and Adaptive Resonance Theory Reset Mechanism
    to Accelerate Validation and Achieve Multiprototype Unsupervised Representations,"
    in IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 12, pp. 9757-9770, Dec. 2023, doi: 10.1109/TNNLS.2022.3160381.

    https://ieeexplore.ieee.org/document/9745260

    This implementation returns a dictionary of updated parameters when calling functions, which can then be passed with
    the update function to accept the changes. This allows for testing changes/additions to the categories without doing a
    deep copy of the object.

    Implemented methods:
        Add Sample to cluster
        Remove Sample from cluster
        Switch Sample from one cluster to another
        Merge 2 clusters together.

    Calculation of the co-variance matrix included in the paper is not implemented.

    Derived classes implement the calculation for their criterion value, and define the increasing_optimal property

    Parameters:
        n_samples: an int with the total number of samples.
        mu: a numpy array representing the average of all data points.
        CD: ClusterData... a dict holding all the data for each cluster, with the parameters.
            n: the number of samples belonging to each cluster/label.
            v: the prototypes/centriods of each cluster.
            CP: the compactness of each cluster.
        WGSS: Within Group Sum of Squares. Equal to the sum of the compactness of all clusters.
        criterion_value: The calculated criterion value.
        increasing_optimal: Defines if a larger or smaller criterion value represents a better clustering.
    """

    @property
    @abstractmethod
    def increasing_optimal(self):
        pass

    def __init__(self) -> None:
        """Create the iCVI object.

        """
        self.n_samples: int = 0   # number of samples encountered
        self.mu = np.array([])    # geometric mean of the data
        self.CD = {}              # Dict for each cluster label containing n, v, CP
        self.WGSS = 0             # within group sum of squares, equal to sum of CP for all clusters.
        self.criterion_value = 0  # calcualted CH index

    def add_sample(self, x, label, calc_criterion=True) -> dict:
        """Calculate the result of adding a new sample with a given label.

        Create a dictionary containing the updated values after assigning a label to a given sample.
        To accept the outcome of the sample being added, pass the returned parameter dict to update.

        In general, if newP['criterion_value'] > obj.criterion_value, the clustering has been improved.

        Args:
            x: The sample to add to the current validity index calculation
            label: an int representing the sample category/cluster

        Returns:
            newP: a dictionary contained the values after the sample is added, to be passed to update call.
        """
        newP = {'x': x, 'label': label}  # New Parameters
        newP['n_samples'] = self.n_samples + 1
        if self.mu.size == 0:  # mu will be size 0 if no samples in dataset.
            newP['mu'] = x
        else:
            newP['mu'] = self.mu + delta_add_sample_to_average(self.mu, x, newP['n_samples'])

        CD = {}
        newP['CD'] = CD
        if label not in self.CD:
            CD['n'] = 1
            CD['v'] = x
            CD['CP'] = 0
            newP['CPdiff'] = 0
        else:
            Data = self.CD[label]
            CD['n'] = Data['n'] + 1

            deltaV = delta_add_sample_to_average(Data['v'], x, CD['n'])
            CD['v'] = Data['v'] + deltaV
            diff_x_v = x - Data['v']

            newP['CPdiff'] = Data['n'] / CD['n'] * distance_squared(diff_x_v)
            CD['CP'] = Data['CP'] + newP['CPdiff']

        if calc_criterion:
            self.calculate_criterion(newP)
        return newP

    def update(self, params) -> None:
        """Update the parameters of the object.

        Takes the updated params from adding/removing a sample or switching its label, and updates the object.
        Switching a label needs more updates, so those dicts have an extra set of things to update, signified with
        the 'label2' key existing

        Args:
            params: dict containing the parameters to update.
        """
        self.n_samples = params['n_samples']
        self.mu = params['mu']
        self.criterion_value = params['criterion_value']

        if params['CD']['n'] == 0:
            del self.CD[params['label']]
            self.WGSS -= params['CD']['CP']
        else:
            self.CD[params['label']] = params['CD']
            self.WGSS += params['CPdiff']

        if 'label2' in params:
            if params['CD2']['n'] == 0:
                del self.CD[params['label2']]
                self.WGSS -= params['CD2']['CP']
            else:
                self.CD[params['label2']] = params['CD2']
                self.WGSS += params['CPdiff2']

    def switch_label(self, x, label_old, label_new):
        """Calculates the parameters if a sample has its label changed.

        This essentially removes a sample with the old label from the clusters, then adds it back with the new sample.
        There are a few optimizations, such as keeping mu the same since adding and removing it doesn't affect any calculations
        that are needed.

        Otherwise it should work the same as removing a sample and updating, then adding the sample back and updating, without
        the need to create a deep copy of the object if just testing the operation.

        Args:
            x: The sample to switch the label of for the current validity index calculation
            label_old: an int representing the sample category/cluster the sample belongs to
            label_new: an int representing the sample category/cluster the sample will be assigned to

        Returns:
            newP: a dictionary contained the values after the sample is added, to be passed to update call."""
        if label_new == label_old:
            return {'n_samples': self.n_samples,
                    'mu': self.mu,
                    'criterion_value': self.criterion_value,
                    'label': label_old,
                    'CPdiff': 0,
                    'CD': {
                        'n': self.CD[label_old]['n'],
                        'v': self.CD[label_old]['v'],
                        'CP': self.CD[label_old]['CP']}
                    }
        if self.CD[label_old]['n'] <= 0:
            raise Exception("Can't remove a value from a cluster of 0")

        newP = {'x': x, 'label': label_old, 'label2': label_new}
        newP['mu'] = self.mu
        newP['n_samples'] = self.n_samples
        params_remove = self.remove_sample(x, label_old, False)
        params_add = self.add_sample(x, label_new, False)
        newP['CD'] = params_remove['CD']
        newP['CPdiff'] = params_remove['CPdiff']
        newP['CD2'] = params_add['CD']
        newP['CPdiff2'] = params_add['CPdiff']

        self.calculate_criterion(newP)
        return newP

    def remove_sample(self, x, label, calc_criterion=True):
        """Remove a sample from the clusters

        Args:
            x: The sample to remove from the current validity index calculation
            label: an int representing the sample category/cluster

        Returns:
            newP: a dictionary contained the values after the sample is remove, to be passed to update call.
        """
        Data = self.CD[label]
        newP = {'x': x, 'label': label}  # New Parameters after removal
        newP['mu'] = self.mu + delta_remove_sample_from_average(self.mu, x, self.n_samples)
        newP['n_samples'] = self.n_samples - 1

        CD = {}
        newP['CD'] = CD
        if Data['n'] <= 1:
            CD['n'] = 0
            CD['v'] = x  # This won't contribute anything, but needs to be the right size.
            CD['CP'] = 0
            newP['CPdiff'] = -1 * Data['CP']
        else:
            CD['n'] = Data['n'] - 1

            deltaV = delta_remove_sample_from_average(Data['v'], x, Data['n'])
            CD['v'] = Data['v'] + deltaV
            diff_x_v = x - Data['v']
            newP['CPdiff'] = -1 * Data['n'] / CD['n'] * distance_squared(diff_x_v)
            CD['CP'] = Data['CP'] + newP['CPdiff']

        if calc_criterion:
            self.calculate_criterion(newP)
        return newP

    def merge_clusters(self, delete_label, merge_label):
        """Merge 2 clusters together

         Calculates parameters after combining 2 clusters into 1.

        Args:
            delete_label: The cluster to remove from the current validity index calculation
            merge_label: The cluster label which the cluster data will be moved to.

        Returns:
            newP: a dictionary contained the values after the merge, to be passed to update call.
        """
        newP = {'label': delete_label, 'label2': merge_label, 'mu': self.mu, 'n_samples': self.n_samples}

        delete_data = self.CD[delete_label]
        merge_data = self.CD[merge_label]

        newP['CPdiff'] = -1 * delete_data['CP']
        newP['CD'] = {
            'n': 0,
            'v': delete_data['v'],  # This won't contribute anything, but needs to be the right size.
            'CP': 0
        }
        CD = {}
        newP['CD2'] = CD

        CD['n'] = delete_data['n'] + merge_data['n']
        CD['v'] = (delete_data['n'] * delete_data['v'] + merge_data['n'] * merge_data['v']) / CD['n']

        diff_v = delete_data['v'] - merge_data['v']
        newP['CPdiff2'] = delete_data['CP'] + (diff_v @ diff_v.T) * delete_data['n'] * merge_data['n'] / CD['n']
        CD['CP'] = merge_data['CP'] + newP['CPdiff2']
        return newP

    def clustering_improved(self, newP, vigilence=0) -> bool:
        """If the clustering has been improved.

        Compares the criterion_value between the updated parameters, and the current parameters,
        and notes if clustering has been improved.

        Args:
            newP: Parameters dict from one of the ICVI operations.
            vigilence: Optional value to require a larger change in criterion_value to approve clustering improving

        Returns:
            bool: If the clustering has been improved.
        """
        if self.increasing_optimal:
            return newP['criterion_value'] >= (self.criterion_value + vigilence)
        else:
            return newP['criterion_value'] <= (self.criterion_value - vigilence)

    @abstractmethod
    def calculate_criterion(self, newP) -> None:
        """Abstract method for the criterion calculation

        Calculates the criterion value, and stores it into the parameter dictionary.

        Several unique cases need to be handled.

        If a label doesn't exist in self.CD dict (new label)
        If a label's CD['n'] value is 0 (cluster is being deleted)
        If there is a 'label2' key in the parameters (2 clusters are being editted)

        This just assigns the criterion value to newP, and doesn't change anything else.

        Args:
            newP: Dictionary with the updated parameters. The 'criterion_value' key needs to be assigned.

        Returns:
            None
        """
        pass
