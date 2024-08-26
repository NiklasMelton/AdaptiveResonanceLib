"""
Some things to consider in the future.

Removing entire labels from the dataset should be possible.

Right now I think its not possible for Fuzzy Art to delete a cluster, so no need to do so now.

Creating functions to explictly add a sample, remove or switch, instead of requiring the update call would be nice
Or at least change the function names for add, remove, and switch, since they imply that its done, not that it will
update after update is called.
"""
import numpy as np


def deltaAddSampleToAverage(average, sample, totalSamples):
    """Calculate the new average if sample is added"""
    return (sample - average) / totalSamples


def deltaRemoveSampleFromAverage(average, sample, totalSamples):
    """Calculate the new average if sample is removed"""
    return (average - sample) / (totalSamples - 1)


class iCVI_CH():
    """Implementation of the Calinski Harabasz Validity Index in incremental form.

    Expanded implementation of the incremental version of the Calinski Harabasz Cluster Validity Index.

    The original matlab code can be found at https://github.com/ACIL-Group/iCVI-toolbox/blob/master/classes/CVI_CH.m
    The formulation is available at
    https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=3833&context=doctoral_dissertations Pages 314-316 and 319-320

    This implementation returns a dictionary of updated parameters when calling functions, which can then be passed with
    the update function to accept the changes. This allows for testing changes/additions to the categories without doing a
    deep copy of the object.

    In addition, the calculations for removing a sample, or switching the label of a sample from the dataset are included.
    This allows for very efficient calculations on clustering algorithms that would like to prune or adjust the labels of
    samples in the dataset.

    For the Calinski Harabasz validity Index, larger values represent better clusters.

    Parameters:
        dim: an int storing the dimensionality of the input data.
        nSamples: an int with the total number of samples.
        mu: a numpy array representing the average of all data points.
        CD: ClusterData... a dict holding all the data for each cluster, with the parameters
            n: the number of samples belonging to each cluster/label.
            v: the prototypes/centriods of each cluster.
            CP: the compactness of each cluster. Not really needed since WGSS can be calculated incrementally.
            G: the vector g of each cluster.
        WGSS: Within Groupd sum of squares.
        criterionValue: the calculated CH score.
    """

    def __init__(self, x: np.ndarray) -> None:
        """Create the iCVI object.

        Args:
            x: a sample from the dataset used for recording data dimensionality.
        """
        self.dim = x.shape[0]  # Dimension of the input data
        self.nSamples: int = 0  # number of samples encountered
        self.mu = np.array([])  # geometric mean of the data
        self.CD = {}            # Dict for each cluster label containing n,v,CP, and G
        self.WGSS = 0           # within group sum of squares
        self.criterionValue = 0  # calcualted CH index

    def addSample(self, x, label) -> dict:
        """Calculate the result of adding a new sample with a given label.

        Create a dictionary containing the updated values after assigning a label to a given sample.
        To accept the outcome of the sample being added, pass the returned parameter dict to update.

        In general, if newP['criterionValue'] > obj.criterionValue, the clustering has been improved.

        Args:
            x: The sample to add to the current validity index calculation
            label: an int representing the sample category/cluster

        Returns:
            newP: a dictionary contained the values after the sample is added, to be passed to update call.
        """
        newP = {'x': x, 'label': label}  # New Parameters
        newP['nSamples'] = self.nSamples + 1
        if self.mu.size == 0:  # mu will be size 0 if no samples in dataset.
            newP['mu'] = x
        else:
            newP['mu'] = self.mu + deltaAddSampleToAverage(self.mu, x, newP['nSamples'])

        CD = {}
        newP['CD'] = CD
        SEP = []  # separation between each cluster and the mean of the data
        if label not in self.CD:
            nClusters = len(self.CD) + 1
            CD['n'] = 1
            CD['v'] = x
            CD['CP'] = 0
            CD['G'] = np.zeros(self.dim)
            newP['CPdiff'] = 0
            diff = x - newP['mu']
            SEP.append(sum(diff ** 2))  # Handle SEP for new clusters now instead of later.
        else:
            nClusters = len(self.CD)
            Data = self.CD[label]
            CD['n'] = Data['n'] + 1

            # The paper defines deltaV = Vold - Vnew, so I need to switch this sign. Consider changing functions to do this.
            deltaV = -1 * deltaAddSampleToAverage(Data['v'], x, CD['n'])
            CD['v'] = Data['v'] - deltaV  # Vnew = Vold - deltaV
            diff_x_v = x - CD['v']

            newP['CPdiff'] = (diff_x_v @ diff_x_v.T) + (CD['n'] - 1) * (deltaV @ deltaV.T) + 2 * (deltaV @ Data['G'])
            CD['CP'] = Data['CP'] + newP['CPdiff']
            CD['G'] = Data['G'] + diff_x_v + (CD['n'] - 1) * deltaV

        if nClusters < 2:
            newP['criterionValue'] = 0
        else:
            for i in self.CD:  # A new label won't be processed, but handled earlier
                if i == label:
                    n = CD['n']
                    v = CD['v']
                else:
                    n = self.CD[i]['n']
                    v = self.CD[i]['v']
                diff = v - newP['mu']
                SEP.append(n * sum(diff ** 2))

            WGSS = self.WGSS + newP['CPdiff']
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters, which is a divide by 0 error.
                newP['criterionValue'] = 0
            else:
                newP['criterionValue'] = (BGSS / WGSS) * (newP['nSamples'] - nClusters) / (nClusters - 1)
        return newP

    def update(self, params) -> None:
        """Update the parameters of the object.

        Takes the updated params from adding/removing a sample or switching its label, and updates the object.
        Switching a label needs more updates, so those dicts have an extra set of things to update, signified with
        the 'label2' key existing

        Args:
            params: dict containing the parameters to update.
        """
        self.nSamples = params['nSamples']
        self.mu = params['mu']
        self.criterionValue = params['criterionValue']
        self.CD[params['label']] = params['CD']
        self.WGSS += params['CPdiff']
        if 'label2' in params:
            self.CD[params['label2']] = params['CD2']
            self.WGSS += params['CPdiff2']

    def switchLabel(self, x, label_old, label_new):
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
            return {'nSamples': self.nSamples,
                    'mu': self.mu,
                    'criterionValue': self.criterionValue,
                    'label': label_old,
                    'CPdiff': 0,
                    'CD': {
                        'n': self.CD[label_old]['n'],
                        'v': self.CD[label_old]['v'],
                        'CP': self.CD[label_old]['CP'],
                        'G': self.CD[label_old]['G']}
                    }
        if self.CD[label_old]['n'] <= 1:
            raise Exception("Can't remove a value from a cluster of 1")

        newP = {'x': x, 'label': label_old, 'label2': label_new}  # New Parameters
        newP['mu'] = self.mu
        newP['nSamples'] = self.nSamples
        paramsRemove = self.removeSample(x, label_old)
        paramsAdd = self.addSample(x, label_new)
        newP['CD'] = paramsRemove['CD']
        newP['CPdiff'] = paramsRemove['CPdiff']
        newP['CD2'] = paramsAdd['CD']
        newP['CPdiff2'] = paramsAdd['CPdiff']

        SEP = []  # separation between each cluster and the mean of the data
        if label_new not in self.CD:
            nClusters = len(self.CD) + 1
            diff = x - newP['mu']
            SEP.append(sum(diff ** 2))  # Handle SEP for new clusters now instead of later.
        else:
            nClusters = len(self.CD)

        if nClusters < 2:  # I don't think this can ever happen with remove label not deleting categories.
            newP['criterionValue'] = 0
        else:
            for i in self.CD:
                if i == label_old:
                    n = paramsRemove['CD']['n']
                    v = paramsRemove['CD']['v']
                elif i == label_new:
                    n = paramsAdd['CD']['n']
                    v = paramsAdd['CD']['v']
                else:
                    n = self.CD[i]['n']
                    v = self.CD[i]['v']
                diff = v - newP['mu']
                SEP.append(n * sum(diff ** 2))

            WGSS = self.WGSS + newP['CPdiff']
            WGSS += newP['CPdiff2']
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters, which is a divide by 0 error.
                newP['criterionValue'] = 0
            else:
                newP['criterionValue'] = (BGSS / WGSS) * (newP['nSamples'] - nClusters) / (nClusters - 1)
        return newP

    def removeSample(self, x, label):  # This is left here mostly as an extra, and not really meant to be used.
        """Remove a sample from the clusters

         Calculates parameters after removing a sample from the clusters, or the opposite of an add operation.

        Args:
            x: The sample to remove from the current validity index calculation
            label: an int representing the sample category/cluster

        Returns:
            newP: a dictionary contained the values after the sample is remove, to be passed to update call.
        """
        Data = self.CD[label]
        if Data['n'] <= 1:
            raise Exception("Can't remove a value from a cluster of 1")  # At least for now

        newP = {'x': x, 'label': label}  # New Parameters after removal
        newP['mu'] = self.mu - deltaRemoveSampleFromAverage(self.mu, x, self.nSamples)
        newP['nSamples'] = self.nSamples - 1

        CD = {}
        newP['CD'] = CD
        CD['n'] = Data['n'] - 1

        # We need the delta v from when the sample was added, but the paper defines deltaV = Vold - Vnew, so I need to keep these signs
        deltaVPrior = deltaRemoveSampleFromAverage(Data['v'], x, Data['n'])
        CD['v'] = Data['v'] + deltaVPrior  # Vnew + deltaV = Vold
        diff_x_vPrior = x - Data['v']

        # We already subtracted 1 from n
        CD['G'] = Data['G'] - (diff_x_vPrior + (CD['n']) * deltaVPrior)
        # CD's G is the old G, which is what we added before.
        newP['CPdiff'] = -1 * ((diff_x_vPrior @ diff_x_vPrior.T) + (CD['n']) * (deltaVPrior @ deltaVPrior.T) + 2 * (deltaVPrior @ CD['G']))
        CD['CP'] = Data['CP'] + newP['CPdiff']

        nClusters = len(self.CD)  # Move this up if deleting clusters are allowed.
        if nClusters < 2:
            newP['criterionValue'] = 0
        else:
            SEP = []  # separation between each cluster and the mean of the data
            for i in self.CD:
                if i == label:
                    n = CD['n']
                    v = CD['v']
                else:
                    n = self.CD[i]['n']
                    v = self.CD[i]['v']
                diff = v - newP['mu']
                SEP.append(n * sum(diff ** 2))

            WGSS = self.WGSS + newP['CPdiff']
            BGSS = sum(SEP)  # between-group sum of squares
            if WGSS == 0:  # this can be 0 if all samples in different clusters, which is a divide by 0 error.
                newP['criterionValue'] = 0
            else:
                newP['criterionValue'] = (BGSS / WGSS) * (newP['nSamples'] - nClusters) / (nClusters - 1)
        return newP
