from .BaseICVI import Base_iCVI, distance_squared


class CH(Base_iCVI):
    increasing_optimal = True

    def calculate_criterion(self, newP) -> None:
        """Calculate the criterion value for Calinkski Harabasz

        Need to calculate BGSS, the squared distance between each cluster centriod and the data centriod times the number of samples in the cluster.
        WGSS is stored in the base class, but the changes to compactness from labels needs to be accounted for.
        """
        BGSS = 0
        n_clusters = len(self.CD)

        if newP['label'] not in self.CD:
            n_clusters += 1
            diff = newP['CD']['v'] - newP['mu']
            BGSS += distance_squared(diff)

        if newP['CD']['n'] == 0:
            n_clusters -= 1

        if 'label2' in newP:
            if newP['label2'] not in self.CD:
                n_clusters += 1
                diff = newP['CD2']['v'] - newP['mu']
                BGSS += distance_squared(diff)

            if newP['CD2']['n'] == 0:
                n_clusters -= 1

        if n_clusters < 2:
            newP['criterion_value'] = 0
        else:
            for i in self.CD:
                if i == newP['label']:
                    n = newP['CD']['n']
                    v = newP['CD']['v']
                elif 'label2' in newP and i == newP['label2']:
                    n = newP['CD2']['n']
                    v = newP['CD2']['v']
                else:
                    n = self.CD[i]['n']
                    v = self.CD[i]['v']
                diff = v - newP['mu']
                BGSS += n * distance_squared(diff)

            WGSS = self.WGSS + newP['CPdiff']
            if 'label2' in newP:
                WGSS += newP['CPdiff2']
            if WGSS == 0:  # this can be 0 if all samples in different clusters, which is a divide by 0 error.
                newP['criterion_value'] = 1
            else:
                newP['criterion_value'] = (BGSS / WGSS) * (newP['n_samples'] - n_clusters) / (n_clusters - 1)
