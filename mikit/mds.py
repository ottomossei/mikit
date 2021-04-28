import pandas as pd
import numpy as np
from sklearn.manifold import MDS


class MDSBit:
    def __init__(self):
        pass
    
    @staticmethod
    def _tanimoto_from_bits(bit1, bit2, i, j):
        counts_and = np.count_nonzero(bit1 + bit2 == 2)
        counts_or = np.count_nonzero(bit1 + bit2 != 0)
        return 1 - counts_and / counts_or
        
    def _calc_mds(self, n):
        matrix =  np.zeros((self.atom_num, self.atom_num))
        for i, atom_1 in enumerate(self.data.T):
            for j, atom_2 in enumerate(self.data.T):
                matrix[i, j] = self._tanimoto_from_bits(atom_1, atom_2, i, j)
        embedding = MDS(n_components = n, dissimilarity = "precomputed", random_state = 1234)
        X_transformed = embedding.fit_transform(matrix)
        df = pd.DataFrame(X_transformed)
        stress = embedding.stress_
        s = np.sum(matrix) / 2
        print("dimennsions : " + str(n) + ", error rate : " + str((stress / s)))
        return (stress / s), df

    def get_mds(self, data, rate=0.05):
        self.data = data
        self.atom_num = data.shape[1]
        n, e_rate = np.zeros(self.atom_num), np.zeros(self.atom_num)
        for i in reversed(range(1,self.atom_num)):
            e_rate[i], df = self._calc_mds(i)
            if e_rate[i] >= rate:
                output = old_df
                break
            old_df = df
        return output