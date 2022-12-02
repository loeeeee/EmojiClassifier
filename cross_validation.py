import numpy as np

class CrossValidation:
    def __init__(self, data, partition_num: int = 5):
        self.data = data
        self.partition_num = partition_num
        self.ele_per_par = int(len(self.data) / self.partition_num)

        self._current_ind = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_ind + 1 > self.partition_num:
            raise StopIteration

        train = np.concatenate((self.data[:self.ele_per_par*(self._current_ind)],self.data[self.ele_per_par*(self._current_ind+1):]))
        test = np.array(self.data[self.ele_per_par*(self._current_ind):self.ele_per_par*(self._current_ind+1)])
        
        self._current_ind += 1
        return train,test