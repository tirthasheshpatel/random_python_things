import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, nb_samples, path):
        self.nb_samples = nb_samples
        self.path = path
    
    def gen_dataset(self):
        self.dataset = np.empty((self.nb_samples, 2), dtype=np.float32)
        self.dataset[:, 0] = np.arange(start=1, stop=self.nb_samples+1, dtype=np.float32)
        self.dataset[:, 1] = np.random.randint(low=50, high=100, size=self.nb_samples)
        
        try:
            with open(self.path, 'w') as f:
                for i in range(self.nb_samples):
                    f.write(f"{self.dataset[i, 0]},{self.dataset[i, 1]}\n")
        except:
            print("illegal path")

    def get_dataset(self):
        self.dataset = np.loadtxt(self.path, dtype=np.float32, delimiter=',').reshape(-1, 2)
        return self.dataset

class HistTransformer:
    def __init__(self, dataset, bins=5, usecols=[1]):
        """
        Transforms columns of a dataset to discrete bins.
        
        Parameters
        --------
        dataset: array_like
                    The dataset of shape (nb_samples, nb_features)
        
        bins: int, (default=5)
                    Number of bins in discretized array.
        """
        self.dataset = dataset
        self.bins = bins

    def _discretize_dataset(self, usecol):
        """Discretize a single column of the dataset at a time"""
        # Extract the lowest and highest value of marks
        self.__low = np.min(self.dataset[:, usecol])
        self.__high = np.max(self.dataset[:, usecol])
        self._bins = np.linspace(self.__low,
                                 self.__high,
                                 num=self.bins+1,
                                 endpoint=True,
                                 dtype=np.float32).reshape(-1, )

    def fit_transform(self, usecols=[1]):
        """Discretize the given columns of the dataset"""
        self.counts = np.zeros((self.bins, len(usecols)), dtype=np.int32)
        self.labels = np.zeros((2, self.bins, len(usecols)))
        _col = 0
        for col in usecols:
            data = self.dataset[:, col].reshape(-1,)
            self._discretize_dataset(col)
            for datapoint in data:
                for i in range(1, self._bins.size):
                    if datapoint <= self._bins[i]:
                        self.counts[i-1, _col] += 1
                        self.labels[0, i-1, _col] = self._bins[i-1]
                        self.labels[1, i-1, _col] = self._bins[i]
                        break
            _col += 1


data_generator = Dataset(100, "./dataset.csv")
data_generator.gen_dataset()
dataset = data_generator.get_dataset()

transformer = HistTransformer(dataset)
transformer.fit_transform(usecols=[1])
counts = transformer.counts
labels = transformer.labels

print(transformer._bins)
print(counts)
print(labels)
plt.bar(labels[1, :, 0].reshape(-1, ), counts.reshape(-1, ))
plt.show()
