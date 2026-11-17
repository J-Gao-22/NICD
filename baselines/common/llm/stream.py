import numpy as np
from torch.utils.data import default_collate

class SampleStream:

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        self.order = self.rng.permutation(len(dataset))
        self.position = 0

    def next(self, count):
        indices = []
        while len(indices) < count:
            if self.position == len(self.order):
                self.order = self.rng.permutation(len(self.dataset))
                self.position = 0
            take = min(count - len(indices), len(self.order) - self.position)
            indices.extend(self.order[self.position:self.position + take].tolist())
            self.position += take
        return default_collate([self.dataset[index] for index in indices])

    def state_dict(self):
        return dict(order=self.order, position=self.position, rng=self.rng.get_state())

    def load_state_dict(self, state):
        assert len(state['order']) == len(self.dataset)
        self.order, self.position = (state['order'], state['position'])
        self.rng.set_state(state['rng'])
