'''
This work is adapted from [AL-MDN](https://github.com/NVlabs/AL-MDN), and [SSD](https://github.com/lufficc/SSD/tree/master).
Please, check the README file for proper citation details. 

'''


from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
