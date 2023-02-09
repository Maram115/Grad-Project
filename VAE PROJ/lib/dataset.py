from sklearn.datasets import make_swiss_roll
from .utils import show
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader




G_MEAN=0.0
G_STD=1.0
G_SET_STD=1.0


class DatasetBase(object):
    def gen_data_xyz(self, size=512):
        raise NotImplementedError('you need implement gen_data_xy')

    def __len__(self):
        return len(self.samples)

    def show(self, fix=False, ax=None):
        samples = self.gen_data_xyz()
        show(samples, type(self).__name__, fix=fix, ax=ax)

    def normalize(self,set_std):
        global G_MEAN,G_STD,G_SET_STD
        if self.iscenter:
            G_MEAN=np.mean(self.data,axis=0,keepdims=True)
            G_STD=np.std(self.data,axis=0,keepdims=True)
            G_SET_STD=set_std
            self.data=(self.data-G_MEAN)/G_STD*set_std

class DatasetRandom(DatasetBase):
    def gen_data_xyz(self, size=1024):
        #rand_data=torch.rand(size,3)
        rand_data = np.random.rand(size,3)
        #samples = rand_data[:,[0,3]]
        dataloader=DataLoader(rand_data,shuffle = True)
        return dataloader

