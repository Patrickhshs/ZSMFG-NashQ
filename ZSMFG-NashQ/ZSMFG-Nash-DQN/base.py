import torch 
import torch.nn as nn
from tqdm import tqdm

class ValueNet():

    def __init__(self,layers=None):
        self.layers = layers
        