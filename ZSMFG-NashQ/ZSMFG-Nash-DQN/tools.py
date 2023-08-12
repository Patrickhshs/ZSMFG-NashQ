import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")