import os
import time
import copy
import random
import math
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

import IPython


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
