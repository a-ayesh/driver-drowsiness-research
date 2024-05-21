import torch
import torch.nn.functional as F
from util import pml, extract_features, pca


class CovarianceDescriptorTransform(object):
    def __init__(self, levels):
        self.levels = levels

    def __call__(self, image):
        
        cov_descriptor = pml(self.levels, image.numpy())
        cov_descriptor = torch.tensor(pca(cov_descriptor, 6), dtype=torch.float32).unsqueeze(0)
        return cov_descriptor


