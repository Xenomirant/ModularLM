import gc
import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm
import sys
import typing

Array = typing.TypeVar('Array')

class GetSVD(object):
    '''
    Computes SVD of some matrix based on dataset statistics -- currently for diff of vectors on custom layer

    TODO: rewrite using dask for CuPy usage. Also makes possible to compute for larger matrices when necessary
    '''

    def __init__(self, *, model: nn.Module, dataloader: torch.utils.data.DataLoader, division_layer: int = None, mean_over_changed: bool = True) -> None:
        '''
        model: nn.Module with forward()
        dataloader: data with two columns that are diffed
        division_layer: actual division layer of model -- currently adds +1 as bert returns embedding outputs as zero layer
        mean_over_changed: bool, whether to take same values into account before 
        '''
        self.model = model
        self.dataloader = dataloader
        self.division_layer = division_layer
        self.matrix = []
        self.svd = None
        

    def get_matrix(self, division_layer = None):

        if division_layer is None:
            division_layer = self.division_layer
        
        ## add one to mitigate design flaws of the model
        division_layer += 1
    
        self.model.eval()
    
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.dataloader)):
                    pred = self.model(**{k: v.to(self.model.device) for k, v in batch[0].items()},
                                 output_hidden_states=True)  
        
                    pred_new = self.model(**{k: v.to(self.model.device) for k, v in batch[1].items()},
                                 output_hidden_states=True)  
        
                    hid_ref = torch.mean(pred.hidden_states[self.division_layer], dim=1)
                    hid_cur = torch.mean(pred_new.hidden_states[self.division_layer], dim=1)
        
                    self.matrix.extend(hid_ref.detach().cpu().numpy() - hid_cur.detach().cpu().numpy())
    
        return None

    def compute_svd(self, matrix = None, compute_uv: bool = False):
        '''
        Can compute SVD on any matrix -- if not specified, gets matrix from precomputed
        '''
        if matrix is None:
            matrix = np.array(self.matrix)
        self.svd = scipy.linalg.svd(matrix, overwrite_a = True, full_matrices = False, compute_uv=compute_uv)
        print("SVD computed")
        return self.svd


def compute_anisotropy(matrix: Array, write_to_file: typing.Optional[str] = None) -> float:
    '''
    Computes anisotropy for a given matrix
    Params:
    matrix -- array for computation
    write_to_file -- whether to write to the specified filename; if not, returns as output
    '''
    res = scipy.linalg.svd(matrix, overwrite_a = True, full_matrices = False, compute_uv=False)

    anisotropy = res[0]**2 / np.sum(res**2)

    if write_to_file is None:
        return anisotropy

    with open(write_to_file, "a") as f:
        print(anisotropy, end="\n", file = f)
    return None
    



    