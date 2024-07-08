import numpy as np
import gudhi as gd
import gudhi.representations
import torch
from collections import namedtuple
from ripser import ripser
from functools import wraps
import warnings

warnings.filterwarnings("ignore")

Stats = namedtuple('Stats', ["entropy", "mean", "std"])

# TODO for further regularization experiments without projection
def pers_entropy_loss(points: torch.Tensor):
    pass


# from Birdal et al, Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks (NIPS 2021)
def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=isRandom)
    return W[random_indices]

def write_to_file(func):
    @wraps(func)
    def wrapper(write_to_file = None, *args, **kwargs):
        if write_to_file is None:
            return func(*args, **kwargs)
        res = func(*args, **kwargs)
        with open(write_to_file, "a") as f:
            print(res, end="\n", file=f)
        return None
    return wrapper
        
        
@write_to_file
def calculate_ph_dim(W, min_points=225, max_points=1000, point_jump=50,  
        h_dim=0, print_error=False):
    
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        diagrams = ripser(sample_W(W, n))['dgms']
        
        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append((d[:, 1] - d[:, 0]).sum())
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)

@write_to_file
def compute_tda_features(points: torch.Tensor, *, max_dim: int = 2):

    rips = gd.RipsComplex(points=points, max_edge_length=np.inf)

    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()

    _, bars = zip(*st.persistence())

    # filter inf components
    bars = np.array([bar for bar in bars if bar[1] != np.inf])

    entropy = gd.representations.Entropy()

    stat_ent = entropy.fit_transform([bars]).item()

    bar_length = bars[:, 1] - bars[:, 0]

    return Stats(stat_ent, np.mean(bar_length), np.std(bar_length))
