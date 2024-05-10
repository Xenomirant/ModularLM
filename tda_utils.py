import numpy as np
import gudhi as gd
import gudhi.representations
import torch
import ot
from collections import namedtuple
from ripser import ripser
from functools import wraps
import warnings

warnings.filterwarnings("ignore")

Stats = namedtuple('Stats', ["entropy", "mean", "std"])


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
        with open(write_to_file, "a") as f:
            print(func(*args, **kwargs), end="\n", file=f)
        return None
    return wrapper
        
        
@write_to_file
def calculate_ph_dim(W, min_points=150, max_points=800, point_jump=50,  
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


def entropy_loss(points: torch.TensorType, max_dim: int = 2):

    # Note: may consider using log1p with \frac{\sum_{i!=j} p_i}{\sum_i p_i}
    
    # default value
    ent = 0
    # compute persistence
    vr = gh.RipsComplex(points=points).create_simplex_tree(max_dimension=max_dim)
    vr.compute_persistence()
    # get critical simplices
    ind0, ind1 = vr.flag_persistence_generators()[:-2]
    
    res0 = torch.norm(points[ind0[:, 1]] - points[ind0[:, 2]], dim=-1)
    ent0 = -torch.sum((res0/torch.sum(res0))*torch.log(res0/torch.sum(res0)))
    
    # compute entropy for higher dimensional simplices 
    for i in ind1:
        res = torch.norm(points[i[:, (0, 2)]] - points[i[:, (1, 3)]], dim=-1)
        lens = res[:, 1] - res[:, 0]
        ent += -torch.sum((lens/torch.sum(lens))*torch.log(lens/torch.sum(lens)))
        
    return ent + ent0


def diagram_divergence_loss(ref: torch.Tensor, cur: torch.Tensor, max_dim: int = 2, num_samples: int = 100):

    was_dist = .0
    
    vr_ref = gh.RipsComplex(points=ref).create_simplex_tree(max_dimension=max_dim)
    vr_ref.compute_persistence()
    vr_cur = gh.RipsComplex(points=cur).create_simplex_tree(max_dimension=max_dim)
    vr_cur.compute_persistence()

    ind0_ref, ind1_ref = vr_ref.flag_persistence_generators()[:-2]
    ind0_cur, ind1_cur = vr_ref.flag_persistence_generators()[:-2]

    crit_ref0 = torch.norm(ref[ind0_ref[:, (0, 1)]] - ref[ind0_ref[:,(0, 2)]], dim=-1)
    crit_cur0 = torch.norm(cur[ind0_cur[:, (0, 1)]] - cur[ind0_cur[:,(0, 2)]], dim=-1)

    zero_ord_dist = ot.sliced_wasserstein_distance(crit_ref0, crit_cur0, n_projections=num_samples)

    for i in range(len(ind1_cur)):
        
        crit_ref = torch.norm(ref[ind1_ref[i][:, (0, 2)]] - ref[ind1_ref[i][:,(1, 3)]], dim=-1)
        crit_cur = torch.norm(cur[ind1_cur[i][:, (0, 2)]] - cur[ind1_cur[i][:,(1, 3)]], dim=-1)

        was_dist += ot.sliced_wasserstein_distance(crit_ref, crit_cur, n_projections=num_samples)

    return was_dist
    