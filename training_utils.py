import torch
import numpy as np
import gc
from tda_utils import entropy_loss, diagram_divergence_loss

def cleanup():

    gc.collect()
    torch.cuda.empty_cache()


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return None


def save_gradients(model, division_layer, logger):
    layers = {}
    for name, param in model.named_parameters():
        # division layer passed == division layer + 1 as is inside train
        if name.startswith(f'bert.encoder.layer.{division_layer}'):
            break
        if (param.requires_grad) and param.grad is not None:
            layers[name] = param.grad.detach().clone()
    if LOG_LAYERS:
        logger.info(f"Saved layers: {str(layers.keys())}")
    return layers


def change_gradients(*, model, layers, 
                     division_layer,
                     logger,
                     weight_mlm=0.5, 
                     weight_loss=1, 
                    ):

    global LOG_LAYERS
    
    for name, param in model.named_parameters():
        # division layer passed == division layer + 1 as is inside train
        if name.startswith(f'bert.encoder.layer.{division_layer}'):
            break
        if name in layers:
            param.grad = weight_loss * param.grad + weight_mlm * layers[name]
            if LOG_LAYERS:
                logger.info(f"Changed layer: {name}")
                logger.info(f"gradients changed. {(weight_loss * param.grad).norm(), (weight_mlm * layers[name]).norm()}\n")
    LOG_LAYERS = False
    return None


class LossWeightDecay:
    '''
    Cosine Weight with decaying step sizes after each multiplication
    '''
    def __init__(self, init_state=1, decay=0.5):
        self.init_state = init_state
        self.cur_state = init_state
        self.decay = decay

    def __mul__(self, other):
        res = self.cur_state * other
        return res

    def __repr__(self):
        return str(self.cur_state)

    def step(self):
        self.cur_state = self.cur_state * self.decay
        return None

    def reset(self):
        self.cur_state = self.init_state
        return None
    
    @property
    def weight(self):
        return self.cur_state


class LossWeightSum2One:
    '''
    Cosine Weight summing to 1 over 10 steps (must be subject to change in case other step size is required)
    '''
    def __init__(self, init_coef = 1, steps: int = 10, linear = False):
        self.counter = -1
        if linear:
            self.steps = init_coef*np.ones(steps)
            return None
        self.steps = init_coef*np.arange(2, steps+2)**-1.5
        return None

    def __mul__(self, other):
        res = self.steps[self.counter] * other
        return res
    
    def step(self):
        self.counter+=1
        return None

    def __repr__(self):
        return str(self.steps[self.counter])

    def reset(self):
        self.counter = -1
        return None

    @property
    def weight(self):
        return self.steps[self.counter]


class CosLoss:
    def __init__(self, vector=None, alpha=0):
        self.loss = torch.nn.CosineEmbeddingLoss()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.target = -torch.ones(BATCH_SIZE).to(device)
        self.alpha = alpha
        self.vector = vector

    def __call__(self, hid_ref, hid_cur, target):
        cos_loss = self.loss(hid_ref, hid_cur, target)
        if self.vector is not None:
            cos_loss += self.alpha * self.loss(self.vector, hid_ref - hid_cur,
                                               self.target)
        return cos_loss


class ProjectionLoss:
    def __init__(self, *, subspace_matrix, weight_cosine: int = 1, weight_projection: int = 1):
        '''
        Minimizes distance from vector to its projection while maximizing the cosine distance between target and given vectors
        (Overall, separates and orthogonalizes subspace w.r.t. given reference vectors)
        subspace_matrix -- matrix used for projection construction using M@(M.T@M)^{-1}@M.T
        alpha -- weight of cosine loss
        beta -- weight of projection distance minimization
        '''
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss = torch.nn.CosineEmbeddingLoss()

        self.weight_cosine = weight_cosine
        self.weight_projection = weight_projection
        subspace_matrix = torch.tensor(subspace_matrix)
        self.orth_proj = torch.eye(768) - subspace_matrix @ torch.linalg.inv(subspace_matrix.T @ subspace_matrix) @ subspace_matrix.T
        self.orth_proj = self.orth_proj.to(device)        

    def __call__(self, hid_ref, hid_cur, target):
        
        cos_loss = self.weight_cosine * self.loss(hid_ref, hid_cur, target)

        proj_loss = self.weight_projection * torch.linalg.norm(hid_cur@self.orth_proj, ord=2, dim=1).sum()

        total_loss = cos_loss + proj_loss
        
        return total_loss, cos_loss, proj_loss


class EntropyLoss:

    def __init__(self, *, weight_cosine: int = 1, weight_entropy: int = 1, max_dim: int=2, ):

        self.loss = torch.nn.CosineEmbeddingLoss()
        
        self.entropy = entropy_loss

        self.weight_cosine = weight_cosine
        self.weight_entropy = weight_entropy
        self.max_simplex_dim = max_dim

    def __call__(self, hid_ref, hid_cur, target):

        cos_loss = self.weight_cosine * self.loss(hid_ref, hid_cur, target)

        entropy_loss = self.weight_entropy * self.entropy(hid_cur, max_dim=self.max_simplex_dim)

        total_loss = cos_loss + entropy_loss

        return total_loss, cos_loss, entropy_loss


class DiagramDivergenceLoss:

    def __init__(self, *, weight_cosine: int = 1, weight_divergence: int = 1, max_dim: int=2, num_samples=100):

        self.loss = torch.nn.CosineEmbeddingLoss()
        
        self.divergence = diagram_divergence_loss

        self.weight_cosine = weight_cosine
        self.weight_divergence = weight_divergence
        self.max_simplex_dim = max_dim
        self.num_samples = num_samples

    def __call__(self, hid_ref, hid_cur, target):

        cos_loss = self.weight_cosine * self.loss(hid_ref, hid_cur, target)

        divergence_loss = self.weight_divergence * self.divergence(ref=hid_ref, cur=hid_cur, 
                                                                max_dim=self.max_simplex_dim,
                                                                num_samples=self.num_samples 
                                                               )
        total_loss = cos_loss + divergence_loss

        return total_loss, cos_loss, divergence_loss
