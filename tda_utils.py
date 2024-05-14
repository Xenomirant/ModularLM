import numpy as np
import gudhi as gd
import gudhi.representations
import torch
import ot
from collections import namedtuple
from ripser import ripser
from functools import wraps
import warnings
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
import copy
from typing import List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

Stats = namedtuple('Stats', ["entropy", "mean", "std"])


class TwoHeadedBert(BertPreTrainedModel):

    def __init__(self, model):
        super().__init__(model.config)
        self.bert = model.bert
        self.first_head = model.cls
        self.second_head = copy.deepcopy(model.cls)
        self.second_head.predictions.decoder.weight = self.first_head.predictions.decoder.weight
        
        self.use_return_dict = model.config.use_return_dict

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head_type: Optional[str] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        head_type: ["first", "second"]
            The MLM classifier head to use for output
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        head_type = head_type if head_type is not None else "first"
        
        sequence_output = outputs[0]
        if head_type == "first":
            prediction_scores = self.first_head(sequence_output)
        elif head_type == "second":
            prediction_scores = self.second_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    vr = gd.RipsComplex(points=points).create_simplex_tree(max_dimension=max_dim)
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
    
    vr_ref = gd.RipsComplex(points=ref).create_simplex_tree(max_dimension=max_dim)
    vr_ref.compute_persistence()
    vr_cur = gd.RipsComplex(points=cur).create_simplex_tree(max_dimension=max_dim)
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
    