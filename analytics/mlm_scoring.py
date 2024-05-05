import minicons
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass(slots=True)
class ValueCounter:
    
        values: int = field(default=0)
        count: int = field(default=0)


def ppl_mlm_score(model, dataloader, *, top_k: int, PPL_metric: str="within_word_l2r") -> (list[float], list[float], dict, dict):
    '''
    Model -- MLM trained model to test perplexity
    Dataloader -- used for batching, nothing else. Should return batched sentences from dataset, not their indices.
    top_k: how many tokens to add to ppl dict
    PPL_metric: "original" or "within_word_l2r"
    '''
    def score(batch: list, scores: list, tkn_dict: dict):

        score = model.token_score(batch, PLL_metric=PPL_metric)
        scores.extend([sum(n for _, n, *_ in sc)/len(sc) for sc in score])

        top_tkn_scores = [sorted(sc, key=operator.itemgetter(1))[:top_k] for sc in score]
        
        for seq in top_tkn_scores:
            for item, value in seq:
                tkn_dict[item].values += value
                tkn_dict[item].count += 1

        return scores, tkn_dict

    def sort(tkn_dict):
        return sorted(tkn_dict.items(), key=lambda x: x[1].values/x[1].count)
        
    try:
        
        base_scores = []
        poly_scores = []
        tkn_dict_base = defaultdict(ValueCounter)
        tkn_dict_poly = defaultdict(ValueCounter)
        
        for batch in tqdm(dataloader):
    
            base_scores, tkn_dict_base = score(
                batch["base"], scores=base_scores, tkn_dict=tkn_dict_base
            )
            poly_scores, tkn_dict_poly = score(
                batch["polypers"], scores=poly_scores, tkn_dict=tkn_dict_poly
            )
    except KeyboardInterrupt:
        pass
    return base_scores, poly_scores, sort(tkn_dict_base), sort(tkn_dict_poly)
    