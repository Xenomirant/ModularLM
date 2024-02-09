from datasets import load_dataset, load_from_disk
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import os
from gensim.models import Word2Vec
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BatchIterator:
    def __init__(self, data, tokenizer=None, bsize=2048):
        self.data = data
        self.tok = tokenizer
        self.bsize = bsize

    def __iter__(self):
        for i in range(0, len(self.data), self.bsize):
            cur = self.data[i:i + self.bsize]["text"]
            if self.tok is not None:
                cur = tok(cur).data['input_ids']
                for line in cur:
                    yield line
            else:
                yield cur


def get_dataset(path='dataset'):
    if path not in os.listdir('.'):
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ru")
        dataset['train'].save_to_disk(path)
    else:
        dataset = load_from_disk(path)
    return dataset


def build_tokenizer():
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens, show_progress=True)
    return tokenizer, trainer


def train_tokenizer(tok, train, data):
    biter = BatchIterator(data, bsize=2048)
    tok.train_from_iterator(biter, trainer=train)
    cls_token_id = tok.token_to_id("[CLS]")
    sep_token_id = tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tok.decoder = decoders.WordPiece(prefix="##")
    new_tokenizer = BertTokenizerFast(tokenizer_object=tok)
    return new_tokenizer


def train_word2vec(data, tok):
    biter = BatchIterator(data, tokenizer=tok, bsize=2048)
    model = Word2Vec(sentences=biter, vector_size=300, window=5, min_count=1, workers=10)
    model.save("word2vec.model")


if __name__ == '__main__':
    data = get_dataset()
    # tok, train = build_tokenizer()
    # new_tok = train_tokenizer(tok, train, data)
    # new_tok.save_pretrained("small-bert-tokenizer")
    tok = BertTokenizerFast.from_pretrained('small-bert-tokenizer')
    # train_word2vec(data, tok)
    w2v = Word2Vec.load("word2vec.model")
    w2v.wv.save("word2vec.wordvectors")
