from allennlp.data.dataset_readers import DatasetReader
import pickle

from allennlp.data import Instance
from allennlp.data.fields import TextField

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict

import torch
from torch.nn import LSTM
from torch.nn import MarginRankingLoss, BCEWithLogitsLoss

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.metrics import BooleanAccuracy

from allennlp.modules.token_embedders import (
    PretrainedBertEmbedder
)
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.data.token_indexers import (
    PretrainedBertIndexer
)
# from allennlp.training.util import evaluate
from pytorch_pretrained_bert import BertAdam
from src.scripts.classifier import Classifier, EvidenceDatasetReader

import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class PipelineDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers

    def text_to_instance(self, prompt: List[List[str]], evidence: List[str], non_evidence: List[str]):

        fields = {
            'comb_evidence': TextField([Token(x) for x in (['[CLS]'] + prompt[0] + prompt[1] + prompt[2] +
                                                           ['[SEP]'] + evidence)], self.token_indexers),
            'comb_non_evidence': TextField([Token(x) for x in (['[CLS]'] + prompt[0] + prompt[1] + prompt[2] +
                                                               ['[SEP]'] + non_evidence)], self.token_indexers)
        }
        return Instance(fields)

    def _read(self, dataset) -> Iterator[Instance]:
        for item in dataset:
            yield None


def main():
    parser = argparse.ArgumentParser(description='Evidence sentence classifier')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit (default: 5)')
    parser.add_argument('--patience', type=int, default=1,
                        help='trainer patience  (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--loss', type=str, default='hinge',
                        help='loss function to train the model - choose bce or hinge (default: hinge)')
    parser.add_argument('--hinge_margin', type=float, default=0.5,
                        help='the margin for the hinge loss, if used (default: 0.5)')
    parser.add_argument('--model_name', type=str, default='ev_classifier_bert',
                        help='model name (default: ev_classifier_bert)')
    parser.add_argument('--tunable', action='store_true',
                        help='tune the underlying embedding model (default: False)')
    args = parser.parse_args()

    classifier_train = pickle.load(open('data/classifier_train.p', 'rb'))
    classifier_val = pickle.load(open('data/classifier_val.p', 'rb'))

    bert_token_indexer = {'bert': PretrainedBertIndexer('scibert/vocab.txt', max_pieces=512)}

    reader = EvidenceDatasetReader(bert_token_indexer)
    train_data = reader.read(classifier_train)
    valid_data = reader.read(classifier_val)

    vocab = Vocabulary.from_instances(train_data + valid_data)

    bert_token_embedding = PretrainedBertEmbedder(
        'scibert/weights.tar.gz', requires_grad=args.tunable
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"bert": bert_token_embedding},
        {"bert": ['bert']},
        allow_unmatched_keys=True
    )

    ev_classifier = Classifier(word_embeddings=word_embeddings,
                               vocab=vocab,
                               loss='bce',
                               hinge_margin=0)

    ev_classifier.load_state_dict(torch.load('model_checkpoints/f_oracle_sentence/best.th'))
    print('Classifier model loaded successfully')


if __name__ == '__main__':
    main()
