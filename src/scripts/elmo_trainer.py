from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict

import torch
from torch.nn import LSTM
import torch.nn.functional as F
from torch.nn import Dropout, CrossEntropyLoss

from allennlp.models import Model
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_text_field_mask, masked_softmax, masked_log_softmax

from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    ELMoTokenCharactersIndexer
)
from allennlp.training.util import evaluate

import logging
import argparse
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class EIDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, feature_dictionary: Dict[str, List] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.feature_dictionary = feature_dictionary

    def text_to_instance(self, article_text: List[str], label: str, outcome: List[str], intervention: List[str],
                         comparator: List[str]):
        fields = {
            'article': TextField([Token(x) for x in article_text], self.token_indexers),
            'outcome': TextField([Token(x) for x in article_text], self.token_indexers),
            'intervention': TextField([Token(x) for x in article_text], self.token_indexers),
            'comparator': TextField([Token(x) for x in article_text], self.token_indexers),
            'labels': LabelField(label)
        }
        return Instance(fields)

    def _read(self, dataset) -> Iterator[Instance]:
        for pmcid in dataset:
            if pmcid in self.feature_dictionary:
                for sample in self.feature_dictionary[pmcid]:
                    if not isinstance(sample[0], str):
                        continue
                    yield self.text_to_instance(sample[0].lower().split(), sample[1], sample[2].lower().split(),
                                            sample[3].lower().split(), sample[4].lower().split())


class Baseline(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        self.text_seq_encoder = PytorchSeq2VecWrapper(LSTM(word_embeddings.get_output_dim(),
                                                      int(word_embeddings.get_output_dim()/2),
                                                      batch_first=True,
                                                      bidirectional=True))

        self.out = torch.nn.Linear(
            in_features=self.text_seq_encoder.get_output_dim()*4,
            out_features=vocab.get_vocab_size('labels')
        )
        self.accuracy = CategoricalAccuracy()
        self.f_score_0 = F1Measure(positive_label=0)
        self.f_score_1 = F1Measure(positive_label=1)
        self.f_score_2 = F1Measure(positive_label=2)
        self.loss = CrossEntropyLoss()

    def forward(self,
                article: Dict[str, torch.Tensor],
                outcome: Dict[str, torch.Tensor],
                intervention: Dict[str, torch.Tensor],
                comparator: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        a_mask = get_text_field_mask(article)
        a_embeddings = self.word_embeddings(article)
        a_vec = self.text_seq_encoder(a_embeddings, a_mask)

        o_mask = get_text_field_mask(outcome)
        o_embeddings = self.word_embeddings(outcome)
        o_vec = self.text_seq_encoder(o_embeddings, o_mask)

        i_mask = get_text_field_mask(intervention)
        i_embeddings = self.word_embeddings(intervention)
        i_vec = self.text_seq_encoder(i_embeddings, i_mask)

        c_mask = get_text_field_mask(comparator)
        c_embeddings = self.word_embeddings(comparator)
        c_vec = self.text_seq_encoder(c_embeddings, c_mask)

        logits = self.out(torch.cat((a_vec, o_vec, i_vec, c_vec), dim=1))

        output = {'logits': logits}

        if labels is not None:
            self.accuracy(logits, labels)
            self.f_score_0(logits, labels)
            self.f_score_1(logits, labels)
            self.f_score_2(logits, labels)
            output['loss'] = self.loss(logits, labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        _, _, f_score0 = self.f_score_0.get_metric(reset)
        _, _, f_score1 = self.f_score_1.get_metric(reset)
        _, _, f_score2 = self.f_score_2.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'f-score': np.mean([f_score0, f_score1, f_score2])}


def main():
    parser = argparse.ArgumentParser(description='Evidence Inference experiments')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='GPU number (default: 0)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit (default: 2)')
    parser.add_argument('--patience', type=int, default=1,
                        help='trainer patience  (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout for the model (default: 0.2)')
    parser.add_argument('--emb_size', type=int, default=256,
                        help='elmo embeddings size (default: 256)')
    parser.add_argument('--model_name', type=str, default='baseline',
                        help='model name (default: baseline)')
    args = parser.parse_args()

    annotations = pd.read_csv('data/data/annotations_merged.csv')
    prompts = pd.read_csv('data/data/prompts_merged.csv')

    feature_dictionary = {}
    prompts_dictionary = {}

    for index, row in prompts.iterrows():
        prompts_dictionary[row['PromptID']] = [row['Outcome'], row['Intervention'], row['Comparator']]

    for index, row in annotations.iterrows():
        if row['PMCID'] not in feature_dictionary:
            feature_dictionary[row['PMCID']] = []
        feature_dictionary[row['PMCID']].append([row['Annotations'], row['Label']]
                                                + prompts_dictionary[row['PromptID']])

    train = []
    valid = []
    test = []

    with open('data/splits/train_article_ids.txt') as train_file:
        for line in train_file:
            train.append(int(line.strip()))

    with open('data/splits/validation_article_ids.txt') as valid_file:
        for line in valid_file:
            valid.append(int(line.strip()))

    with open('data/splits/test_article_ids.txt') as test_file:
        for line in test_file:
            test.append(int(line.strip()))

    elmo_token_indexer = {'elmo': ELMoTokenCharactersIndexer(), 'tokens': SingleIdTokenIndexer()}

    reader = EIDatasetReader(elmo_token_indexer, feature_dictionary)
    train_data = reader.read(train)
    valid_data = reader.read(valid)
    test_data = reader.read(test)

    vocab = Vocabulary.from_instances(train_data + valid_data + test_data)

    urls = [
        'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_'
        '2xhighway_options.json',
        'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_'
        '2xhighway_weights.hdf5'
    ]

    elmo_token_embedding = ElmoTokenEmbedder(urls[0], urls[1], dropout=args.dropout, requires_grad=False,
                                             projection_dim=args.emb_size)

    word_embeddings = BasicTextFieldEmbedder({'elmo': elmo_token_embedding}, allow_unmatched_keys=True)

    model = Baseline(word_embeddings, vocab)

    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[('article', 'num_tokens')],
                              padding_noise=0.1)
    iterator.index_with(vocab)

    serialization_dir = 'model_checkpoints/' + args.model_name

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=test_data,
                      patience=args.patience,
                      validation_metric='+accuracy',
                      num_epochs=args.epochs,
                      cuda_device=cuda_device,
                      serialization_dir=serialization_dir)

    result = trainer.train()
    for key in result:
        print(str(key) + ': ' + str(result[key]))

    test_metrics = evaluate(trainer.model, test_data, iterator,
                            cuda_device=cuda_device,
                            batch_weight_key="")

    print('Test Data statistics:')
    for key, value in test_metrics.items():
        print(str(key) + ': ' + str(value))


if __name__ == '__main__':
    main()

