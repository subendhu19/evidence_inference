from allennlp.data.dataset_readers import DatasetReader

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict

import torch
from torch.nn import LSTM
from torch.nn import CrossEntropyLoss

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from allennlp.modules.token_embedders import (
    PretrainedBertEmbedder
)
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from pytorch_pretrained_bert import BertAdam
from allennlp.training.util import evaluate

from allennlp.data.token_indexers import (
    PretrainedBertIndexer
)

import logging
import argparse
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class PipelineDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers

    def text_to_instance(self, prompt: List[List[str]], evidence: List[str], label: str):

        fields = {
            'comb_sentences': TextField([Token(x) for x in (['[CLS]'] + prompt[0] + prompt[1] + prompt[2] +
                                                            ['[SEP]'] + evidence)], self.token_indexers),
            'labels': LabelField(label)
        }
        return Instance(fields)

    def _read(self, dataset) -> Iterator[Instance]:
        for item in dataset:
            for s in item['sentence_span']:
                yield self.text_to_instance([item['I'], item['C'], item['O']], [w.lower() for w in s[0]],
                                            str(item['y'][0][0]))


class EIDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers

    def text_to_instance(self, intervention: List[str], comparator: List[str], outcome: List[str],
                         evidence: List[str], label: str):

        fields = {
            'comb_prompt_ev': TextField([Token(x) for x in (['[CLS]'] + intervention + comparator + outcome +
                                                            ['[SEP]'] + evidence)], self.token_indexers),
            'labels': LabelField(label)
        }
        return Instance(fields)

    def _read(self, dataset) -> Iterator[Instance]:
        for instance in dataset:
            yield self.text_to_instance(instance['I'], instance['C'], instance['O'],
                                        instance['evidence'].lower().split(), str(instance['y_label']))


class Oracle(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        self.out = torch.nn.Linear(
            in_features=word_embeddings.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )
        self.accuracy = CategoricalAccuracy()
        self.f_score_0 = F1Measure(positive_label=0)
        self.f_score_1 = F1Measure(positive_label=1)
        self.f_score_2 = F1Measure(positive_label=2)
        self.loss = CrossEntropyLoss()

    def forward(self,
                comb_prompt_ev: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        pev_embeddings = self.word_embeddings(comb_prompt_ev)
        pev_vec = pev_embeddings[:, 0, :]

        logits = self.out(pev_vec)

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
    parser = argparse.ArgumentParser(description='Evidence oracle QA')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit (default: 5)')
    parser.add_argument('--patience', type=int, default=1,
                        help='trainer patience  (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--model_name', type=str, default='sentence_oracle_bert',
                        help='model name (default: sentence_oracle_bert)')
    parser.add_argument('--tunable', action='store_true',
                        help='tune the underlying embedding model (default: False)')
    parser.add_argument('--ev_type', type=str, default='sentence',
                        help='how to train the oracle - sentence or full (evidence) (default: sentence)')
    args = parser.parse_args()

    if args.ev_type == 'sentence':
        train = pickle.load(open('data/oracle_train.p', 'rb'))
        valid = pickle.load(open('data/oracle_val.p', 'rb'))
        test = pickle.load(open('data/oracle_test.p', 'rb'))
    elif args.ev_type == 'full':
        train = pickle.load(open('data/oracle_full_train.p', 'rb'))
        valid = pickle.load(open('data/oracle_full_val.p', 'rb'))
        test = pickle.load(open('data/oracle_full_test.p', 'rb'))
    else:
        print('ev_type should be either sentence or full')
        return

    bert_token_indexer = {'bert': PretrainedBertIndexer('scibert/vocab.txt', max_pieces=512)}

    pipeline_train = pickle.load(open('data/train_instances.p', 'rb'))
    pipeline_val = pickle.load(open('data/val_instances.p', 'rb'))
    pipeline_test = pickle.load(open('data/test_instances.p', 'rb'))

    pipeline_reader = PipelineDatasetReader(bert_token_indexer)
    p_train = pipeline_reader.read(pipeline_train)
    p_val = pipeline_reader.read(pipeline_val)
    p_test = pipeline_reader.read(pipeline_test)

    p_vocab = Vocabulary.from_instances(p_train + p_val + p_test)

    reader = EIDatasetReader(bert_token_indexer)
    train_data = reader.read(train)
    valid_data = reader.read(valid)
    test_data = reader.read(test)

    bert_token_embedding = PretrainedBertEmbedder(
        'scibert/weights.tar.gz', requires_grad=args.tunable
    )

    word_embeddings = BasicTextFieldEmbedder(
        {"bert": bert_token_embedding},
        {"bert": ['bert']},
        allow_unmatched_keys=True
    )

    model = Oracle(word_embeddings, p_vocab)

    cuda_device = list(range(torch.cuda.device_count()))

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        cuda_device = -1

    t_total = len(train_data) // args.epochs

    optimizer = BertAdam(model.parameters(), lr=2e-5, warmup=0.1, t_total=t_total)

    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[('comb_prompt_ev', 'num_tokens')],
                              padding_noise=0.1)
    iterator.index_with(p_vocab)

    serialization_dir = 'model_checkpoints/' + args.model_name

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=valid_data,
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

