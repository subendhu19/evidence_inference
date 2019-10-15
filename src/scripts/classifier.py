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
# from transformers import AdamW, WarmupLinearSchedule
from pytorch_pretrained_bert import BertAdam

import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class EvidenceDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.num_duplicate = 5

    def text_to_instance(self, prompt: List[List[str]], evidence: List[str], non_evidence: List[str]):

        fields = {
            'I': TextField([Token(x) for x in prompt[0]], self.token_indexers),
            'C': TextField([Token(x) for x in prompt[1]], self.token_indexers),
            'O': TextField([Token(x) for x in prompt[2]], self.token_indexers),
            'evidence': TextField([Token(x) for x in evidence], self.token_indexers),
            'non_evidence': TextField([Token(x) for x in non_evidence], self.token_indexers),
            'comb_evidence': TextField([Token(x) for x in (['[CLS]'] + prompt[0] + prompt[1] + prompt[2] +
                                                           ['[SEP]'] + evidence)], self.token_indexers),
            'comb_non_evidence': TextField([Token(x) for x in (['[CLS]'] + prompt[0] + prompt[1] + prompt[2] +
                                                               ['[SEP]'] + non_evidence)], self.token_indexers)
        }
        return Instance(fields)

    def _read(self, dataset) -> Iterator[Instance]:
        for item in dataset:
            if len(item[0][0]) > 0 and len(item[0][1]) > 0 and len(item[0][2]) > 0:
                for positive in item[1]:
                    p = positive.lower().split()
                    if len(p) > 0:
                        for i in range(self.num_duplicate):
                            if len(item[2]) > 0:
                                neg1 = np.random.choice(item[2]).lower().split()
                                if len(neg1) > 0:
                                    yield self.text_to_instance(list(item[0]), p, neg1)
                            if len(item[3]) > 0:
                                neg2 = np.random.choice(item[3]).lower().split()
                                if len(neg2) > 0:
                                    yield self.text_to_instance(list(item[0]), p, neg2)
                            if len(item[4]) > 0:
                                neg3 = np.random.choice(item[4]).lower().split()
                                if len(neg3) > 0:
                                    yield self.text_to_instance(list(item[0]), p, neg3)


class Classifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 loss: str,
                 hinge_margin: float) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        self.out = torch.nn.Linear(
            in_features=word_embeddings.get_output_dim(),
            out_features=1
        )
        self.accuracy = BooleanAccuracy()
        self.loss_name = loss
        if loss == 'hinge':
            self.loss = MarginRankingLoss(margin=hinge_margin, reduction='mean')
        else:
            self.loss = BCEWithLogitsLoss(reduction='mean')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,
                I: Dict[str, torch.Tensor],
                C: Dict[str, torch.Tensor],
                O: Dict[str, torch.Tensor],
                evidence: Dict[str, torch.Tensor],
                non_evidence: Dict[str, torch.Tensor],
                comb_evidence: Dict[str, torch.Tensor],
                comb_non_evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        ev_embeddings = self.word_embeddings(comb_evidence)
        ev_vec = ev_embeddings[:, 0, :]

        nev_embeddings = self.word_embeddings(comb_non_evidence)
        nev_vec = nev_embeddings[:, 0, :]

        p_labels = torch.ones(size=(ev_vec.size(0), 1), device=ev_vec.device, requires_grad=False)
        n_labels = torch.zeros(size=(ev_vec.size(0), 1), device=ev_vec.device, requires_grad=False)
        all_labels = torch.cat((p_labels, n_labels), dim=0)

        if self.loss_name == 'hinge':
            ev_probs = self.sigmoid(self.out(ev_vec))
            nev_probs = self.sigmoid(self.out(nev_vec))

            all_probs = torch.cat((ev_probs, nev_probs), dim=0)
            all_preds = (all_probs > 0.5).float()

            output = {'logits': all_probs}

            self.accuracy(all_preds.squeeze(), all_labels.squeeze())

            output['loss'] = self.loss(ev_probs, nev_probs, p_labels)
        else:
            ev_logits = self.out(ev_vec)
            nev_logits = self.out(nev_vec)

            all_logits = torch.cat((ev_logits, nev_logits), dim=0)
            all_preds = (self.sigmoid(all_logits) > 0.5).float()

            output = {'logits': all_logits}

            self.accuracy(all_preds.squeeze(), all_labels.squeeze())

            output['loss'] = self.loss(all_logits, all_labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}

    def predict_evidence_probs(self, comb_spans: Dict[str, torch.Tensor]) -> torch.Tensor:
        span_embeddings = self.word_embeddings(comb_spans)
        span_vec = span_embeddings[:, 0, :]
        span_probs = self.sigmoid(self.out(span_vec))

        return span_probs


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

    if args.loss not in ['bce', 'hinge']:
        print('Loss must be bce or hinge')
        return

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

    model = Classifier(word_embeddings=word_embeddings,
                       vocab=vocab,
                       loss=args.loss,
                       hinge_margin=args.hinge_margin)

    cuda_device = list(range(torch.cuda.device_count()))

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        cuda_device = -1

    t_total = len(train_data) // args.epochs

    optimizer = BertAdam(model.parameters(), lr=2e-5, warmup=0.1, t_total=t_total)

    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[('comb_evidence', 'num_tokens')],
                              padding_noise=0.1,
                              biggest_batch_first=True)
    iterator.index_with(vocab)

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
                      # learning_rate_scheduler=scheduler,
                      serialization_dir=serialization_dir)

    result = trainer.train()
    for key in result:
        print(str(key) + ': ' + str(result[key]))

    # test_metrics = evaluate(trainer.model, valid_data, iterator,
    #                         cuda_device=cuda_device,
    #                         batch_weight_key="")
    #
    # print('Test Data statistics:')
    # for key, value in test_metrics.items():
    #     print(str(key) + ': ' + str(value))


if __name__ == '__main__':
    main()
