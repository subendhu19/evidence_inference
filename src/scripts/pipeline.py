from allennlp.data.dataset_readers import DatasetReader
import pickle

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm

from allennlp.data.vocabulary import Vocabulary

from typing import Iterator, List, Dict

import torch
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.modules.token_embedders import (
    PretrainedBertEmbedder
)
from allennlp.data.iterators import BasicIterator

from allennlp.data.token_indexers import (
    PretrainedBertIndexer
)

from allennlp.training.util import evaluate

import logging
import argparse
import sys
import os

# Path hack
sys.path.insert(0, os.path.abspath('./'))

from src.scripts.classifier import Classifier
from src.scripts.oracle import Oracle, EIDatasetReader

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


def main():
    parser = argparse.ArgumentParser(description='Evidence sentence classifier')
    parser.add_argument('--k', type=int, default=3,
                        help='number of evidence sentences to pick from the classifier (default: 3)')
    parser.add_argument('--probs', type=str, default=None,
                        help='Pickled sentence probs file (default: None)')
    args = parser.parse_args()

    with torch.no_grad():
        bert_token_indexer = {'bert': PretrainedBertIndexer('scibert/vocab.txt', max_pieces=512)}

        # pipeline_train = pickle.load(open('data/train_instances.p', 'rb'))
        pipeline_val = pickle.load(open('data/val_instances.p', 'rb'))
        pipeline_test = pickle.load(open('data/test_instances.p', 'rb'))

        pipeline_reader = PipelineDatasetReader(bert_token_indexer)
        # p_train = pipeline_reader.read(pipeline_train)
        p_val = pipeline_reader.read(pipeline_val)
        p_test = pipeline_reader.read(pipeline_test)

        p_vocab = Vocabulary.from_instances(p_val + p_test)

        bert_token_embedding = PretrainedBertEmbedder(
            'scibert/weights.tar.gz', requires_grad=False
        )

        word_embeddings = BasicTextFieldEmbedder(
            {"bert": bert_token_embedding},
            {"bert": ['bert']},
            allow_unmatched_keys=True
        )

        ev_classifier = Classifier(word_embeddings=word_embeddings,
                                   vocab=p_vocab,
                                   loss='bce',
                                   hinge_margin=0)
        predictor = Oracle(word_embeddings=word_embeddings,
                           vocab=p_vocab)

        cuda_device = 0

        if torch.cuda.is_available():
            ev_classifier = ev_classifier.cuda()
            predictor = predictor.cuda()
        else:
            cuda_device = -1

        ev_classifier.load_state_dict(torch.load('model_checkpoints/f_evidence_sentence_classifier/best.th'))
        predictor.load_state_dict(torch.load('model_checkpoints/f_oracle_full/best.th'))

        logger.info('Classifier and Predictor models loaded successfully')
        ev_classifier.eval()
        predictor.eval()

        iterator = BasicIterator(batch_size=256)
        iterator.index_with(p_vocab)

        if args.probs is None:
            iterator_obj = iterator(p_test, num_epochs=1, shuffle=False)
            generator_tqdm = Tqdm.tqdm(iterator_obj, total=iterator.get_num_batches(p_test))

            output_probs = []
            for batch in generator_tqdm:
                batch = nn_util.move_to_device(batch, cuda_device)
                probs = ev_classifier.predict_evidence_probs(**batch)
                probs = probs.numpy()
                output_probs.append(probs)

            output_probs = [i for item in output_probs for i in item]
            logger.info('Obtained all sentence evidence probabilities - total {}'.format(len(output_probs)))
            pickle.dump(output_probs, open('sentence_ev_probs.p', 'wb'))

        else:
            output_probs = pickle.load(open(args.probs, 'rb'))

        top_k_sentences = []
        prob_counter = 0
        for i in range(len(pipeline_test)):
            sentences = [' '.join(s[0]) for s in pipeline_test[i]['sentence_span']]
            probs = list(output_probs[prob_counter: prob_counter + len(sentences)])
            prob_counter += len(sentences)
            sorted_sentences = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)
            top_k = [s[0] for s in sorted_sentences[:args.k]]
            top_k_sentences.append({'I': pipeline_test[i]['I'],
                                    'C': pipeline_test[i]['C'],
                                    'O': pipeline_test[i]['O'],
                                    'y_label': pipeline_test[i]['y'][0][0],
                                    'evidence': ' '.join(top_k)})

        logger.info('Obtained the top sentences from the evidence classifier')

        predictor_reader = EIDatasetReader(bert_token_indexer)
        predictor_test = predictor_reader.read(top_k_sentences)

        test_metrics = evaluate(predictor, predictor_test, iterator,
                                cuda_device=cuda_device,
                                batch_weight_key="")

        print('Test Data statistics:')
        for key, value in test_metrics.items():
            print(str(key) + ': ' + str(value))


if __name__ == '__main__':
    main()
