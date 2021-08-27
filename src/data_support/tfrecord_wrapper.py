import tensorflow as tf
from tqdm import tqdm
import os
import json
from collections import defaultdict, Mapping
from itertools import chain
from copy import deepcopy

from transformers import BertTokenizer, BertConfig, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel

import constants
from data_support.dependency import DependencyDistance, DependencyDepth
from data_support.lexical import LexicalDistance, LexicalDepth
from data_support.random import RandomDistance, RandomDepth
from data_support.positional import PositionalDistance, PositionalDepth
from data_support.shuffled import ShuffledDistance, ShuffledDepth


central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()

conllu_wrappers = {
    "dep_distance": DependencyDistance,
    "dep_depth": DependencyDepth,
    "lex_distance": LexicalDistance,
    "lex_depth": LexicalDepth,
    "rnd_depth": RandomDepth,
    "rnd_distance": RandomDistance,
    "pos_depth": PositionalDepth,
    "pos_distance": PositionalDistance,
    "shf_depth": ShuffledDepth,
    "shf_distance": ShuffledDistance
}


def merge_dict(d1, d2):
    """
    Modifies d1 in-place to contain values from d2.  If any value
    in d1 is a dictionary (or dict-like), *and* the corresponding
    value in d2 is also a dictionary, then merge them in-place.
    """
    for k,v2 in d2.items():
        v1 = d1.get(k) # returns None if v1 has no value for this key
        if ( isinstance(v1, Mapping) and
             isinstance(v2, Mapping) ):
            merge_dict(v1, v2)
        else:
            d1[k] = v2

class TFRecordWrapper:

    modes = ['train', 'dev', 'test']
    data_map_fn = "data_map.json"

    def __init__(self, tasks, models, languages):
        self.tasks = tasks
        self.models = models
        self.languages = list(set(languages))

        self.map_tfrecord = dict()
        for mode in self.modes:
            self.map_tfrecord[mode] = dict()
            for model in models:
                self.map_tfrecord[mode][model] = dict()
                for lang in languages:
                    self.map_tfrecord[mode][model][lang] = dict()
                    for task in tasks:
                        self.map_tfrecord[mode][model][lang][task] = None

        self.map_conll = deepcopy(self.map_tfrecord)

    def _from_json(self, data_dir):
        with open(os.path.join(data_dir,self.data_map_fn),'r') as in_json:
            in_dict = json.load(in_json)
        for attribute, value in in_dict.items():
            if attribute in ("tasks", "models", "languages"):
                present_value = self.__getattribute__(attribute)
                self.__setattr__(attribute, list(set(present_value + value)))
            elif attribute == "map_tfrecord":
                merge_dict(self.map_tfrecord, value)
            elif attribute == "map_conll":
                merge_dict(self.map_conll, value)

    def _to_json(self, data_dir):
        out_dict = {"tasks": self.tasks,
                    "models": self.models,
                    "languages": self.languages,
                    "map_conll": self.map_conll,
                    "map_tfrecord": self.map_tfrecord}

        with open(os.path.join(data_dir,self.data_map_fn), 'w') as out_json:
            json.dump(out_dict, out_json, indent=2, sort_keys=True)


class TFRecordWriter(TFRecordWrapper):

    def __init__(self, models, mode_language_tasks_conll, data_dir):

        languages = [lang for _, lang, _, _ in mode_language_tasks_conll]
        unique_tasks = list(set(chain(*[tasks.split(',') for _, _, tasks, _ in mode_language_tasks_conll])))
        assert {mode for mode, _, _, _ in mode_language_tasks_conll} <= set(self.modes), \
            "Unrecognized dataset mode, use `train`, `dev`, or `test`"

        #map_connl = {mode: {lang: {tasks: conll}} for mode, lang, tasks, conll in mode_language_tasks_conll}

        super().__init__(unique_tasks, models, languages)
        if os.path.isfile(os.path.join(data_dir, self.data_map_fn)):
            self._from_json(data_dir)

        self.model2conll = defaultdict(set)

        self.tfr2tasks = defaultdict(set)
        self.conll2dep_tfr_fn = dict()
        self.conll2der_tfr_fn = dict()
        self.conll2cor_tfr_fn = dict()
        self.conll2lang = dict()

        self.model2tfrs = defaultdict(set)
        self.tfr2tasks = defaultdict(set)
        self.tfr2conll = dict()
        self.tfr2lang = dict()

        for mode, lang, tasks, conll in mode_language_tasks_conll:
            for model in models:
                for task in tasks.split(','):
                    # Data for some tasks can be saved in the same file, e.g. dependency and lexical
                    if task in ['dep_distance', 'dep_depth', 'lex_distance', 'lex_depth', 'rnd_distance', 'rnd_depth',
                                'pos_distance', 'pos_depth']:
                        fn_task = 'dep+lex+rnd+pos'
                    elif task in ['shf_distance', 'shf_depth']:
                        fn_task = 'shf'
                    elif task == 'cor_distance':
                        fn_task = 'cor'
                    else:
                        raise ValueError(f"Unrecognized task: {task}")
                    tfr_fn = self.struct_tfrecord_fn(model, fn_task, lang, conll)
                    # TODO: think about a case where some tfrecord are already saved
                    self.map_tfrecord[mode][model][lang][task] = tfr_fn
                    self.map_conll[mode][model][lang][task] = conll

                    self.model2tfrs[model].add(tfr_fn)
                    self.tfr2tasks[tfr_fn].add(task)
                    self.tfr2conll[tfr_fn] = conll
                    self.tfr2lang[tfr_fn] = lang

    def compute_and_save(self, data_dir):

        for model_path in self.models:
            # This is crude, but should work
            do_lower_case = "uncased" in model_path
            model, tokenizer = self.get_model_tokenizer(model_path, do_lower_case=do_lower_case)
            for tfrecord_file in self.model2tfrs[model_path]:
                if os.path.isfile(os.path.join(data_dir, tfrecord_file)):
                    print(f"File {os.path.join(data_dir, tfrecord_file)} already exists, skipping!")
                    continue

                conll_fn = self.tfr2conll[tfrecord_file]
                lang = self.tfr2lang[tfrecord_file]
                tasks = list(self.tfr2tasks[tfrecord_file])

                in_datasets = [conllu_wrappers[task](conll_fn, tokenizer, lang=lang) for task in tasks]
                all_wordpieces, all_segments, all_token_len = in_datasets[0].training_examples()

                options = tf.io.TFRecordOptions()#compression_type='GZIP')
                with tf.io.TFRecordWriter(os.path.join(data_dir, tfrecord_file), options=options) as tf_writer:
                    for idx, (wordpieces, segments, token_len, target_mask) in \
                            tqdm(enumerate(
                                zip(tf.unstack(all_wordpieces), tf.unstack(all_segments), tf.unstack(all_token_len),
                                    self.generate_target_masks(tasks, in_datasets))), desc="Embedding computation"):
                        embeddings = self.calc_embeddings(model, wordpieces, segments, token_len, constants.MAX_WORDPIECES)
                        train_example = self.serialize_example(idx, embeddings, token_len, target_mask)
                        tf_writer.write(train_example.SerializeToString())
        self._to_json(data_dir)

    @staticmethod
    def struct_tfrecord_fn(model,fn_task,lang,conll_fn):
        conll_base = os.path.basename(conll_fn)
        conll_name = os.path.splitext(conll_base)[0]
        return f"{model}_{lang}_{fn_task}_{conll_name}.tfrecord"

    @staticmethod
    def get_model_tokenizer(model_path, do_lower_case, seed=42):
        if model_path.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
            model = TFBertModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('roberta'):
            tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case, add_prefix_space=True)
            model = TFRobertaModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('jplu/tf-xlm-roberta'):
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
            model = TFXLMRobertaModel.from_pretrained(model_path, output_hidden_states=True, output_attentions=False)
        elif model_path.startswith('random-bert'):
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
            config = BertConfig(seed=seed, output_hidden_states=True, output_attentions=False)
            model = TFBertModel(config)
        else:
            raise ValueError(f"Unknown Transformer name: {model_path}. "
                             f"Please select one of the supported models: {constants.SUPPORTED_MODELS}")
        
        return model, tokenizer

    @staticmethod
    def calc_embeddings(model, wordpieces, segments, token_len, max_wordpieces):
        wordpieces = tf.expand_dims(wordpieces, 0)
        segments = tf.expand_dims(segments, 0)
        max_token_len = tf.constant(token_len, shape=(1,), dtype=tf.int64)

        model_output = model(wordpieces, attention_mask=tf.sign(wordpieces), training=False)
        embeddings = model_output.hidden_states[1:]

        # average wordpieces to obtain word representation
        # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same
        embeddings = [tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                                (emb, segments, max_token_len), dtype=tf.float32) for emb in embeddings]
        embeddings = [tf.pad(tf.squeeze(emb, axis=[0]), [[0, max_wordpieces - token_len], [0, 0]]) for emb in embeddings]
        return embeddings

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def serialize_example(idx, embeddings, token_len, task_target_mask):
        feature = {'index': TFRecordWriter._int64_feature(idx),
                   'num_tokens': TFRecordWriter._int64_feature(token_len)}
        feature.update({f'layer_{idx}': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(layer_embeddings))
                        for idx, layer_embeddings in enumerate(embeddings)})

        for task, (target, mask) in task_target_mask.items():
            feature.update({f'target_{task}': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(target)),
                            f'mask_{task}': TFRecordWriter._bytes_feature(tf.io.serialize_tensor(mask))})

        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def generate_target_masks(tasks, in_datasets):
        """ This is basiclly ziping many generators into one, maybe there is simpler solution
        for that."""
        in_generators = zip(*(ds.target_and_mask() for ds in in_datasets))
        for target_mask in in_generators:
            yield {task: (target, mask) for task, (target, mask) in zip(tasks, target_mask)}


class TFRecordReader(TFRecordWrapper):

    def __init__(self, data_dir, model_name):
        super().__init__([], [], [])
        self.data_dir = data_dir
        self.model_name = model_name
        self._from_json(data_dir)
        TFRecordReader.parse_factory(self.tasks, self.model_name)

    def read(self, read_tasks, read_languages):
        if self.model_name not in self.models:
            raise ValueError(f"Data for this model are not available in the directory: {self.model_name}\n"
                             f" supported models: {self.models}")

        for mode in self.modes:
            data_set = dict()
            for langs in read_languages:
                for lang in langs.split('+'):
                    if lang not in self.languages:
                        raise ValueError(f"Data for this language is not available in the directory: {lang}\n"
                                        f" supported languages: {self.languages}")
                    data_set[lang] = dict()
                    for task in read_tasks:
                        if task not in self.tasks:
                            raise ValueError(f"Data for this task is not available in the directory: {task}\n"
                                         f" supported tasks: {self.tasks}")
                        tfr_fn = os.path.join(self.data_dir, self.map_tfrecord[mode][self.model_name][lang][task])
                        data_set[lang][task] = tf.data.TFRecordDataset(tfr_fn,
                                                                   #compression_type='GZIP',
                                                                   buffer_size=constants.BUFFER_SIZE)

            self.__setattr__(mode, data_set)

    @staticmethod
    def parse(example):
        pass

    @classmethod
    def parse_factory(cls, tasks, model_name):

        def parse(example):
            features_dict = {"num_tokens": tf.io.FixedLenFeature([], tf.int64),
                             "index": tf.io.FixedLenFeature([], tf.int64)}
            features_dict.update({f"layer_{idx}": tf.io.FixedLenFeature([], tf.string)
                                  for idx in range(constants.MODEL_LAYERS[model_name])})
            for task in tasks:
                features_dict.update(
                    {f'target_{task}': tf.io.FixedLenFeature([], tf.string),
                     f'mask_{task}': tf.io.FixedLenFeature([], tf.string)})

            example = tf.io.parse_single_example(example, features_dict)

            return example

        cls.parse = parse

