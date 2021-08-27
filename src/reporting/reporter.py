import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from scipy import sparse
from collections import defaultdict

from network import Network

from reporting.metrics import UAS, Spearman


class Reporter():

	def __init__(self, args, network, dataset, dataset_name):
		self.network = network
		self.dataset = dataset
		self.dataset_name = dataset_name

		self._probe_threshold = args.probe_threshold
		self._drop_parts = args.drop_parts

	def get_embedding_gate(self, task, part_to_drop=0):

		if 'distance' in task:
			diagonal_probe = self.network.distance_probe.DistanceProbe[task].numpy()
		elif 'depth' in task:
			diagonal_probe = self.network.depth_probe.DepthProbe[task].numpy()
		embedding_gate = (np.abs(diagonal_probe) > self._probe_threshold).astype(np.float)

		if self._drop_parts:
			dim_num = np.sum(embedding_gate)
			part_start = int(dim_num * part_to_drop / self._drop_parts)
			part_end = int(dim_num * (part_to_drop+1) / self._drop_parts)
			dims_to_drop = np.where(embedding_gate)[-1][part_start:part_end]
			embedding_gate[...,dims_to_drop] = 0.

		return tf.constant(embedding_gate, dtype=tf.float32)

	def predict(self, args, language, lang, task):
		data_pipe = Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
		validation_steps = self._drop_parts or 1

		for part_to_drop in range(validation_steps):
			progressbar = tqdm(enumerate(data_pipe), desc="Predicting, {}, {}".format(language, task))
			for batch_idx, (_, _, batch) in progressbar:
				conll_indicies, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

				if self._probe_threshold:
					embedding_gate = self.get_embedding_gate(task, part_to_drop)
				else:
					embedding_gate = None

				if 'distance' in task:
					pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
					                                                           language, task, embedding_gate)
					pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
					               in zip(tf.unstack(pred_values), batch_num_tokens)]
					gold_values = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
					               in zip(tf.unstack(batch_target), batch_num_tokens)]
					mask = [sent_mask.numpy().astype(bool)[:sent_len, :sent_len] for sent_mask, sent_len
					        	in zip(tf.unstack(batch_mask), batch_num_tokens)]
				elif 'depth' in task:
					pred_values = self.network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
					                                                        language, task, embedding_gate)
					pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
					               in zip(tf.unstack(pred_values), batch_num_tokens)]
					gold_values = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
					               in zip(tf.unstack(batch_target), batch_num_tokens)]
					mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
					        	in zip(tf.unstack(batch_mask), batch_num_tokens)]
				else:
					raise ValueError("Unrecognized task, need to contain either `distance` or `depth` in name.")
				yield conll_indicies, batch_num_tokens, pred_values, gold_values, mask


class CorrelationReporter(Reporter):

	def __init__(self, args, network, tasks, dataset, dataset_name):
		super().__init__(args, network, dataset, dataset_name)

		self._languages = args.languages
		self._tasks = tasks
		self.spearman_d = defaultdict(dict)

	def write(self, args):
		for language in self._languages:
			for lang in language.split('+'):                	
				for task in self._tasks:
					prefix = '{}.{}.{}.'.format(self.dataset_name, lang, task)

					if self._probe_threshold:
						prefix += 'gated.'
						if self._drop_parts:
							prefix += 'dp{}.'.format(self._drop_parts)

					with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
						for sent_l, val in self.spearman_d[lang][task].result().items():
							sperarman_f.write(f'{sent_l}\t{val}\n')

					with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
						result = str(np.nanmean(np.fromiter(self.spearman_d[lang][task].result().values(), dtype=float)))
						sperarman_mean_f.write(result + '\n')

	def compute(self, args):

		for language in self._languages:
			for lang in language.split('+'):
				for task in self._tasks:
					self.spearman_d[lang][task] = Spearman()
					for _, _, pred_values, gold_values, mask in self.predict(args, language, lang, task):
						self.spearman_d[lang][task](gold_values, pred_values, mask)


class UASReporter(Reporter):
	def __init__(self, args, network, dataset, dataset_name, conll_dict, depths=None):
		super().__init__(args, network, dataset, dataset_name)
		self.punctuation_masks = {lang: conll_data.punctuation_mask for lang, conll_data in conll_dict.items()}
		self.uu_rels = {lang: conll_data.filtered_relations for lang, conll_data in conll_dict.items()}
		self._languages = args.languages
		self.uas = dict()

		self.depths = depths

	def write(self, args):
		for language in self._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			if self._probe_threshold:
				prefix += 'gated.'
				if self._drop_parts:
					prefix += 'dp{}.'.format(self._drop_parts)
			if self.depths:
				with open(os.path.join(args.out_dir, prefix + 'uas'), 'w') as uas_f:
					uas_f.write(str(self.uas[language].result())+'\n')
			else:
				with open(os.path.join(args.out_dir, prefix + 'uuas'), 'w') as uuas_f:
					uuas_f.write(str(self.uas[language].result())+'\n')

	def compute(self, args):

		for language in self._languages:
			self.uas[language] = UAS()
			for conll_indices, num_tokens, pred_values, gold_values, mask in self.predict(args, language, language, 'dep_distance'):

				for conll_idx, sent_predicted, sent_gold, sent_len in zip(conll_indices.numpy(), pred_values, gold_values, num_tokens):
					sent_punctuation_mask = self.punctuation_masks[language][conll_idx]
					if self.depths:
						predicted_depths = self.depths[language][conll_idx]["predicted"]
						gold_depths = self.depths[language][conll_idx]["gold"]
						predicted_root = np.argmin(predicted_depths) + 1
						gold_root = np.argmin(gold_depths) + 1

					for i in range(sent_len):
						for j in range(sent_len):
							if sent_punctuation_mask[i] or sent_punctuation_mask[j]:
								sent_predicted[i, j] = np.inf
								sent_gold[i, j] = np.inf
							elif self.depths is None:
								if i > j:
									sent_predicted[i, j] = np.inf
									sent_gold[i, j] = np.inf
							else:
								if predicted_depths[i] > predicted_depths[j]:
									sent_predicted[i, j] = np.inf
								if gold_depths[i] > gold_depths[j]:
									sent_gold[i, j] = np.inf

					min_spanning_tree = sparse.csgraph.minimum_spanning_tree(sent_predicted).tocoo()
					min_spanning_tree_gold = sparse.csgraph.minimum_spanning_tree(sent_gold).tocoo()

					predicted = set(map(tuple, zip(min_spanning_tree.col + 1, min_spanning_tree.row + 1)))
					gold = set(map(tuple, zip(min_spanning_tree_gold.col + 1, min_spanning_tree_gold.row + 1)))

					if self.depths:
						predicted.add((predicted_root, 0))
						gold.add((gold_root, 0))

					self.uas[language].update_state(gold, predicted)


class DependencyDepthReporter(Reporter):

	def init(self, args, network, dataset, dataset_name):
		super().__init__(args, network, dataset, dataset_name)

	def compute(self, args):
		results = {lang: dict() for lang in args.languages}
		for language in args.languages:
			for conll_indices, num_tokens, pred_values, gold_values, mask in self.predict(args, language, language, 'dep_depth'):
				for conll_idx, sent_predicted, sent_gold in zip(conll_indices.numpy(), pred_values, gold_values):
					results[language][conll_idx] = {'predicted': sent_predicted, 'gold': sent_gold}

		return results


class SelectedDimensionalityReporter(Reporter):

	def __init__(self, args, network, tasks, dataset, dataset_name):
		super().__init__(args, network, dataset, dataset_name)
		self._languages = args.languages
		self._tasks = tasks

		self.dimension_matrices = dict()

	def write(self, args):
		for language in self._languages:
			for task_idx, task in enumerate(self._tasks):
				prefix = '{}.{}.'.format(language, task)

				with open(os.path.join(args.out_dir, prefix + 'selected_dims'), 'w') as selected_dims_f:
					result = str(self.dimension_matrices[language][task_idx, task_idx])
					selected_dims_f.write(result + '\n')

			prefix = '{}.'.format(language)
			with open(os.path.join(args.out_dir, prefix + 'inter_dims'), 'w') as inter_dims_f:
				inter_dims_f.write(',\t'.join([' '] + self._tasks) + '\n')
				for task, matrix_row in zip(self._tasks, self.dimension_matrices[language]):
					inter_dims_f.write(',\t'.join([task] + list(matrix_row.astype(str))) + '\n')

	def compute(self, args):
		for language in self._languages:
			self.dimension_matrices[language] = np.zeros((len(self._tasks), len(self._tasks)), dtype=np.int)
			selected_dims = dict()
			for task_idx, task in enumerate(self._tasks):
				selected_dims[task_idx] = self.get_embedding_gate(task)
				for task_jdx in range(task_idx+1):
					dims_n = tf.math.reduce_sum(selected_dims[task_idx] * selected_dims[task_jdx]).numpy().astype(int)
					self.dimension_matrices[language][task_idx, task_jdx] = dims_n
					self.dimension_matrices[language][task_jdx, task_idx] = dims_n
