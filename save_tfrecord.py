import argparse

import constants
from tfrecord_wrapper import TFRecordWriter

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--casing", default=constants.CASING_CASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")
	args = parser.parse_args()

	args.bert_path = "bert-{}-{}-{}".format(args.size, args.language, args.casing)
	models = [args.bert_path]
	languages = ['en']
	data_spec = [('train', 'en', 'resources/endev.conllu'),
	             ('dev', 'en', 'resources/endev.conllu'),
	             ('test', 'en', 'resources/endev.conllu')]

	tasks = ['dep-distance', 'dep-depth']

	tf_writer = TFRecordWriter(tasks, models, data_spec)
	tf_writer.compute_and_save('resources/tf_data')
