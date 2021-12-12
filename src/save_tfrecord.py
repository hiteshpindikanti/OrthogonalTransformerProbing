import argparse

import constants
from data_support.tfrecord_wrapper import TFRecordWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default='../resources/tf_data',
                        help="Directory where tfrecord files are stored")
    parser.add_argument("--model",
                        default=f"bert-{constants.SIZE_BASE}-{constants.LANGUAGE_MULTILINGUAL}-{constants.CASING_CASED}",
                        help="Transformer model name (see: https://huggingface.co/transformers/pretrained_models.html)")

    args = parser.parse_args()

    models = [args.model]
    data_spec = [
        # ('train', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,pos_distance,pos_depth,rnd_distance,rnd_depth',
        #  "/net/data/universal-dependencies-2.6/UD_English-EWT/en_ewt-ud-train.conllu"),
        # ('dev', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,pos_distance,pos_depth,rnd_distance,rnd_depth',
        #  "/net/data/universal-dependencies-2.6/UD_English-EWT/en_ewt-ud-dev.conllu"),
        # ('test', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,pos_distance,pos_depth,rnd_distance,rnd_depth',
        # "/net/data/universal-dependencies-2.6/UD_English-EWT/en_ewt-ud-test.conllu"),
        # ('train', 'es','dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Spanish-AnCora/es_ancora-ud-train.conllu"),
        # ('dev', 'es', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Spanish-AnCora/es_ancora-ud-dev.conllu"),
        # ('test', 'es', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        # "/net/data/universal-dependencies-2.6/UD_Spanish-AnCora/es_ancora-ud-test.conllu"),
        # ('train', 'sl','dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Slovenian-SSJ/sl_ssj-ud-train.conllu"),
        # ('dev', 'sl', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Slovenian-SSJ/sl_ssj-ud-dev.conllu"),
        # ('test', 'sl', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        # "/net/data/universal-dependencies-2.6/UD_Slovenian-SSJ/sl_ssj-ud-test.conllu"),
        # ('train', 'zh','dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Chinese-GSD/zh_gsd-ud-train.conllu"),
        # ('dev', 'zh', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Chinese-GSD/zh_gsd-ud-dev.conllu"),
        # ('test', 'zh', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        # "/net/data/universal-dependencies-2.6/UD_Chinese-GSD/zh_gsd-ud-test.conllu"),
        # ('train', 'id', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Indonesian-GSD/id_gsd-ud-train.conllu"),
        # ('dev', 'id', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        #  "/net/data/universal-dependencies-2.6/UD_Indonesian-GSD/id_gsd-ud-dev.conllu"),
        # ('test', 'id', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
        # "/net/data/universal-dependencies-2.6/UD_Indonesian-GSD/id_gsd-ud-test.conllu")
        # ('train', 'fi', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Finnish-TDT/fi_tdt-ud-train.conllu"),
		# ('dev', 'fi', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Finnish-TDT/fi_tdt-ud-dev.conllu"),
		# ('test', 'fi', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Finnish-TDT/fi_tdt-ud-test.conllu"),
		# ('train', 'ar', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Arabic-PADT/ar_padt-ud-train.conllu"),
		# ('dev', 'ar', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Arabic-PADT/ar_padt-ud-dev.conllu"),
		# ('test', 'ar', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Arabic-PADT/ar_padt-ud-test.conllu"),
		# ('train', 'fr', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_French-GSD/fr_gsd-ud-train.conllu"),
		# ('dev', 'fr', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_French-GSD/fr_gsd-ud-dev.conllu"),
		# ('test', 'fr', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_French-GSD/fr_gsd-ud-test.conllu"),
		# ('train', 'eu', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Basque-BDT/eu_bdt-ud-train.conllu"),
		# ('dev', 'eu', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Basque-BDT/eu_bdt-ud-dev.conllu"),
		# ('test', 'eu', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth',
		# "/net/data/universal-dependencies-2.6/UD_Basque-BDT/eu_bdt-ud-test.conllu")
    ]

    tf_writer = TFRecordWriter(models, data_spec, args.data_dir)
    tf_writer.compute_and_save(args.data_dir)
