from nltk.corpus import wordnet as wn

# Wordpieces | Tokens limits
MAX_TOKENS = 128
MAX_WORDPIECES = 128

# Conllu indices
CONLLU_ID = 0
CONLLU_ORTH = 1
CONLLU_LEMMA = 2
CONLLU_POS = 3
CONLLU_FEATS = 5
CONLLU_HEAD = 6
CONLLU_LABEL = 7

CONLL_POS = 4
CONLL_COREF = 16
CONLL_NODE = 17

# WordNet constants
# we only consider NOUNs and VERBs, because there is no graph structure for other POS
pos2wnpos = {"NOUN": wn.NOUN,
             "VERB": wn.VERB}
# "ADJ": wn.ADJ,
# "ADV": wn.ADV}

lang2iso = {'en': 'eng', 'es': 'spa', 'fi': 'fin', 'pl': 'pol', 'ar': 'arb',
            'id': 'ind', 'zh': 'cmn', 'fr': 'fra', 'sl': 'slv', 'eu': 'eus'}

# BERT model parameters
LANGUAGE_CHINESE = "chinese"
LANGUAGE_MULTILINGUAL = "multilingual"

SIZE_BASE = "base"
SIZE_LARGE = "large"

CASING_CASED = "cased"
CASING_UNCASED = "uncased"

SUPPORTED_MODELS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}",
                    f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}",
                    f"bert-{SIZE_BASE}-{CASING_CASED}",
                    f"bert-{SIZE_BASE}-{CASING_UNCASED}",
                    f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}",
                    f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}",
                    f"bert-{SIZE_LARGE}-{CASING_CASED}",
                    f"bert-{SIZE_LARGE}-{CASING_UNCASED}",
                    f"roberta-{SIZE_BASE}",
                    f"roberta-{SIZE_LARGE}",
                    f"xlm-roberta-{SIZE_BASE}",
                    f"xlm-roberta-{SIZE_LARGE}",
                    f"random-bert"
                    }

MODEL_DIMS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 768,
              f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 768,
              f"bert-{SIZE_BASE}-{CASING_CASED}": 768,
              f"bert-{SIZE_BASE}-{CASING_UNCASED}": 768,
              f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 1024,
              f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 1024,
              f"bert-{SIZE_LARGE}-{CASING_CASED}": 1024,
              f"bert-{SIZE_LARGE}-{CASING_UNCASED}": 1024,
              f"roberta-{SIZE_BASE}": 768,
              f"roberta-{SIZE_LARGE}": 1024,
              f"xlm-roberta-{SIZE_BASE}": 768,
              f"xlm-roberta-{SIZE_LARGE}": 1024,
              f"random-bert": 768
              }

MODEL_LAYERS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 12,
                f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 12,
                f"bert-{SIZE_BASE}-{CASING_CASED}": 12,
                f"bert-{SIZE_BASE}-{CASING_UNCASED}": 12,
                f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 24,
                f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 24,
                f"bert-{SIZE_LARGE}-{CASING_CASED}": 24,
                f"bert-{SIZE_LARGE}-{CASING_UNCASED}": 24,
                f"roberta-{SIZE_BASE}": 12,
                f"roberta-{SIZE_LARGE}": 24,
                f"xlm-roberta-{SIZE_BASE}": 12,
                f"xlm-roberta-{SIZE_LARGE}": 24,
                f"random-bert": 12
                }

BERT_MODEL_DIR = "/net/projects/bert/models/"

# data pipeline options
BUFFER_SIZE = 50 * 1000 * 1000
SHUFFLE_SIZE = 512
