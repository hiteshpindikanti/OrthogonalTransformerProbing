import os
import argparse
import json

from data_support.tfrecord_wrapper import TFRecordReader
from network import Network

import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("parent_dir", type=str, default="../experiments", help="Parent experiment directory")
    parser.add_argument("data_dir", type=str, default='../resources/tf_data',
                        help="Directory where tfrecord files are stored")
    parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")

    parser.add_argument("--languages", nargs='*', default=['en'], type=str,
                        help="Languages to probe.")
    parser.add_argument("--tasks", nargs='*', type=str,
                        help="Probing tasks (distance, lex-distance, depth or lex-depth)")

    # Probe arguments
    parser.add_argument("--probe-rank", default=None, type=int, help="Rank of the probe")
    parser.add_argument("--no-ortho-probe", action="store_true", help="Resign from ortho probe (store false)")
    parser.add_argument("--only-sv", action="store_true",
                        help="Probe with only Scaling Vector, this option will automatically diable Orthogonal Transformation")
    parser.add_argument("--with-sv", action="store_true",
                        help="Probe with Scaling Vector, even without orthogonal constraint")
    parser.add_argument("--layer-index", default=6, type=int, help="Index of BERT's layer to probe."
                                                                   "If -1 all layers embeddings are averaged")
    parser.add_argument("--norm", default='euclidean', type=str, help="Distance/Depth Norm calculation. Available options same as ord attribute for tf.norm ['euclidean, 1, 2, 3, tf.inf']")
    # Train arguments
    parser.add_argument("--seed", default=42, type=int, help="Seed for variable initialisation")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
    parser.add_argument("--epochs", default=40, type=int, help="Maximal number of training epochs")
    parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--ortho", default=None, type=float,
                        help="Orthogonality reguralization (SRIP) for language map matrices.")
    parser.add_argument("--l1", default=None, type=float, help="L1 reguralization of the weights.")
    parser.add_argument("--clip-norm", default=None, type=float, help="Clip gradient norm to this value")

    parser.add_argument("--subsample-train", default=None, type=int,
                        help="Size of subsample taken from a training set.")
    parser.add_argument("--zs-dep-languages", nargs='*', default=[], type=str,
                        help="List of languages to disregard in dependency probing training (to evaluate 0 shot capability).")
    parser.add_argument("--fs-dep-languages", nargs='*', default=[], type=str, help="List of few shot languages.")
    parser.add_argument("--fewshot-size", default=10, type=int,
                        help="Number of trainining sentences for few shot languages")
    # Specify Bert Model
    parser.add_argument("--model",
                        default=f"bert-{constants.SIZE_BASE}-{constants.LANGUAGE_MULTILINGUAL}-{constants.CASING_CASED}",
                        help="Transformer model name (see: https://huggingface.co/transformers/pretrained_models.html)")

    args = parser.parse_args()

    args.ml_probe = not args.no_ortho_probe
    if args.only_sv:
        args.ml_probe = False
    if not args.probe_rank:
        args.probe_rank = constants.MODEL_DIMS[args.model]

    do_lower_case = (constants.CASING_UNCASED in args.model)

    if args.seed == 42:
        experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}"
    else:
        experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}-seed_{args.seed}"
    args.out_dir = os.path.join(args.parent_dir, experiment_name)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    tf_reader = TFRecordReader(args.data_dir, args.model)
    tf_reader.read(args.tasks, args.languages)

    network = Network(args)

    network.train(tf_reader, args)
