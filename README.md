# Orthogonal Probing

The repository contains code for [Introducing Orthogonal Constraint in Structural Probes](https://arxiv.org/abs/2012.15228) and some extensions to run various ablation experiments and reproduce the original paper results. 

## Data preparation

Before performing probing, save language data and Transformer embeddings in TFRecord format. This is necessary
clear out lengthy Transformer inference during probe training.

The code allows converting the [ConLL-U](https://universaldependencies.org/format.html) language data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files ready for probing. Firstly, save your train, development, and test sets
in `resources/entrain.conllu`, `resources/endev.conllu`, and `resources/entest.conllu`, respectively. Secondly, remember to install required
python dependencies (Python 3.8 recommended).
```
pip install -r requirements.txt
```
Then run the conversion script:

```
cd src
python save_tfrecord.py ../resources/tf_data
```

The files will be saved to `../resources/tf_data`. By default, BERT cased multilingual
model is used. It is possible to store data for multiple models and datasets in an output directory. However, it may
take up a significant storage space.


## Probing

To perform probing run:

```
cd src
python probe.py ../experiments ../resources/tf_data --languages "en" --layer-index 6 --tasks "dep_depth dep_distance pos_depth pos_distance" --learning-rate 0.02 --ortho 0.05 
```

The probe will be trained on data stored in `../resources/tf_data`. Its weights and tensorboard tracking data will be saved in
a new subdirectory of `../experiments`. Essential arguments are:
* `languages`: a list of languages for which we probe, by default only **en**.
* `layer-index`: a zero-based index of the layer on top of which we train probes.
* `tasks`: a list of probing objectives, described below.
* `learning-rate`: initial learning rate.
* `ortho`: orthogonal penalty weight.
* `norm`: distance norm used to approximate tree distances (euclidean, 1, 2, 3, tf.inf)

Description of other arguments can be found in `src/probe.py`.

Supported probing objectives:
* `dep_depth`: syntactic dependency depth
* `dep_distance`: syntactic dependency distance
* `lex_depth`: lexical hypernymy depth
* `lex_distance`: lexical hypernymy distance
* `pos_depth`: sentence positional depth (i.e. word's index in a sentence)
* `pos_distance`: sentence positional distance
* `rnd_depth`: random structure depth
* `rnd_distance`: random structure distance 



## Reporting results

To report results (Spearman's rank correlation and (U)UAS) run:

```
cd src
python report.py ../experiments ../resources/tf_data --languages "en" --layer-index 6 --tasks "dep_depth dep_distance pos_depth pos_distance"
```

The definition of the arguments is the same as in `src/report.py`. Results will be saved to a subdirectory created in the previous
step, thus it's important to call `report.py` with the same arguments as `probe.py`.

## Ablation Experiments

Sample command to train the model with L1 distance metric over all the layers on engligh dataset:

`python probe.py ../experiments ../resources/tf_data --languages "en" --layer-index -1 --tasks "dep_depth dep_distance pos_depth pos_distance" --learning-rate 0.02 --ortho 0.05 --norm "1" --batch-size 12`

Sample command to evaluate the model with various correlation methods:

`python report.py ../experiments ../resources/tf_data --languages "en" --layer-index -1 --tasks "dep_depth dep_distance pos_depth pos_distance" --correlation "spearman"`

`python report.py ../experiments ../resources/tf_data --languages "en" --layer-index -1 --tasks "dep_depth dep_distance pos_depth pos_distance" --correlation "pearson"`

`python report.py ../experiments ../resources/tf_data --languages "en" --layer-index -1 --tasks "dep_depth dep_distance pos_depth pos_distance" --correlation "kendall"`

