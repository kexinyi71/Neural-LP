# Neural LP

This is my implementation of Neural Logic Programming, proposed in the following paper:

[Differentiable Learning of Logical Rules for Knowledge Base Reasoning](https://arxiv.org/abs/1702.08367).
Fan Yang, Zhilin Yang, William W. Cohen.
NIPS 2017.

I copy this from [fanyangxyz](https://github.com/fanyangxyz)/**Neural-LP**  and change some code.

## Requirements

- Python 3.6.5
- Numpy 
- Tensorflow 1.11.0

## Quick start

The following command starts training a dataset about kinship relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/kinship --exps_dir=exps/ --exp_name=demo
```

Navigate to `exps/demo/`, there is `rules.txt` that contains learned logical rules. 

## Evaluation

To evaluate the prediction results, follow the steps below. The first two steps is preparation so that we can compute _filtered_ ranks (see [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) for details).

We use the experiment from Quick Start as an example. Change the folder names (datasets/kinship exps/dev) for other experiments.

```
run "cat train.txt facts.txt valid.txt test.txt > all.txt" in your data path
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds=exps/demo/test_predictions.txt --truths=datasets/kinship/truths.pckl
```