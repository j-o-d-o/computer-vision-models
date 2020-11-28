# Semantic Segmentation

Loosely implements https://arxiv.org/abs/1505.04597. Note: all following python commands expect the root directory of the repo included in the PYTHONPATH</br>

## Training
```
activate computer-vision-models
python models/semseg/train.py
```
Results will be stored in ${workspaceRoot}/trained_models. It includes also a metrics.json which can be visualized with `python common/scripts/plot_metrics.py --path ./trained_models/semseg_*/metrics.json `

## Inference
```bash
# Inference on gpu
activate computer-vision-models
python models/semseg/inference.py --model_path ./trained_models/semseg_*/tf_model_*
```
Inference for tf lite is still work in progress as well as testing the ege tpu inference from the host machine.

## Tests
Assume a working mongodb connection and semseg data (comma10k) uploaded to it
