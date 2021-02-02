# Semantic Segmentation

## Data
Upload the comma10k data to a MongoDB. See data/comma10k.py for the upload script and semseg_spec.py for the label spec.

## Training
```
conda activate computer-vision-models
python models/semseg/train.py
```
Results will be stored in ${workspaceRoot}/trained_models. It includes also a metrics.json which can be visualized with `python common/scripts/plot_metrics.py --path ./trained_models/semseg_*/metrics.json `

## Inference
```bash
# Inference on gpu
conda activate computer-vision-models
python models/semseg/inference.py --model_path ./trained_models/semseg_*/tf_model_*
```
