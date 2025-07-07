# Traffic Sign Detection
## Train
python3 train.py

## Dataset Url
https://drive.google.com/file/d/1AQ2cDKEmsRGtud8AfGN5v_KltmfWdzyZ/view?usp=sharing

## Prune
### Structure prune
python3 structure_prune.py
Didn't work due to structure of yolo.

### Unstructure prune
python3 simple_prune.py
Reduce size of Conv layers 10%.

## Quantization
Use onnxruntime to quantize the model to int8
python3 quantize.py

## Analysis
- Disk Usage: 2.80 MBs
- Total Model Parameters: 2,607,255
- Estimated Weight RAM Usage: 2.53 MB
- FLOPS: 63.58M