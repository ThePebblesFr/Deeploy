# ES_CNN

## Model Information

Model: ES_CNN from example_models.py

## Test Details

- **Input shapes**: (1, 1, 16, 1)
- **Random seed**: None (random)
- **Generated**: Automatically using create_deeploy_test.py

## Files

- `network.onnx`: ONNX model file
- `inputs.npz`: Test input data
- `outputs.npz`: Expected output data
- `activations.npz`: Intermediate activations (if available)

## Running the Test

```bash
cd DeeployTest

# Generic platform
python testRunner_generic.py -t Tests/ES_CNN

# Siracusa platform
python testRunner_siracusa.py -t Tests/ES_CNN --cores=8

# With tiling
python testRunner_tiled_siracusa.py -t Tests/ES_CNN --cores=8 --l1=16000
```

