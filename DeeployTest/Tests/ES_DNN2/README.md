# ES_DNN2

## Model Information

Model: ES_DNN2 from example_models.py

## Test Details

- **Input shapes**: (1, 16)
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
python testRunner_generic.py -t Tests/ES_DNN2

# Siracusa platform
python testRunner_siracusa.py -t Tests/ES_DNN2 --cores=8

# With tiling
python testRunner_tiled_siracusa.py -t Tests/ES_DNN2 --cores=8 --l1=16000
```

