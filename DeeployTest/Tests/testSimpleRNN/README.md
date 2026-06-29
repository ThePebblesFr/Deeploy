# testSimpleRNN

## Model Information

Model: SimpleRNN from example_models.py

## Test Details

- **Input shapes**: (1, 10)
- **Random seed**: 42
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
python testRunner_generic.py -t Tests/testSimpleRNN

# Siracusa platform
python testRunner_siracusa.py -t Tests/testSimpleRNN --cores=8

# With tiling
python testRunner_tiled_siracusa.py -t Tests/testSimpleRNN --cores=8 --l1=16000
```

