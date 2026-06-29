# ES_DNN1_int
Generated from an existing ONNX model.
## Files
- `network.onnx`: shape-inferred ONNX model copied/generated for Deeploy
- `inputs.npz`: generated or supplied test inputs
- `outputs.npz`: ONNX Runtime reference outputs

## Inputs
- `input_0`: shape `(1, 16)`, dtype `float32`

## Outputs
- `output_0`: shape `(1, 10)`, dtype `float32`

## Notes
- NPZ key style: `named`
- The saved `network.onnx` is shape-inferred to avoid Deeploy missing-shape errors.
