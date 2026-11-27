from MultiDeeploy.utils import normalize_layer_name

class LayerProfile:

    def __init__(self,
                 model_name,
                 layer_name,
                 tile,
                 double_buffering,
                 input_DMA,
                 kernel_exec,
                 output_DMA,
                 ops=None):

        self.model_name = model_name
        self.layer_name = layer_name
        self.tile = tile
        self.double_buffering = double_buffering
        self.input_DMA = input_DMA
        self.kernel_exec = kernel_exec
        self.output_DMA = output_DMA
        self.layer_processing_time = input_DMA + kernel_exec + output_DMA
        # Whole-layer op count (see testRunner.py's profiling-JSON writer): identical
        # across every tile of this layer, not a per-tile share. None for profiling
        # JSONs produced before this field existed.
        self.ops = ops


    def from_dict(d):
        return LayerProfile(
            model_name=None,
            layer_name=d["layer_name"],
            tile=d["tile"],
            double_buffering=d["double_buffering"],
            input_DMA=d["input_DMA"],
            kernel_exec=d["kernel_exec"],
            output_DMA=d["output_DMA"],
            ops=d.get("ops"),
        )

    def to_dict(self):
        return {
            "layer_name": self.layer_name,
            "tile": self.tile,
            "double_buffering": self.double_buffering,
            "input_DMA": self.input_DMA,
            "kernel_exec": self.kernel_exec,
            "output_DMA": self.output_DMA,
            "ops": self.ops,
        }
