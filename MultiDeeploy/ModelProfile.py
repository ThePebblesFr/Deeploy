from MultiDeeploy.LayerProfile import LayerProfile
from MultiDeeploy.utils import normalize_layer_name

class ModelProfile:

    def __init__(self,
                 name,
                 nb_dedicated_cores,
                 L1_dedicated_size,
                 layers=None):
        
        self.name = name
        self.nb_dedicated_cores = nb_dedicated_cores
        self.L1_dedicated_size = L1_dedicated_size
        self.layers = layers if layers is not None else []

    def from_dict(d):
        model_name = d["name"]

        layers = []
        for layer_dict in d.get("layers", []):
            layer = LayerProfile.from_dict(layer_dict)
            layer.model_name = model_name
            layer.layer_name = normalize_layer_name(model_name + "__" + layer.layer_name)[:-3]
            layers.append(layer)

        return ModelProfile(
            name=model_name,
            nb_dedicated_cores=d["nb_dedicated_cores"],
            L1_dedicated_size=d["L1_dedicated_size"],
            layers=layers,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "nb_dedicated_cores": self.nb_dedicated_cores,
            "L1_dedicated_size": self.L1_dedicated_size,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def total_ops(self):
        """
        Total operation count (theta) for one network inference: 1 MAC = 2 ops,
        matching Deeploy/DeeployTypes.py's NetworkDeployer.numberOfOps(), the
        authoritative compile-time source this is meant to reproduce from a
        profiling JSON. Deliberately NOT a sum over self.layers: each LayerProfile
        is one (layer, tile) pair, and a layer's `ops` (Deeploy's nodeOps) is its
        WHOLE op count baked identically into every tile's profiling line, not a
        per-tile share -- summing over tiles would overcount by the tile count for
        any layer tiled more than once. Sums one `ops` value per distinct
        layer_name instead.

        Returns None if this profile predates the `ops` field (profiling JSON
        generated before testRunner.py started capturing it).
        """
        seen = {}
        for layer in self.layers:
            if layer.ops is None:
                return None
            seen.setdefault(layer.layer_name, layer.ops)
        return sum(seen.values())
