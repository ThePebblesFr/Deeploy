from MultiDeeploy.LayerProfile import LayerProfile

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
            layer.layer_name = model_name + "__" + layer.layer_name
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
