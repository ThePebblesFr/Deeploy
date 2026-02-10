from typing import List
from MultiDeeploy.ModelProfile import ModelProfile

class WorkloadScheduler:

    def __init__(self,
                 models: List[ModelProfile]):
        """
        models: List of ModelProfile
        """
        self.models = models

        self.Models_layers = [m.layers for m in models]

        # Outputs
        self.Scheduling_queue = []
        self.Longest_layer_model_idx_per_round = []

    def run(self):
        """
        Executes the scheduling algorithm and fills:
            - self.Scheduling_queue
            - self.Longest_layer_model_idx_per_round
        """
        num_models = len(self.Models_layers)

        # Track which layer index to read next for each model
        next_processed_layers = [0 for _ in range(num_models)]

        all_layers_processed = False

        while not all_layers_processed:

            # STEP 1: Gather "next layer" of each model
            next_layers = []
            for i in range(num_models):
                if next_processed_layers[i] == -1:
                    next_layers.append(None)
                else:
                    next_layers.append(self.Models_layers[i][next_processed_layers[i]])

            # STEP 2: Select the longest next layer among models
            next_longest_layer = max(
                (x for x in next_layers if x is not None),
                key=lambda x: x.layer_processing_time
            )
            next_longest_layer_index = next_layers.index(next_longest_layer)

            # Save longest-model index
            self.Longest_layer_model_idx_per_round.append(next_longest_layer_index)

            processed_layers_next_round = []

            # STEP 3: For each model, decide batch to process
            for i in range(num_models):

                # Case: this model contributes the longest layer
                if i == next_longest_layer_index:
                    processed_layers_next_round.append([next_longest_layer])

                    potential = next_processed_layers[i] + 1
                    next_processed_layers[i] = (
                        -1
                        if potential >= len(self.Models_layers[i])
                        else potential
                    )

                # Case: this model is already finished
                elif next_processed_layers[i] == -1:
                    processed_layers_next_round.append([])

                # Case: contribute a batch of layers that fits under the longest kernel_exec
                else:
                    remaining = self.Models_layers[i][next_processed_layers[i]:]

                    batch = [remaining[0]]
                    total = remaining[0].layer_processing_time

                    j = 1
                    while (
                        j < len(remaining)
                        and total + remaining[j].layer_processing_time < next_longest_layer.layer_processing_time
                    ):
                        batch.append(remaining[j])
                        total += remaining[j].layer_processing_time
                        j += 1

                    processed_layers_next_round.append(batch)

                    potential = next_processed_layers[i] + len(batch)
                    next_processed_layers[i] = (
                        -1
                        if potential >= len(self.Models_layers[i])
                        else potential
                    )

            # Save scheduling round
            self.Scheduling_queue.append(processed_layers_next_round)

            # STEP 4: Check whether all models are done
            all_layers_processed = all(x == -1 for x in next_processed_layers)

        return self.Scheduling_queue
    
    def print_schedule(self):
        """
        Pretty-print the scheduling rounds, showing:
            - layers executed each round per model
            - per-layer duration (layer_processing_time)
            - per-round total duration (max layer time)
        """

        print("\n===== Workload Scheduling Summary =====\n")

        for round_idx, round_layers in enumerate(self.Scheduling_queue):
            print(f"--- Round {round_idx} ---")

            # Collect all layer durations to compute round duration
            round_layer_times = []

            for model_idx, layers in enumerate(round_layers):
                if not layers:
                    print(f"  Model {model_idx}: (no layers)")
                    continue

                layer_desc = []
                for layer in layers:
                    layer_desc.append(f"{layer.layer_name} [time={layer.layer_processing_time}]")
                    round_layer_times.append(layer.layer_processing_time)

                print(f"  Model {model_idx}: " + ", ".join(layer_desc))

            # round duration = max layer_processing_time among layers processed this round
            round_duration = max(round_layer_times) if round_layer_times else 0
            print(f"  → Round duration: {round_duration} cycles\n")

        print("===== End of Scheduling Summary =====\n")
