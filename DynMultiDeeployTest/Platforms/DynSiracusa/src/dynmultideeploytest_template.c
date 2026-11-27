#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"
#include "scenario_data.h"

// Task 1 = AnomalyDetection (light/heavy 2-class classifier: critical vs normal),
// Task 2 = TinyClassifier/ImageClassification (light/heavy 10-class classifier).
// Every symbol below is the LITERAL per-config name (e.g.
// AnomalyDetection_heavy_6_output_0), not a Model_1/Model_2 placeholder: unlike the
// earlier same-model-everywhere placeholder, each mode now uses a genuinely
// different model directory per task, which a single global regex substitution
// (Model_1 -> one fixed name) can't express per-mode.

typedef enum {
    IDLE,
    LOW_ENERGY,
    // Task 1 (AnomalyDetection) critical -> heavy + 6 cores; Task 2 stays light on 2.
    CRITICAL_TASK1,
    // Task 2 (TinyClassifier) critical -> heavy + 6 cores; Task 1 stays light on 2.
    CRITICAL_TASK2,
    BOTH_CRITICAL,
    DONE
} OperationMode;

// Which modes run Task 1's heavy variant, indexed directly by the OperationMode enum
// value -- generated from the active --configs file's per-mode v1 ('l' = heavy) by
// testUtils/ModeSwitchBodyGen.py (see its module docstring for why this, and the two
// switch bodies below, are generated rather than hand-written).
// ===== GENERATED: TASK1_HEAVY_LOOKUP START =====
static const int TASK1_MODE_IS_HEAVY[6] = {0, 0, 1, 0, 1, 0};
// ===== GENERATED: TASK1_HEAVY_LOOKUP END =====

#define FREQUENCY_HZ 50000000ULL
// TASK1_NUM_CLASSES is hand-set to match whichever model currently occupies Task 1's
// role across ALL 4 modes in the active --configs file (compute_margin() itself is
// class-count-agnostic) -- 2 for the AnomalyDetection classifier pair, 10 when Task 1
// is instead a second ImageClassification instance (imageclassification_
// imageclassification_4mode.json). Not config-driven (unlike TASK1_MODE_IS_HEAVY
// below): revert to 2 before rerunning the AnomalyDetection-based experiment.
#define TASK1_NUM_CLASSES 2
#define TASK2_NUM_CLASSES 2

// Per-tick scenario inputs are no longer static const L2 arrays (see
// ScenarioHeaderGen.py's docstring): they're flashed as raw blobs and loaded whole
// into HyperRAM once at boot, then DMA'd a single tick-row at a time into these two
// small, DELTA_T-independent L2 staging buffers via ram_read().
static int8_t task1_tick_buf[TASK1_INPUT_SIZE];
static int8_t task2_tick_buf[TASK2_INPUT_SIZE];

// Boot-time scenario blob load, dispatched to the cluster (see LoadScenarioData()
// below for why).
static int8_t *task1_scenario_ram;
static int8_t *task2_scenario_ram;

// load_file_to_ram() (dory_mem.c) is internally built on pi_cl_fs_read()/
// cl_ram_write() -- cluster-only primitives, same as every Deeploy-generated
// InitNetwork()'s own load_file_to_ram() call for oversized weight tensors, which is
// always dispatched to the cluster rather than called directly from the FC's main().
// Calling it directly from the FC (as this code originally did) crashed GVSoC with
// an "Invalid access" inside load_file_to_ram, before any of the actual file read
// happened -- mirroring the cluster-dispatch requirement fixes it.
static void LoadScenarioData(void *args) {
    if (pi_core_id() == 0) {
        task1_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK1_INPUT_SIZE);
        load_file_to_ram(task1_scenario_ram, TASK1_SCENARIO_HEX_FILE);
        task2_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK2_INPUT_SIZE);
        load_file_to_ram(task2_scenario_ram, TASK2_SCENARIO_HEX_FILE);
    }
}

// Confidence proxy shared by both tasks: margin between the top and runner-up
// logits. No argmax/softmax kernel exists standalone in this repo (see plan notes),
// so this is a small hand-written loop rather than wiring in the ONNX-graph-bound
// Softmax_s8. Task 1 (AnomalyDetection) used to be an autoencoder and used a
// reconstruction-error signal instead (input vs output, same-shaped buffers) --
// now that it's a real 2-class classifier (critical vs normal), its output has a
// completely different shape from its input, so it uses this same margin proxy as
// Task 2 instead, just over TASK1_NUM_CLASSES (2) logits.
static int32_t compute_margin(const int8_t *logits, uint32_t n) {
    int32_t max1 = logits[0];
    int32_t max2 = INT32_MIN;
    for (uint32_t i = 1; i < n; i++) {
        int32_t v = logits[i];
        if (v > max1) {
            max2 = max1;
            max1 = v;
        } else if (v > max2) {
            max2 = v;
        }
    }
    if (max2 == INT32_MIN) {
        max2 = max1;
    }
    return max1 - max2;
}

int main(void) {

    // INITIALIZATIONS
    #ifndef CI
    printf("HELLO WORLD:\r\n");
    #endif

    struct pi_cluster_conf conf;
    pi_cluster_conf_init(&conf);
    conf.id = 0;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
        return -1;
    mem_init();

    #ifndef NOFLASH
    open_fs();
    #endif

    printf("Intializing\r\n");

    // MAIN
    OperationMode operation_mode = LOW_ENERGY;
    OperationMode previous_operation_mode = IDLE;
    struct pi_cluster_task cluster_task;

    // Stream the two scenario blobs into HyperRAM once, whole, at boot -- exactly
    // the same load_file_to_ram() pattern Deeploy itself uses for weight tensors too
    // large for L2 (Deeploy/Targets/PULPOpen/Deployer.py's _L3InitTemplate). The
    // per-tick row DMA (ram_read()) happens inside the loop below.
    pi_cluster_task(&cluster_task, LoadScenarioData, NULL);
    cluster_task.stack_size = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    // Context state, updated after each inference and consumed by the FSM decision
    // at the top of the next iteration -- computed fresh every tick here since both
    // tasks always run together in this co-resident policy.
    int task1_critical = 0;
    int task2_critical = 0;
    uint64_t total_energy_estimate_nj = 0;
    uint32_t last_cycles = 0;
    int32_t task1_marg = 0;
    int32_t task2_marg = 0;

    for (int t = 0; t < DELTA_T; t++) {

        // DMA this tick's input row out of HyperRAM into the small local staging
        // buffers -- both tasks run every tick regardless of mode, so this happens
        // unconditionally rather than per-case below.
        ram_read(task1_tick_buf, task1_scenario_ram + (uint32_t) t * TASK1_INPUT_SIZE, TASK1_INPUT_SIZE);
        ram_read(task2_tick_buf, task2_scenario_ram + (uint32_t) t * TASK2_INPUT_SIZE, TASK2_INPUT_SIZE);

        // Context-driven mode decision for this tick, using the previous tick's
        // task confidence/energy state and this tick's operator order. Priority
        // order (highest first), per the paper's spec: power budget > operator
        // order > task confidence.
        int power_over_budget = (total_energy_estimate_nj > ENERGY_THRESHOLD_NJ);
        uint8_t operator_cmd = operator_order[t];

        if (power_over_budget) {
            operation_mode = LOW_ENERGY;
        } else if (operator_cmd == 3) {
            operation_mode = BOTH_CRITICAL;
        } else if (operator_cmd == 1) {
            operation_mode = CRITICAL_TASK1;
        } else if (operator_cmd == 2) {
            operation_mode = CRITICAL_TASK2;
        } else if (task1_critical && task2_critical) {
            operation_mode = BOTH_CRITICAL;
        } else if (task1_critical) {
            operation_mode = CRITICAL_TASK1;
        } else if (task2_critical) {
            operation_mode = CRITICAL_TASK2;
        } else {
            operation_mode = LOW_ENERGY;
        }

        printf("[HOST CORE] Tick %d: operation mode %d (task1_critical=%d, task2_critical=%d, "
               "operator_cmd=%u, power_over_budget=%d)\r\n",
               t, operation_mode, task1_critical, task2_critical, operator_cmd, power_over_budget);

        // Weights are compile-time-constant globals resident in L2 for every config
        // already baked into the binary -- they are NEVER reloaded on a mode switch,
        // matching the paper's "no re-deployment or weight reload" claim. What DOES
        // change per mode is the memory-arena layout InitNetworks_X_Y() lays out for
        // that mode's specific core/L1 split, so it only needs to run on an actual
        // mode transition, not on every inference: consecutive inferences that stay
        // in the same mode reuse the arena already laid out and skip straight to
        // RunNetworks_X_Y(), the lightweight path.
        int mode_changed = (operation_mode != previous_operation_mode);

        if (mode_changed) {
            // Flush the arenas allocated by the previous mode's InitNetworks_X_Y():
            // it always mallocs fresh arenas without freeing the old ones, so this
            // must run once, right before laying out the new mode's arena.
            switch (previous_operation_mode) {
                // ===== GENERATED: ARENA_FREE_CASES START =====
                case LOW_ENERGY:
                    pmsis_l1_malloc_free(TinyClassifier_light_4_MEMORYARENA_L1, TinyClassifier_light_4_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_light_4_MEMORYARENA_L2, TinyClassifier_light_4_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_light_4_MEMORYARENA_L1, AnomalyDetection_light_4_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_light_4_MEMORYARENA_L2, AnomalyDetection_light_4_MEMORYARENA_L2_len);
                    break;
                case CRITICAL_TASK2:
                    pmsis_l1_malloc_free(TinyClassifier_heavy_6_MEMORYARENA_L1, TinyClassifier_heavy_6_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_heavy_6_MEMORYARENA_L2, TinyClassifier_heavy_6_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_light_2_MEMORYARENA_L1, AnomalyDetection_light_2_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_light_2_MEMORYARENA_L2, AnomalyDetection_light_2_MEMORYARENA_L2_len);
                    break;
                case CRITICAL_TASK1:
                    pmsis_l1_malloc_free(TinyClassifier_light_2_MEMORYARENA_L1, TinyClassifier_light_2_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_light_2_MEMORYARENA_L2, TinyClassifier_light_2_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_heavy_6_MEMORYARENA_L1, AnomalyDetection_heavy_6_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_heavy_6_MEMORYARENA_L2, AnomalyDetection_heavy_6_MEMORYARENA_L2_len);
                    break;
                case BOTH_CRITICAL:
                    pmsis_l1_malloc_free(TinyClassifier_heavy_4_MEMORYARENA_L1, TinyClassifier_heavy_4_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_heavy_4_MEMORYARENA_L2, TinyClassifier_heavy_4_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_heavy_4_MEMORYARENA_L1, AnomalyDetection_heavy_4_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_heavy_4_MEMORYARENA_L2, AnomalyDetection_heavy_4_MEMORYARENA_L2_len);
                    break;
                // ===== GENERATED: ARENA_FREE_CASES END =====
                default:
                    // IDLE: nothing resident yet, nothing to free.
                    break;
            }
        }

        switch (operation_mode) {
            case IDLE:
                break;
            // ===== GENERATED: MODE_DISPATCH_CASES START =====
            case LOW_ENERGY:
                if (mode_changed) {
                    // send InitNetworks_s_s_4_4() cluster task
                    printf("[HOST CORE] Sending InitNetworks_s_s_4_4() cluster task\r\n");
                    pi_cluster_task(&cluster_task, InitNetworks_s_s_4_4, NULL);
                    cluster_task.stack_size = MAINSTACKSIZE;
                    cluster_task.slave_stack_size = SLAVESTACKSIZE;
                    ResetTimer();
                    StartTimer();
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                    StopTimer();
                    printf("Runtime InitNetwork: %u cycles\r\n", getCycles());
                }
                // TinyClassifier_light = Task 2
                for (uint32_t buf = 0; buf < TinyClassifier_light_4_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_light_4_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_light_4_inputs[buf], task2_tick_buf,
                            TinyClassifier_light_4_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_light = Task 1
                for (uint32_t buf = 0; buf < AnomalyDetection_light_4_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_light_4_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_light_4_inputs[buf], task1_tick_buf,
                            AnomalyDetection_light_4_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_s_s_4_4() cluster task
                printf("[HOST CORE] Sending RunNetworks_s_s_4_4() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_s_s_4_4, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                last_cycles = getCycles();
                printf("Runtime RunNetwork: %u cycles\r\n", last_cycles);
                task1_marg = compute_margin(AnomalyDetection_light_4_output_0, TASK1_NUM_CLASSES);
                task2_marg = compute_margin(TinyClassifier_light_4_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_LOW_ENERGY_NW) / FREQUENCY_HZ;
                break;
            case CRITICAL_TASK2:
                if (mode_changed) {
                    // send InitNetworks_l_s_6_2() cluster task
                    printf("[HOST CORE] Sending InitNetworks_l_s_6_2() cluster task\r\n");
                    pi_cluster_task(&cluster_task, InitNetworks_l_s_6_2, NULL);
                    cluster_task.stack_size = MAINSTACKSIZE;
                    cluster_task.slave_stack_size = SLAVESTACKSIZE;
                    ResetTimer();
                    StartTimer();
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                    StopTimer();
                    printf("Runtime InitNetwork: %u cycles\r\n", getCycles());
                }
                // TinyClassifier_heavy = Task 2 (6 cores in this mode)
                for (uint32_t buf = 0; buf < TinyClassifier_heavy_6_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_heavy_6_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_heavy_6_inputs[buf], task2_tick_buf,
                            TinyClassifier_heavy_6_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_light = Task 1 (2 cores in this mode)
                for (uint32_t buf = 0; buf < AnomalyDetection_light_2_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_light_2_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_light_2_inputs[buf], task1_tick_buf,
                            AnomalyDetection_light_2_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_l_s_6_2() cluster task
                printf("[HOST CORE] Sending RunNetworks_l_s_6_2() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_l_s_6_2, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                last_cycles = getCycles();
                printf("Runtime RunNetwork: %u cycles\r\n", last_cycles);
                task1_marg = compute_margin(AnomalyDetection_light_2_output_0, TASK1_NUM_CLASSES);
                task2_marg = compute_margin(TinyClassifier_heavy_6_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_CRITICAL_TASK2_NW) / FREQUENCY_HZ;
                break;
            case CRITICAL_TASK1:
                if (mode_changed) {
                    // send InitNetworks_s_l_2_6() cluster task
                    printf("[HOST CORE] Sending InitNetworks_s_l_2_6() cluster task\r\n");
                    pi_cluster_task(&cluster_task, InitNetworks_s_l_2_6, NULL);
                    cluster_task.stack_size = MAINSTACKSIZE;
                    cluster_task.slave_stack_size = SLAVESTACKSIZE;
                    ResetTimer();
                    StartTimer();
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                    StopTimer();
                    printf("Runtime InitNetwork: %u cycles\r\n", getCycles());
                }
                // TinyClassifier_light = Task 2 (2 cores in this mode)
                for (uint32_t buf = 0; buf < TinyClassifier_light_2_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_light_2_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_light_2_inputs[buf], task2_tick_buf,
                            TinyClassifier_light_2_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_heavy = Task 1 (6 cores in this mode)
                for (uint32_t buf = 0; buf < AnomalyDetection_heavy_6_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_heavy_6_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_heavy_6_inputs[buf], task1_tick_buf,
                            AnomalyDetection_heavy_6_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_s_l_2_6() cluster task
                printf("[HOST CORE] Sending RunNetworks_s_l_2_6() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_s_l_2_6, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                last_cycles = getCycles();
                printf("Runtime RunNetwork: %u cycles\r\n", last_cycles);
                task1_marg = compute_margin(AnomalyDetection_heavy_6_output_0, TASK1_NUM_CLASSES);
                task2_marg = compute_margin(TinyClassifier_light_2_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_CRITICAL_TASK1_NW) / FREQUENCY_HZ;
                break;
            case BOTH_CRITICAL:
                if (mode_changed) {
                    // send InitNetworks_l_l_4_4() cluster task
                    printf("[HOST CORE] Sending InitNetworks_l_l_4_4() cluster task\r\n");
                    pi_cluster_task(&cluster_task, InitNetworks_l_l_4_4, NULL);
                    cluster_task.stack_size = MAINSTACKSIZE;
                    cluster_task.slave_stack_size = SLAVESTACKSIZE;
                    ResetTimer();
                    StartTimer();
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                    StopTimer();
                    printf("Runtime InitNetwork: %u cycles\r\n", getCycles());
                }
                // TinyClassifier_heavy = Task 2
                for (uint32_t buf = 0; buf < TinyClassifier_heavy_4_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_heavy_4_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_heavy_4_inputs[buf], task2_tick_buf,
                            TinyClassifier_heavy_4_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_heavy = Task 1
                for (uint32_t buf = 0; buf < AnomalyDetection_heavy_4_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_heavy_4_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_heavy_4_inputs[buf], task1_tick_buf,
                            AnomalyDetection_heavy_4_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_l_l_4_4() cluster task
                printf("[HOST CORE] Sending RunNetworks_l_l_4_4() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_l_l_4_4, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                last_cycles = getCycles();
                printf("Runtime RunNetwork: %u cycles\r\n", last_cycles);
                task1_marg = compute_margin(AnomalyDetection_heavy_4_output_0, TASK1_NUM_CLASSES);
                task2_marg = compute_margin(TinyClassifier_heavy_4_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_BOTH_CRITICAL_NW) / FREQUENCY_HZ;
                break;
            // ===== GENERATED: MODE_DISPATCH_CASES END =====
            default:
                break;
        }

        // Which threshold applies depends on whether Task 1 ran heavy this tick --
        // read from the generated TASK1_MODE_IS_HEAVY lookup rather than assuming
        // which enum values are heavy, since that's config-dependent.
        int task1_used_heavy = TASK1_MODE_IS_HEAVY[operation_mode];
        task1_critical = (task1_marg < (task1_used_heavy ? TASK1_MARGIN_THRESHOLD_HEAVY : TASK1_MARGIN_THRESHOLD_LIGHT));
        task2_critical = (task2_marg < TASK2_MARGIN_THRESHOLD);
        printf("[HOST CORE] Tick %d context: task1_margin=%d task2_margin=%d total_energy_est_nj=%llu\r\n",
               t, task1_marg, task2_marg, (unsigned long long) total_energy_estimate_nj);

        // Captures the mode we just dispatched -- this is what mode_changed compares
        // against on the next loop iteration.
        previous_operation_mode = operation_mode;
    }

    return 0;
}
