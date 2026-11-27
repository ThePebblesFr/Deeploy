#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"
#include "scenario_data.h"

// Single-shot variant of ../DynSiracusa/src/dynmultideeploytest_template.c:
// instead of looping DELTA_T times in-process (one pi_cl_team_fork per tick
// on the same Combined_closure_*), this binary processes EXACTLY ONE
// (tick, mode) pair per process launch, read from a tiny "control.hex" blob
// flashed alongside the scenario data, then exits.
//
// Why this exists: enabling MultiDeeploy's double-buffered (DB) closures in
// the real multi-tick scenario harness hangs deterministically after a
// handful of repeated in-process pi_cl_team_fork calls -- investigated at
// length (channel-leak, barrier-event-mask-leak, and deferred-single-tile-
// flush hypotheses were each tested and disproven/found insufficient; see
// the plan file's "Step 2 follow-up" section). A single inference through
// this same FSM logic is proven correct (and faster) with DB. The workaround
// sidesteps the repeated-invocation trigger entirely: a host-side Python
// orchestrator (scenarioUtils/run_singleshot_scenario.py) relaunches this
// binary once per tick, in a genuinely fresh GVSoC process each time.
//
// The mode DECISION itself moves entirely to the orchestrator (Python), not
// on-device: the orchestrator already needs each tick's real task1_err/
// task2_margin (parsed back from this binary's own printed output) to decide
// the NEXT tick's mode -- inherently sequential either way, so there is no
// benefit to also deciding a tick's own mode on-device, and doing it
// host-side avoids needing to carry any FSM state (previous mode, running
// energy estimate) across separate process launches at all.
//
// previous_operation_mode is hardcoded to IDLE below rather than threaded
// through from the real previous mode: this deliberately forces
// mode_changed=1 on every single launch (required for correctness in a
// fresh process, where no weights are pre-loaded yet) and makes the
// ARENA_FREE_CASES switch below hit its harmless `default:` case for free
// (nothing was allocated in this fresh process to free). This changes
// nothing about the reported hardware metrics: the power-measurement
// bracket below (0xabbaabba/0xdeadcaca) wraps ONLY the RunNetworks_*
// dispatch, never InitNetworks_* -- exactly like the original multi-tick
// template -- so Init's cost was already excluded from energy_j/gamma_eff
// in the existing continuous-process measurement methodology, and always
// running it here for correctness has zero effect on the aggregated metrics.

typedef enum {
    IDLE,
    LOW_ENERGY,
    CRITICAL_TASK1,
    CRITICAL_TASK2,
    BOTH_CRITICAL,
    DONE
} OperationMode;

// Which modes run Task 1's heavy variant -- unused by this binary itself
// (the orchestrator already knows this from the same --configs file used to
// generate it), kept only because inject_generated_sections() (shared,
// unmodified, with ../DynSiracusa/src/dynmultideeploytest_template.c)
// requires this marker pair to be present.
// ===== GENERATED: TASK1_HEAVY_LOOKUP START =====
static const int TASK1_MODE_IS_HEAVY[6] = {0, 0, 1, 0, 1, 0};
// ===== GENERATED: TASK1_HEAVY_LOOKUP END =====

#define FREQUENCY_HZ 50000000ULL
#define TASK2_NUM_CLASSES 10

// Per-tick scenario inputs: same HyperRAM-streaming pattern as the multi-tick
// template -- the WHOLE DELTA_T-tick blob is loaded once at boot (cheap,
// <100KB), then exactly one tick's row is DMA'd out via ram_read() below,
// at the offset named by the control blob. No scenario regeneration is
// needed between launches -- only control.hex's bytes change.
static int8_t task1_tick_buf[TASK1_INPUT_SIZE];
static int8_t task2_tick_buf[TASK2_INPUT_SIZE];

static int8_t *task1_scenario_ram;
static int8_t *task2_scenario_ram;

// Tiny per-launch control blob: which tick's input row to process, and which
// mode to dispatch. Written fresh by the host orchestrator before every
// relaunch as "control.hex" in the SAME GENERATED_SOURCE/hex/ directory the
// scenario data already lives in -- picked up automatically by the existing
// `file(GLOB_RECURSE HEXLIST "${GENERATED_SOURCE}/hex/**")` glob in
// DynMultiDeeployTest/CMakeLists.txt (no CMake changes needed). Its on-disk
// path is fixed once at the first `cmake configure`; only its BYTES change
// between the ~100 relaunches, so only one configure+build is ever needed.
typedef struct {
    uint32_t tick_index;
    uint32_t mode;
} ControlBlob;

static int8_t *control_scenario_ram;
static ControlBlob control;

// load_file_to_ram() (dory_mem.c) is internally built on pi_cl_fs_read()/
// cl_ram_write() -- cluster-only primitives -- so this must be dispatched to
// the cluster rather than called directly from the FC's main(), same as the
// multi-tick template. Loads all three HyperRAM-resident blobs (task1/task2
// scenario data + the control blob) from ONE cluster-task dispatch --
// confirmed necessary at
// DELTA_T=100 (not just a cosmetic choice): splitting the control-blob load
// into a SEPARATE pi_cluster_send_task_to_cl dispatch caused a HyperRAM
// "out-of-bound request" at exactly ram_size (0x800000), even though the
// actual bytes allocated across all three blobs are nowhere near the 8MB
// HyperRAM size -- some allocator state does not carry correctly across
// separate cluster-task dispatches at this scale (worked fine at DELTA_T=3,
// broke at DELTA_T=100). Doing all three mallocs+loads within a single
// dispatch, mirroring the multi-tick template's original one-dispatch
// pattern as closely as possible, avoids the issue entirely.
static void LoadScenarioData(void *args) {
    if (pi_core_id() == 0) {
        task1_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK1_INPUT_SIZE);
        load_file_to_ram(task1_scenario_ram, TASK1_SCENARIO_HEX_FILE);
        task2_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK2_INPUT_SIZE);
        load_file_to_ram(task2_scenario_ram, TASK2_SCENARIO_HEX_FILE);
        control_scenario_ram = (int8_t *) cl_ram_malloc(sizeof(ControlBlob));
        load_file_to_ram(control_scenario_ram, "control.hex");
    }
}

// Task 1 (AnomalyDetection) criticality: sum of abs differences between the
// scenario's input row and the reconstructed output, in raw INT8 domain.
// Identical to the multi-tick template's copy.
static uint32_t compute_recon_error(const int8_t *input_row, const int8_t *output, uint32_t len) {
    uint32_t err = 0;
    for (uint32_t i = 0; i < len; i++) {
        int32_t diff = (int32_t) input_row[i] - (int32_t) output[i];
        err += (diff < 0) ? (uint32_t) (-diff) : (uint32_t) diff;
    }
    return err;
}

// Task 2 (TinyClassifier) confidence proxy: margin between the top and
// runner-up logits. Identical to the multi-tick template's copy.
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

    OperationMode previous_operation_mode = IDLE;
    struct pi_cluster_task cluster_task;

    pi_cluster_task(&cluster_task, LoadScenarioData, NULL);
    cluster_task.stack_size = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    ram_read((int8_t *) &control, control_scenario_ram, sizeof(ControlBlob));

    int t = (int) control.tick_index;
    OperationMode operation_mode = (OperationMode) control.mode;

    uint32_t last_cycles = 0;
    uint32_t task1_err = 0;
    int32_t task2_marg = 0;
    // Unused/discarded by this binary itself -- only declared to satisfy the
    // generated MODE_DISPATCH_CASES text (reused verbatim, unmodified, from
    // the multi-tick template), which accumulates into this name. The
    // orchestrator tracks its own real running total host-side instead (see
    // the file-level comment).
    uint64_t total_energy_estimate_nj = 0;

    ram_read(task1_tick_buf, task1_scenario_ram + (uint32_t) t * TASK1_INPUT_SIZE, TASK1_INPUT_SIZE);
    ram_read(task2_tick_buf, task2_scenario_ram + (uint32_t) t * TASK2_INPUT_SIZE, TASK2_INPUT_SIZE);

    // Printed for parse_scenario_log.py's TICK_DECISION_RE compatibility --
    // task1_critical/task2_critical/operator_cmd/power_over_budget are
    // informational only here (the orchestrator already decided
    // operation_mode itself, host-side); printed as 0 rather than
    // recomputed, since this binary has no independent use for them.
    printf("[HOST CORE] Tick %d: operation mode %d (task1_critical=%d, task2_critical=%d, "
           "operator_cmd=%u, power_over_budget=%d)\r\n",
           t, operation_mode, 0, 0, 0, 0);

    // Always 1: previous_operation_mode is hardcoded IDLE above, and
    // operation_mode (read from the control blob) is never IDLE for a real
    // dispatch -- see the file-level comment for why this is deliberate and
    // metrics-neutral.
    int mode_changed = (operation_mode != previous_operation_mode);

    if (mode_changed) {
        switch (previous_operation_mode) {
            // ===== GENERATED: ARENA_FREE_CASES START =====
                case LOW_ENERGY:
                    pmsis_l1_malloc_free(TinyClassifier_light_4_MEMORYARENA_L1, TinyClassifier_light_4_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_light_4_MEMORYARENA_L2, TinyClassifier_light_4_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_light_4_MEMORYARENA_L1, AnomalyDetection_light_4_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_light_4_MEMORYARENA_L2, AnomalyDetection_light_4_MEMORYARENA_L2_len);
                    break;
                case CRITICAL_TASK1:
                    pmsis_l1_malloc_free(TinyClassifier_light_2_MEMORYARENA_L1, TinyClassifier_light_2_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_light_2_MEMORYARENA_L2, TinyClassifier_light_2_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_heavy_6_MEMORYARENA_L1, AnomalyDetection_heavy_6_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_heavy_6_MEMORYARENA_L2, AnomalyDetection_heavy_6_MEMORYARENA_L2_len);
                    break;
                case CRITICAL_TASK2:
                    pmsis_l1_malloc_free(TinyClassifier_heavy_6_MEMORYARENA_L1, TinyClassifier_heavy_6_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_heavy_6_MEMORYARENA_L2, TinyClassifier_heavy_6_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_light_2_MEMORYARENA_L1, AnomalyDetection_light_2_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_light_2_MEMORYARENA_L2, AnomalyDetection_light_2_MEMORYARENA_L2_len);
                    break;
                case BOTH_CRITICAL:
                    pmsis_l1_malloc_free(TinyClassifier_heavy_4_MEMORYARENA_L1, TinyClassifier_heavy_4_MEMORYARENA_L1_len);
                    pi_l2_free(TinyClassifier_heavy_4_MEMORYARENA_L2, TinyClassifier_heavy_4_MEMORYARENA_L2_len);
                    pmsis_l1_malloc_free(AnomalyDetection_heavy_4_MEMORYARENA_L1, AnomalyDetection_heavy_4_MEMORYARENA_L1_len);
                    pi_l2_free(AnomalyDetection_heavy_4_MEMORYARENA_L2, AnomalyDetection_heavy_4_MEMORYARENA_L2_len);
                    break;
            // ===== GENERATED: ARENA_FREE_CASES END =====
            default:
                // IDLE: nothing resident yet in this fresh process, nothing to free.
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
                // TinyClassifier_light_4 = Task 2 (4 cores)
                for (uint32_t buf = 0; buf < TinyClassifier_light_4_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_light_4_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_light_4_inputs[buf], task2_tick_buf,
                            TinyClassifier_light_4_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_light_4 = Task 1 (4 cores)
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
                task1_err = compute_recon_error(task1_tick_buf, AnomalyDetection_light_4_output_0, TASK1_INPUT_SIZE);
                task2_marg = compute_margin(TinyClassifier_light_4_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_LOW_ENERGY_NW) / FREQUENCY_HZ;
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
                // TinyClassifier_light_2 = Task 2 (2 cores)
                for (uint32_t buf = 0; buf < TinyClassifier_light_2_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_light_2_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_light_2_inputs[buf], task2_tick_buf,
                            TinyClassifier_light_2_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_heavy_6 = Task 1 (6 cores)
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
                task1_err = compute_recon_error(task1_tick_buf, AnomalyDetection_heavy_6_output_0, TASK1_INPUT_SIZE);
                task2_marg = compute_margin(TinyClassifier_light_2_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_CRITICAL_TASK1_NW) / FREQUENCY_HZ;
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
                // TinyClassifier_heavy_6 = Task 2 (6 cores)
                for (uint32_t buf = 0; buf < TinyClassifier_heavy_6_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_heavy_6_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_heavy_6_inputs[buf], task2_tick_buf,
                            TinyClassifier_heavy_6_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_light_2 = Task 1 (2 cores)
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
                task1_err = compute_recon_error(task1_tick_buf, AnomalyDetection_light_2_output_0, TASK1_INPUT_SIZE);
                task2_marg = compute_margin(TinyClassifier_heavy_6_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_CRITICAL_TASK2_NW) / FREQUENCY_HZ;
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
                // TinyClassifier_heavy_4 = Task 2 (4 cores)
                for (uint32_t buf = 0; buf < TinyClassifier_heavy_4_num_inputs; buf++) {
                    if ((uint32_t) TinyClassifier_heavy_4_inputs[buf] >= 0x10000000) {
                    memcpy(TinyClassifier_heavy_4_inputs[buf], task2_tick_buf,
                            TinyClassifier_heavy_4_inputs_bytes[buf]);
                    }
                }
                // AnomalyDetection_heavy_4 = Task 1 (4 cores)
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
                task1_err = compute_recon_error(task1_tick_buf, AnomalyDetection_heavy_4_output_0, TASK1_INPUT_SIZE);
                task2_marg = compute_margin(TinyClassifier_heavy_4_output_0, TASK2_NUM_CLASSES);
                total_energy_estimate_nj += ((uint64_t) last_cycles * PHI_ESTIMATE_BOTH_CRITICAL_NW) / FREQUENCY_HZ;
                break;
        // ===== GENERATED: MODE_DISPATCH_CASES END =====
        default:
            break;
    }

    // total_energy_est_nj is printed as 0 (unused/uninitialized on-device
    // accumulator in this binary) rather than a real running total: the
    // orchestrator tracks its own copy host-side (from each tick's real
    // RunNetwork cycle count + the mode's PHI_ESTIMATE, exactly the same
    // formula the multi-tick template's on-device accumulator used) since it
    // needs that value anyway to decide the NEXT tick's mode.
    printf("[HOST CORE] Tick %d context: task1_err=%u task2_margin=%d total_energy_est_nj=%llu\r\n",
           t, task1_err, task2_marg, (unsigned long long) 0);

    return 0;
}
