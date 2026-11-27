#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "scenario_data.h"

// VisualWakeWords_heavy/AnomalyDetection_heavy placeholders substituted by testUtils/UpdateRoundRobinMain.py
// (reused verbatim -- same placeholder convention as Round-Robin-Large, since this
// binary links the exact same fused Network.c/.h that produces). VisualWakeWords_heavy = Task 2
// (TinyClassifier_heavy), AnomalyDetection_heavy = Task 1 (AnomalyDetection_heavy).
//
// True priority preemption, per the user's explicit choice over the two other
// options considered (variant-adapts-to-criticality, and order-swaps-but-both-
// still-run): each tick, ONLY the higher-priority task actually runs on the full
// cluster; the other is skipped outright, not deferred-and-caught-up. Its
// criticality state (task{1,2}_err/marg, task{1,2}_critical) simply stays stale
// -- last-known-value -- until it next gets cluster time, mirroring a real
// preemptive scheduler where a starved task's last observation ages. Any scenario
// input sampled on a tick where its task doesn't run is genuinely dropped, not
// queued: the task that runs always processes the CURRENT tick's fresh sample, not
// a backlog.
//
// This deliberately breaks the "both tasks run every inference tick" invariant
// every other method in this experiment (including Round-Robin-Large) relies on --
// DELTA_T ticks no longer means DELTA_T inferences per task, so per-tick theta is
// attributed only to whichever single task actually ran (see
// parse_priority_sequential_log.py).

#define TASK1_NUM_CLASSES 2
#define TASK2_NUM_CLASSES 10

typedef enum { NONE_RUNNING, TASK2_RUNNING, TASK1_RUNNING } RunningTask;

static int8_t task1_tick_buf[TASK1_INPUT_SIZE];
static int8_t task2_tick_buf[TASK2_INPUT_SIZE];

static int8_t *task1_scenario_ram;
static int8_t *task2_scenario_ram;

// See DynSiracusa's dynmultideeploytest_template.c for why this must be
// cluster-dispatched: load_file_to_ram() is internally built on cluster-only
// primitives (pi_cl_fs_read()/cl_ram_write()).
static void LoadScenarioData(void *args) {
    if (pi_core_id() == 0) {
        task1_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK1_INPUT_SIZE);
        load_file_to_ram(task1_scenario_ram, TASK1_SCENARIO_HEX_FILE);
        task2_scenario_ram = (int8_t *) cl_ram_malloc(DELTA_T * TASK2_INPUT_SIZE);
        load_file_to_ram(task2_scenario_ram, TASK2_SCENARIO_HEX_FILE);
    }
}

// Copied verbatim from DynSiracusa's dynmultideeploytest_template.c (same
// criticality signal, same rationale -- see that file's comments). Task 1
// (AnomalyDetection) is a 2-class classifier (critical vs normal), not an
// autoencoder, so both tasks share this same top-vs-runner-up margin proxy.
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

    struct pi_cluster_task cluster_task;

    pi_cluster_task(&cluster_task, LoadScenarioData, NULL);
    cluster_task.stack_size = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    // Last-known criticality state, persisting across ticks where a task doesn't
    // run (stale until that task next gets cluster time -- see file header).
    int task1_critical = 0;
    int task2_critical = 0;
    int32_t task1_marg = 0;
    int32_t task2_marg = 0;
    RunningTask previous_running = NONE_RUNNING;

    for (int t = 0; t < DELTA_T; t++) {

        ram_read(task1_tick_buf, task1_scenario_ram + (uint32_t) t * TASK1_INPUT_SIZE, TASK1_INPUT_SIZE);
        ram_read(task2_tick_buf, task2_scenario_ram + (uint32_t) t * TASK2_INPUT_SIZE, TASK2_INPUT_SIZE);

        // Priority order: both critical -> Task 1 (AnomalyDetection) wins the tie
        // (arbitrary but fixed/documented choice -- anomaly detection treated as
        // the safety-relevant task); one critical -> that one runs; neither
        // critical -> fair round-robin fallback so a persistently-non-critical
        // task still gets refreshed periodically instead of starving forever.
        RunningTask running;
        if (task1_critical) {
            running = TASK1_RUNNING;
        } else if (task2_critical) {
            running = TASK2_RUNNING;
        } else {
            running = (previous_running == TASK1_RUNNING) ? TASK2_RUNNING : TASK1_RUNNING;
        }

        int task_changed = (running != previous_running);

        printf("[HOST CORE] Tick %d: running %d (task1_critical=%d, task2_critical=%d)\r\n",
               t, running, task1_critical, task2_critical);

        if (task_changed && previous_running != NONE_RUNNING) {
            // Free whichever model was previously resident -- only one model's
            // arena is ever resident at a time (neither fits L1 alongside the
            // other), matching Round-Robin-Large's memory model.
            if (previous_running == TASK2_RUNNING) {
                pmsis_l1_malloc_free(VisualWakeWords_heavy_MEMORYARENA_L1, VisualWakeWords_heavy_MEMORYARENA_L1_len);
                pi_l2_free(VisualWakeWords_heavy_MEMORYARENA_L2, VisualWakeWords_heavy_MEMORYARENA_L2_len);
            } else {
                pmsis_l1_malloc_free(AnomalyDetection_heavy_MEMORYARENA_L1, AnomalyDetection_heavy_MEMORYARENA_L1_len);
                pi_l2_free(AnomalyDetection_heavy_MEMORYARENA_L2, AnomalyDetection_heavy_MEMORYARENA_L2_len);
            }
        }

        if (running == TASK2_RUNNING) {
            if (task_changed) {
                printf("[HOST CORE] Sending InitNetwork_VisualWakeWords_heavy() cluster task\r\n");
                pi_cluster_task(&cluster_task, InitNetwork_VisualWakeWords_heavy, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                printf("[TASK2] Runtime InitNetwork: %u cycles\r\n", getCycles());
            } else {
                printf("[TASK2] Skipped InitNetwork (same task as previous tick)\r\n");
            }

            for (uint32_t buf = 0; buf < VisualWakeWords_heavy_num_inputs; buf++) {
                if ((uint32_t) VisualWakeWords_heavy_inputs[buf] >= 0x10000000) {
                    memcpy(VisualWakeWords_heavy_inputs[buf], task2_tick_buf, VisualWakeWords_heavy_inputs_bytes[buf]);
                }
            }

            printf("[HOST CORE] Sending RunNetwork_VisualWakeWords_heavy() cluster task\r\n");
            pi_cluster_task(&cluster_task, RunNetwork_VisualWakeWords_heavy, NULL);
            cluster_task.stack_size = MAINSTACKSIZE;
            cluster_task.slave_stack_size = SLAVESTACKSIZE;
            *(volatile int *)0x10000000 = 0xabbaabba;
            ResetTimer();
            StartTimer();
            pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
            StopTimer();
            *(volatile int *)0x10000000 = 0xdeadcaca;
            printf("[TASK2] Runtime RunNetwork: %u cycles\r\n", getCycles());

            task2_marg = compute_margin(VisualWakeWords_heavy_output_0, TASK2_NUM_CLASSES);
            task2_critical = (task2_marg < TASK2_MARGIN_THRESHOLD);
        } else {
            if (task_changed) {
                printf("[HOST CORE] Sending InitNetwork_AnomalyDetection_heavy() cluster task\r\n");
                pi_cluster_task(&cluster_task, InitNetwork_AnomalyDetection_heavy, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                printf("[TASK1] Runtime InitNetwork: %u cycles\r\n", getCycles());
            } else {
                printf("[TASK1] Skipped InitNetwork (same task as previous tick)\r\n");
            }

            for (uint32_t buf = 0; buf < AnomalyDetection_heavy_num_inputs; buf++) {
                if ((uint32_t) AnomalyDetection_heavy_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_heavy_inputs[buf], task1_tick_buf, AnomalyDetection_heavy_inputs_bytes[buf]);
                }
            }

            printf("[HOST CORE] Sending RunNetwork_AnomalyDetection_heavy() cluster task\r\n");
            pi_cluster_task(&cluster_task, RunNetwork_AnomalyDetection_heavy, NULL);
            cluster_task.stack_size = MAINSTACKSIZE;
            cluster_task.slave_stack_size = SLAVESTACKSIZE;
            *(volatile int *)0x10000000 = 0xabbaabba;
            ResetTimer();
            StartTimer();
            pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
            StopTimer();
            *(volatile int *)0x10000000 = 0xdeadcaca;
            printf("[TASK1] Runtime RunNetwork: %u cycles\r\n", getCycles());

            task1_marg = compute_margin(AnomalyDetection_heavy_output_0, TASK1_NUM_CLASSES);
            task1_critical = (task1_marg < TASK1_MARGIN_THRESHOLD_HEAVY);
        }

        printf("[HOST CORE] Tick %d context: task1_margin=%d task2_margin=%d\r\n", t, task1_marg, task2_marg);

        previous_running = running;
    }

    return 0;
}
