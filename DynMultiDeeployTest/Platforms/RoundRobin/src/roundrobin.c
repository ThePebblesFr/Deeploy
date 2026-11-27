#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "scenario_data.h"

// VisualWakeWords_heavy/AnomalyDetection_heavy are literal placeholders substituted by
// testUtils/UpdateRoundRobinMain.py -- e.g. TinyClassifier_heavy/
// AnomalyDetection_heavy for Round-Robin-Large. Unlike DynSiracusa's fused binary,
// neither model is ever co-resident with the other: each runs alone on the full
// 8-core cluster, one after the other, every tick. Both models' InitNetwork_*/
// weights/arena are therefore torn down and rebuilt on EVERY tick (never skipped
// the way DynSiracusa's mode_changed guard skips same-mode InitNetworks_* calls) --
// that per-switch reload cost is exactly the expensive behavior the paper's
// DynReconf design is meant to avoid, so it must be paid here, not optimized away.

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

    for (int t = 0; t < DELTA_T; t++) {

        ram_read(task1_tick_buf, task1_scenario_ram + (uint32_t) t * TASK1_INPUT_SIZE, TASK1_INPUT_SIZE);
        ram_read(task2_tick_buf, task2_scenario_ram + (uint32_t) t * TASK2_INPUT_SIZE, TASK2_INPUT_SIZE);

        printf("[HOST CORE] Tick %d\r\n", t);

        // ===== Task 2 (VisualWakeWords_heavy), full cluster =====
        printf("[HOST CORE] Sending InitNetwork_VisualWakeWords_heavy() cluster task\r\n");
        pi_cluster_task(&cluster_task, InitNetwork_VisualWakeWords_heavy, NULL);
        cluster_task.stack_size = MAINSTACKSIZE;
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        // Unlike DynSiracusa/Static (where InitNetworks_* runs once, a negligible
        // one-time cost against DELTA_T RunNetworks_* calls), Round-Robin re-runs
        // InitNetwork every tick for BOTH models -- its power/energy is NOT
        // negligible here, so it gets its own @power.measure_N@ bracket too,
        // unlike every other template in this codebase which only brackets
        // RunNetwork(s)_*.
        *(volatile int *)0x10000000 = 0xabbaabba;
        ResetTimer();
        StartTimer();
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
        StopTimer();
        *(volatile int *)0x10000000 = 0xdeadcaca;
        printf("[TASK2] Runtime InitNetwork: %u cycles\r\n", getCycles());

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

        pmsis_l1_malloc_free(VisualWakeWords_heavy_MEMORYARENA_L1, VisualWakeWords_heavy_MEMORYARENA_L1_len);
        pi_l2_free(VisualWakeWords_heavy_MEMORYARENA_L2, VisualWakeWords_heavy_MEMORYARENA_L2_len);

        // ===== Task 1 (AnomalyDetection_heavy), full cluster =====
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

        pmsis_l1_malloc_free(AnomalyDetection_heavy_MEMORYARENA_L1, AnomalyDetection_heavy_MEMORYARENA_L1_len);
        pi_l2_free(AnomalyDetection_heavy_MEMORYARENA_L2, AnomalyDetection_heavy_MEMORYARENA_L2_len);
    }

    return 0;
}
