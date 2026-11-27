#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "scenario_data.h"

// MODEL1_TAG/MODEL2_TAG are literal placeholders substituted by
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

        // ===== Task 2 (MODEL1_TAG), full cluster =====
        printf("[HOST CORE] Sending InitNetwork_MODEL1_TAG() cluster task\r\n");
        pi_cluster_task(&cluster_task, InitNetwork_MODEL1_TAG, NULL);
        cluster_task.stack_size = MAINSTACKSIZE;
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        ResetTimer();
        StartTimer();
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
        StopTimer();
        printf("[TASK2] Runtime InitNetwork: %u cycles\r\n", getCycles());

        for (uint32_t buf = 0; buf < MODEL1_TAG_num_inputs; buf++) {
            if ((uint32_t) MODEL1_TAG_inputs[buf] >= 0x10000000) {
                memcpy(MODEL1_TAG_inputs[buf], task2_tick_buf, MODEL1_TAG_inputs_bytes[buf]);
            }
        }

        printf("[HOST CORE] Sending RunNetwork_MODEL1_TAG() cluster task\r\n");
        pi_cluster_task(&cluster_task, RunNetwork_MODEL1_TAG, NULL);
        cluster_task.stack_size = MAINSTACKSIZE;
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        *(volatile int *)0x10000000 = 0xabbaabba;
        ResetTimer();
        StartTimer();
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
        StopTimer();
        *(volatile int *)0x10000000 = 0xdeadcaca;
        printf("[TASK2] Runtime RunNetwork: %u cycles\r\n", getCycles());

        pmsis_l1_malloc_free(MODEL1_TAG_MEMORYARENA_L1, MODEL1_TAG_MEMORYARENA_L1_len);
        pi_l2_free(MODEL1_TAG_MEMORYARENA_L2, MODEL1_TAG_MEMORYARENA_L2_len);

        // ===== Task 1 (MODEL2_TAG), full cluster =====
        printf("[HOST CORE] Sending InitNetwork_MODEL2_TAG() cluster task\r\n");
        pi_cluster_task(&cluster_task, InitNetwork_MODEL2_TAG, NULL);
        cluster_task.stack_size = MAINSTACKSIZE;
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        ResetTimer();
        StartTimer();
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
        StopTimer();
        printf("[TASK1] Runtime InitNetwork: %u cycles\r\n", getCycles());

        for (uint32_t buf = 0; buf < MODEL2_TAG_num_inputs; buf++) {
            if ((uint32_t) MODEL2_TAG_inputs[buf] >= 0x10000000) {
                memcpy(MODEL2_TAG_inputs[buf], task1_tick_buf, MODEL2_TAG_inputs_bytes[buf]);
            }
        }

        printf("[HOST CORE] Sending RunNetwork_MODEL2_TAG() cluster task\r\n");
        pi_cluster_task(&cluster_task, RunNetwork_MODEL2_TAG, NULL);
        cluster_task.stack_size = MAINSTACKSIZE;
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        *(volatile int *)0x10000000 = 0xabbaabba;
        ResetTimer();
        StartTimer();
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
        StopTimer();
        *(volatile int *)0x10000000 = 0xdeadcaca;
        printf("[TASK1] Runtime RunNetwork: %u cycles\r\n", getCycles());

        pmsis_l1_malloc_free(MODEL2_TAG_MEMORYARENA_L1, MODEL2_TAG_MEMORYARENA_L1_len);
        pi_l2_free(MODEL2_TAG_MEMORYARENA_L2, MODEL2_TAG_MEMORYARENA_L2_len);
    }

    return 0;
}
