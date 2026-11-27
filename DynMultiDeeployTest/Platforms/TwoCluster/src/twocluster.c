#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "scenario_data.h"

// Exploratory experiment: does GVSoC's Siracusa target genuinely support running
// two INDEPENDENT clusters concurrently, each with its own dedicated L1/cores,
// versus MultiDeeploy's existing approach of splitting ONE cluster's 8 cores into
// two groups sharing the same L1/MCHAN? nb_cluster was bumped from 1 to 2 in
// pulp_open.py (a live, in-container GVSoC patch, not a repo change) to make a
// second cluster instance exist at all -- see the plan file for the full
// investigation trail. This reuses AnomalyDetection_light/_light2's Network.c/.h
// AS-IS (produced by RoundRobinFusion, i.e. plain rename+concatenate, no
// MultiDeeploy core-split fusion involved at all) -- each network's own
// InitNetwork_*/RunNetwork_* is dispatched, unmodified, to its OWN cluster device
// (cluster_dev = id 0, cluster_dev2 = id 1) via the async cluster-task API, so
// both run genuinely concurrently with zero shared L1/MCHAN contention.

static struct pi_device cluster_dev2;

static int8_t task1_tick_buf[TASK1_INPUT_SIZE];
static int8_t task2_tick_buf[TASK2_INPUT_SIZE];

static int8_t *task1_scenario_ram;
static int8_t *task2_scenario_ram;

// load_file_to_ram() is internally built on cluster-only primitives
// (pi_cl_fs_read()/cl_ram_write()), same as every other template in this
// codebase -- dispatched to cluster 0 only, since HyperRAM/flash access doesn't
// need to happen per-cluster.
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

    struct pi_cluster_conf conf0;
    pi_cluster_conf_init(&conf0);
    conf0.id = 0;
    pi_open_from_conf(&cluster_dev, &conf0);
    if (pi_cluster_open(&cluster_dev))
        return -1;

    struct pi_cluster_conf conf1;
    pi_cluster_conf_init(&conf1);
    conf1.id = 1;
    pi_open_from_conf(&cluster_dev2, &conf1);
    if (pi_cluster_open(&cluster_dev2))
        return -2;

    mem_init();

    #ifndef NOFLASH
    open_fs();
    #endif

    printf("Intializing\r\n");

    struct pi_cluster_task load_task;
    pi_cluster_task(&load_task, LoadScenarioData, NULL);
    load_task.stack_size = MAINSTACKSIZE;
    load_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &load_task);

    ram_read(task1_tick_buf, task1_scenario_ram, TASK1_INPUT_SIZE);
    ram_read(task2_tick_buf, task2_scenario_ram, TASK2_INPUT_SIZE);

    // ===== InitNetwork on both clusters concurrently =====
    struct pi_cluster_task init_task0, init_task1;
    pi_cluster_task(&init_task0, InitNetwork_AnomalyDetection_light, NULL);
    init_task0.stack_size = MAINSTACKSIZE;
    init_task0.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_task(&init_task1, InitNetwork_AnomalyDetection_light2, NULL);
    init_task1.stack_size = MAINSTACKSIZE;
    init_task1.slave_stack_size = SLAVESTACKSIZE;

    pi_task_t initEvt0, initEvt1;
    pi_task_block(&initEvt0);
    pi_task_block(&initEvt1);
    pi_cluster_send_task_to_cl_async(&cluster_dev, &init_task0, &initEvt0);
    pi_cluster_send_task_to_cl_async(&cluster_dev2, &init_task1, &initEvt1);
    pi_task_wait_on(&initEvt0);
    pi_task_wait_on(&initEvt1);
    printf("[HOST CORE] Both InitNetworks done (2 clusters)\r\n");

    for (uint32_t buf = 0; buf < AnomalyDetection_light_num_inputs; buf++) {
        if ((uint32_t) AnomalyDetection_light_inputs[buf] >= 0x10000000) {
            memcpy(AnomalyDetection_light_inputs[buf], task2_tick_buf, AnomalyDetection_light_inputs_bytes[buf]);
        }
    }
    for (uint32_t buf = 0; buf < AnomalyDetection_light2_num_inputs; buf++) {
        if ((uint32_t) AnomalyDetection_light2_inputs[buf] >= 0x10000000) {
            memcpy(AnomalyDetection_light2_inputs[buf], task1_tick_buf, AnomalyDetection_light2_inputs_bytes[buf]);
        }
    }

    // ===== RunNetwork on both clusters concurrently, power-bracketed =====
    struct pi_cluster_task run_task0, run_task1;
    pi_cluster_task(&run_task0, RunNetwork_AnomalyDetection_light, NULL);
    run_task0.stack_size = MAINSTACKSIZE;
    run_task0.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_task(&run_task1, RunNetwork_AnomalyDetection_light2, NULL);
    run_task1.stack_size = MAINSTACKSIZE;
    run_task1.slave_stack_size = SLAVESTACKSIZE;

    pi_task_t runEvt0, runEvt1;
    pi_task_block(&runEvt0);
    pi_task_block(&runEvt1);

    *(volatile int *)0x10000000 = 0xabbaabba;
    ResetTimer();
    StartTimer();
    pi_cluster_send_task_to_cl_async(&cluster_dev, &run_task0, &runEvt0);
    pi_cluster_send_task_to_cl_async(&cluster_dev2, &run_task1, &runEvt1);
    pi_task_wait_on(&runEvt0);
    pi_task_wait_on(&runEvt1);
    StopTimer();
    *(volatile int *)0x10000000 = 0xdeadcaca;
    printf("[HOST CORE] Both RunNetworks done (2 independent clusters), cycles: %u\r\n", getCycles());

    return 0;
}
