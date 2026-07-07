#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

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
    struct pi_cluster_task cluster_task;
    pi_cluster_task(&cluster_task, InitNetworks_2_6, NULL);
    cluster_task.stack_size = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    #ifndef CI
    printf("Initialized\r\n");
    #endif
    // Copy miniMNV2 input data to L2
    for (uint32_t buf = 0; buf < miniMNV2_2_num_inputs; buf++) {
        if ((uint32_t) miniMNV2_2_inputs[buf] >= 0x10000000) {
        memcpy(miniMNV2_2_inputs[buf], miniMNV2_testInputVector[buf],
                miniMNV2_2_inputs_bytes[buf]);
        }
    }

    // Copy AnomalyDetection input data to L2
    for (uint32_t buf = 0; buf < AnomalyDetection_6_num_inputs; buf++) {
        if ((uint32_t) AnomalyDetection_6_inputs[buf] >= 0x10000000) {
        memcpy(AnomalyDetection_6_inputs[buf], AnomalyDetection_testInputVector[buf],
                AnomalyDetection_6_inputs_bytes[buf]);
        }
    }

    #ifndef CI
    printf("Input copied\r\n");
    #endif

    pi_cluster_task(&cluster_task, RunNetworks_2_6, NULL);
    cluster_task.stack_size = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    *(volatile int *)0x10000000 = 0xabbaabba;
    ResetTimer();
    StartTimer();
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    StopTimer();
    *(volatile int *)0x10000000 = 0xdeadcaca;
    printf("Runtime RunNetwork: %u cycles\r\n", getCycles());

    #ifndef CI
    printf("Output:\r\n");
    #endif

    // OUTPUT VERIFICATION
    uint32_t miniMNV2_tot_tested, AnomalyDetection_tot_tested;

    OutputVerificationArgs miniMNV2_verification_args = {
        .model_name = "miniMNV2",
        .num_outputs = miniMNV2_2_num_outputs,
        .outputs = miniMNV2_2_outputs,
        .outputs_bytes = miniMNV2_2_outputs_bytes,
        .testOutputVector = miniMNV2_testOutputVector,
        .output_elem_size = sizeof(MINIMNV2_OUTPUTTYPE),
        .is_output_float = MINIMNV2_ISOUTPUTFLOAT,
    };
    uint32_t miniMNV2_tot_err = VerifyModelOutput(&miniMNV2_verification_args, &miniMNV2_tot_tested);

    OutputVerificationArgs AnomalyDetection_verification_args = {
        .model_name = "AnomalyDetection",
        .num_outputs = AnomalyDetection_6_num_outputs,
        .outputs = AnomalyDetection_6_outputs,
        .outputs_bytes = AnomalyDetection_6_outputs_bytes,
        .testOutputVector = AnomalyDetection_testOutputVector,
        .output_elem_size = sizeof(ANOMALYDETECTION_OUTPUTTYPE),
        .is_output_float = ANOMALYDETECTION_ISOUTPUTFLOAT,
    };
    uint32_t AnomalyDetection_tot_err = VerifyModelOutput(&AnomalyDetection_verification_args, &AnomalyDetection_tot_tested);

    printf("Runtime: %u cycles\r\n", getCycles());
    printf("[miniMNV2] Errors: %u out of %u \r\n", miniMNV2_tot_err, miniMNV2_tot_tested);
    printf("[AnomalyDetection] Errors: %u out of %u \r\n", AnomalyDetection_tot_err, AnomalyDetection_tot_tested);

    if (miniMNV2_tot_err == 0 && AnomalyDetection_tot_err == 0) {
        return 0;
    }
    return miniMNV2_tot_err + AnomalyDetection_tot_err;
}