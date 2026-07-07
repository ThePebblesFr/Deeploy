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
    // Copy Model_1 input data to L2
    for (uint32_t buf = 0; buf < Model_1_2_num_inputs; buf++) {
        if ((uint32_t) Model_1_2_inputs[buf] >= 0x10000000) {
        memcpy(Model_1_2_inputs[buf], Model_1_testInputVector[buf],
                Model_1_2_inputs_bytes[buf]);
        }
    }

    // Copy Model_2 input data to L2
    for (uint32_t buf = 0; buf < Model_2_6_num_inputs; buf++) {
        if ((uint32_t) Model_2_6_inputs[buf] >= 0x10000000) {
        memcpy(Model_2_6_inputs[buf], Model_2_testInputVector[buf],
                Model_2_6_inputs_bytes[buf]);
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
    uint32_t Model_1_tot_tested, Model_2_tot_tested;

    OutputVerificationArgs Model_1_verification_args = {
        .model_name = "Model_1",
        .num_outputs = Model_1_2_num_outputs,
        .outputs = Model_1_2_outputs,
        .outputs_bytes = Model_1_2_outputs_bytes,
        .testOutputVector = Model_1_testOutputVector,
        .output_elem_size = sizeof(MODEL_1_OUTPUTTYPE),
        .is_output_float = MODEL_1_ISOUTPUTFLOAT,
    };
    uint32_t Model_1_tot_err = VerifyModelOutput(&Model_1_verification_args, &Model_1_tot_tested);

    OutputVerificationArgs Model_2_verification_args = {
        .model_name = "Model_2",
        .num_outputs = Model_2_6_num_outputs,
        .outputs = Model_2_6_outputs,
        .outputs_bytes = Model_2_6_outputs_bytes,
        .testOutputVector = Model_2_testOutputVector,
        .output_elem_size = sizeof(MODEL_2_OUTPUTTYPE),
        .is_output_float = MODEL_2_ISOUTPUTFLOAT,
    };
    uint32_t Model_2_tot_err = VerifyModelOutput(&Model_2_verification_args, &Model_2_tot_tested);

    printf("Runtime: %u cycles\r\n", getCycles());
    printf("[Model_1] Errors: %u out of %u \r\n", Model_1_tot_err, Model_1_tot_tested);
    printf("[Model_2] Errors: %u out of %u \r\n", Model_2_tot_err, Model_2_tot_tested);

    if (Model_1_tot_err == 0 && Model_2_tot_err == 0) {
        return 0;
    }
    return Model_1_tot_err + Model_2_tot_err;
}