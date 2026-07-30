#include "CycleCounter.h"
#include "OutputVerification.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

typedef enum {
    IDLE,
    LOW_ENERGY,
    CRITICAL_M1,
    CRITICAL_M2,
    BOTH_CRITICAL,
    DONE
} OperationMode;

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
    int nb_inferences = 0;
    struct pi_cluster_task cluster_task;

    while (operation_mode != DONE) {

        printf("[HOST CORE] Current operation mode: %d\r\n", operation_mode);
        printf("[HOST CORE] Number of inferences: %d\r\n", nb_inferences);

        // Flush the arenas allocated by the previous iteration's InitNetworks_X_Y():
        // it always mallocs fresh arenas without freeing the old ones, so calling it
        // again every iteration would otherwise leak L1/L2 memory each time around.
        switch (previous_operation_mode) {
            case LOW_ENERGY:
                pmsis_l1_malloc_free(miniMNV2_4_MEMORYARENA_L1, miniMNV2_4_MEMORYARENA_L1_len);
                pi_l2_free(miniMNV2_4_MEMORYARENA_L2, miniMNV2_4_MEMORYARENA_L2_len);
                pmsis_l1_malloc_free(AnomalyDetection_4_MEMORYARENA_L1, AnomalyDetection_4_MEMORYARENA_L1_len);
                pi_l2_free(AnomalyDetection_4_MEMORYARENA_L2, AnomalyDetection_4_MEMORYARENA_L2_len);
                break;
            case CRITICAL_M1:
                pmsis_l1_malloc_free(miniMNV2_6_MEMORYARENA_L1, miniMNV2_6_MEMORYARENA_L1_len);
                pi_l2_free(miniMNV2_6_MEMORYARENA_L2, miniMNV2_6_MEMORYARENA_L2_len);
                pmsis_l1_malloc_free(AnomalyDetection_2_MEMORYARENA_L1, AnomalyDetection_2_MEMORYARENA_L1_len);
                pi_l2_free(AnomalyDetection_2_MEMORYARENA_L2, AnomalyDetection_2_MEMORYARENA_L2_len);
                break;
            case CRITICAL_M2:
                pmsis_l1_malloc_free(miniMNV2_2_MEMORYARENA_L1, miniMNV2_2_MEMORYARENA_L1_len);
                pi_l2_free(miniMNV2_2_MEMORYARENA_L2, miniMNV2_2_MEMORYARENA_L2_len);
                pmsis_l1_malloc_free(AnomalyDetection_6_MEMORYARENA_L1, AnomalyDetection_6_MEMORYARENA_L1_len);
                pi_l2_free(AnomalyDetection_6_MEMORYARENA_L2, AnomalyDetection_6_MEMORYARENA_L2_len);
                break;
            default:
                break;
        }
        previous_operation_mode = operation_mode;

        switch (operation_mode) {
            case IDLE:
                break;
            case LOW_ENERGY:
                // send InitNetworks_4_4() cluster task
                printf("[HOST CORE] Sending InitNetworks_4_4() cluster task\r\n");
                pi_cluster_task(&cluster_task, InitNetworks_4_4, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                for (uint32_t buf = 0; buf < miniMNV2_4_num_inputs; buf++) {
                    if ((uint32_t) miniMNV2_4_inputs[buf] >= 0x10000000) {
                    memcpy(miniMNV2_4_inputs[buf], miniMNV2_testInputVector[buf],
                            miniMNV2_4_inputs_bytes[buf]);
                    }
                }
                for (uint32_t buf = 0; buf < AnomalyDetection_4_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_4_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_4_inputs[buf], AnomalyDetection_testInputVector[buf],
                            AnomalyDetection_4_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_4_4() cluster task
                printf("[HOST CORE] Sending RunNetworks_4_4() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_4_4, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                printf("Runtime RunNetwork: %u cycles\r\n", getCycles());
                nb_inferences++;
                break;
            case CRITICAL_M1:
                // send InitNetworks_6_2() cluster task
                printf("[HOST CORE] Sending InitNetworks_6_2() cluster task\r\n");
                pi_cluster_task(&cluster_task, InitNetworks_6_2, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                for (uint32_t buf = 0; buf < miniMNV2_6_num_inputs; buf++) {
                    if ((uint32_t) miniMNV2_6_inputs[buf] >= 0x10000000) {
                    memcpy(miniMNV2_6_inputs[buf], miniMNV2_testInputVector[buf],
                            miniMNV2_6_inputs_bytes[buf]);
                    }
                }
                for (uint32_t buf = 0; buf < AnomalyDetection_2_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_2_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_2_inputs[buf], AnomalyDetection_testInputVector[buf],
                            AnomalyDetection_2_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_6_2() cluster task
                printf("[HOST CORE] Sending RunNetworks_6_2() cluster task\r\n");
                pi_cluster_task(&cluster_task, RunNetworks_6_2, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                *(volatile int *)0x10000000 = 0xabbaabba;
                ResetTimer();
                StartTimer();
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                StopTimer();
                *(volatile int *)0x10000000 = 0xdeadcaca;
                printf("Runtime RunNetwork: %u cycles\r\n", getCycles());
                nb_inferences++;
                break;
            case CRITICAL_M2:
                // send InitNetworks_2_6() cluster task
                printf("[HOST CORE] Sending InitNetworks_2_6() cluster task\r\n");
                pi_cluster_task(&cluster_task, InitNetworks_2_6, NULL);
                cluster_task.stack_size = MAINSTACKSIZE;
                cluster_task.slave_stack_size = SLAVESTACKSIZE;
                pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                for (uint32_t buf = 0; buf < miniMNV2_2_num_inputs; buf++) {
                    if ((uint32_t) miniMNV2_2_inputs[buf] >= 0x10000000) {
                    memcpy(miniMNV2_2_inputs[buf], miniMNV2_testInputVector[buf],
                            miniMNV2_2_inputs_bytes[buf]);
                    }
                }
                for (uint32_t buf = 0; buf < AnomalyDetection_6_num_inputs; buf++) {
                    if ((uint32_t) AnomalyDetection_6_inputs[buf] >= 0x10000000) {
                    memcpy(AnomalyDetection_6_inputs[buf], AnomalyDetection_testInputVector[buf],
                            AnomalyDetection_6_inputs_bytes[buf]);
                    }
                }
                // send RunNetworks_2_6() cluster task
                printf("[HOST CORE] Sending RunNetworks_2_6() cluster task\r\n");
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
                nb_inferences++;
                break;
            case BOTH_CRITICAL:
                // No dedicated hardware configuration is generated for this mode yet
                nb_inferences++;
                break;
            default:
                break;
        }

        if (nb_inferences == 5) {
            printf("[HOST CORE] Reached %d inferences, setting operation mode to CRITICAL_M1\r\n", nb_inferences);
            operation_mode = CRITICAL_M1;
        }
        if (nb_inferences == 7) {
            printf("[HOST CORE] Reached %d inferences, setting operation mode to CRITICAL_M2\r\n", nb_inferences);
            operation_mode = CRITICAL_M2;
        }
        if (nb_inferences >= 10) {
            printf("[HOST CORE] Reached %d inferences, setting operation mode to DONE\r\n", nb_inferences);
            operation_mode = DONE;
        }
    }

    return 0;
}