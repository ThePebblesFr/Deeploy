/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

#define MAINSTACKSIZE 8000
#define SLAVESTACKSIZE 3800

struct pi_device cluster_dev;

typedef struct {
  void *expected;
  void *actual;
  uint32_t num_elements;
  uint32_t output_buf_index;
  uint32_t *err_count;
} FloatCompareArgs;

void CompareFloatOnCluster(void *args) {

  if (pi_core_id() == 0) {
    FloatCompareArgs *compare_args = (FloatCompareArgs *)args;
    float *expected = (float *)compare_args->expected;
    float *actual = (float *)compare_args->actual;
    uint32_t num_elements = compare_args->num_elements;
    uint32_t output_buf_index = compare_args->output_buf_index;
    uint32_t *err_count = compare_args->err_count;

    uint32_t local_err_count = 0;

    for (uint32_t i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;

      if ((diff < -1e-4) || (diff > 1e-4) || isnan(diff)) {
        local_err_count += 1;

        printf("Expected: %10.6f  ", expected_val);
        printf("Actual: %10.6f  ", actual_val);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", diff, i,
               output_buf_index);
      }
    }, models_name[i]

    *err_count = local_err_count;
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

  pi_cluster_task(&cluster_task, InitNetwork, NULL);
  cluster_task.stack_size = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifndef CI
  printf("Initialized\r\n");
#endif
  // Copy miniMNV2_2 input data to L2
  for (uint32_t buf = 0; buf < miniMNV2_2_num_inputs; buf++) {
    if ((uint32_t) miniMNV2_2_inputs[buf] >= 0x10000000) {
      memcpy(miniMNV2_2_inputs[buf], miniMNV2_2_testInputVector[buf],
             miniMNV2_2_inputs_bytes[buf]);
    }
  }

  // Copy AnomalyDetection_6 input data to L2
  for (uint32_t buf = 0; buf < AnomalyDetection_6_num_inputs; buf++) {
    if ((uint32_t) AnomalyDetection_6_inputs[buf] >= 0x10000000) {
      memcpy(AnomalyDetection_6_inputs[buf], AnomalyDetection_6_testInputVector[buf],
             AnomalyDetection_6_inputs_bytes[buf]);
    }
  }

#ifndef CI
  printf("Input copied\r\n");
#endif

  pi_cluster_task(&cluster_task, RunNetwork, NULL);
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

  // -------------------------------------------------------------------
  // miniMNV2_2 Output Verification
  // -------------------------------------------------------------------
  uint32_t miniMNV2_2_tot_err, miniMNV2_2_tot_tested;
  miniMNV2_2_tot_err = 0;
  miniMNV2_2_tot_tested = 0;
  void *miniMNV2_2_compbuf;
  FloatCompareArgs miniMNV2_2_float_compare_args;
  uint32_t miniMNV2_2_float_error_count = 0;

  for (uint32_t buf = 0; buf < miniMNV2_2_num_outputs; buf++) {
    miniMNV2_2_tot_tested += miniMNV2_2_outputs_bytes[buf] / sizeof(MINIMNV2_2_OUTPUTTYPE);

    if ((uint32_t) miniMNV2_2_outputs[buf] < 0x1000000) {
      miniMNV2_2_compbuf = pi_l2_malloc((int) miniMNV2_2_outputs_bytes[buf]);
      ram_read(miniMNV2_2_compbuf, miniMNV2_2_outputs[buf],
               miniMNV2_2_outputs_bytes[buf]);
    } else {
      miniMNV2_2_compbuf = miniMNV2_2_outputs[buf];
    }

    if (MINIMNV2_2_ISOUTPUTFLOAT) {
      miniMNV2_2_float_error_count = 0;
      miniMNV2_2_float_compare_args.expected = miniMNV2_2_testOutputVector[buf];
      miniMNV2_2_float_compare_args.actual = miniMNV2_2_compbuf;
      miniMNV2_2_float_compare_args.num_elements =
          miniMNV2_2_outputs_bytes[buf] / sizeof(float);
      miniMNV2_2_float_compare_args.output_buf_index = buf;
      miniMNV2_2_float_compare_args.err_count = &miniMNV2_2_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &miniMNV2_2_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      miniMNV2_2_tot_err += miniMNV2_2_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < miniMNV2_2_outputs_bytes[buf] / sizeof(MINIMNV2_2_OUTPUTTYPE); i++) {
        MINIMNV2_2_OUTPUTTYPE expected = ((MINIMNV2_2_OUTPUTTYPE *)miniMNV2_2_testOutputVector[buf])[i];
        MINIMNV2_2_OUTPUTTYPE actual = ((MINIMNV2_2_OUTPUTTYPE *)miniMNV2_2_compbuf)[i];
        int32_t error = expected - actual;
        MINIMNV2_2_OUTPUTTYPE diff = (MINIMNV2_2_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          miniMNV2_2_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) miniMNV2_2_outputs[buf] < 0x1000000) {
      pi_l2_free(miniMNV2_2_compbuf, (int) miniMNV2_2_outputs_bytes[buf]);
    }
  }

  // -------------------------------------------------------------------
  // AnomalyDetection_6 Output Verification
  // -------------------------------------------------------------------
  uint32_t AnomalyDetection_6_tot_err, AnomalyDetection_6_tot_tested;
  AnomalyDetection_6_tot_err = 0;
  AnomalyDetection_6_tot_tested = 0;
  void *AnomalyDetection_6_compbuf;
  FloatCompareArgs AnomalyDetection_6_float_compare_args;
  uint32_t AnomalyDetection_6_float_error_count = 0;

  for (uint32_t buf = 0; buf < AnomalyDetection_6_num_outputs; buf++) {
    AnomalyDetection_6_tot_tested += AnomalyDetection_6_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_6_OUTPUTTYPE);

    if ((uint32_t) AnomalyDetection_6_outputs[buf] < 0x1000000) {
      AnomalyDetection_6_compbuf = pi_l2_malloc((int) AnomalyDetection_6_outputs_bytes[buf]);
      ram_read(AnomalyDetection_6_compbuf, AnomalyDetection_6_outputs[buf],
               AnomalyDetection_6_outputs_bytes[buf]);
    } else {
      AnomalyDetection_6_compbuf = AnomalyDetection_6_outputs[buf];
    }

    if (ANOMALYDETECTION_6_ISOUTPUTFLOAT) {
      AnomalyDetection_6_float_error_count = 0;
      AnomalyDetection_6_float_compare_args.expected = AnomalyDetection_6_testOutputVector[buf];
      AnomalyDetection_6_float_compare_args.actual = AnomalyDetection_6_compbuf;
      AnomalyDetection_6_float_compare_args.num_elements =
          AnomalyDetection_6_outputs_bytes[buf] / sizeof(float);
      AnomalyDetection_6_float_compare_args.output_buf_index = buf;
      AnomalyDetection_6_float_compare_args.err_count = &AnomalyDetection_6_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &AnomalyDetection_6_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      AnomalyDetection_6_tot_err += AnomalyDetection_6_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < AnomalyDetection_6_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_6_OUTPUTTYPE); i++) {
        ANOMALYDETECTION_6_OUTPUTTYPE expected = ((ANOMALYDETECTION_6_OUTPUTTYPE *)AnomalyDetection_6_testOutputVector[buf])[i];
        ANOMALYDETECTION_6_OUTPUTTYPE actual = ((ANOMALYDETECTION_6_OUTPUTTYPE *)AnomalyDetection_6_compbuf)[i];
        int32_t error = expected - actual;
        ANOMALYDETECTION_6_OUTPUTTYPE diff = (ANOMALYDETECTION_6_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          AnomalyDetection_6_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) AnomalyDetection_6_outputs[buf] < 0x1000000) {
      pi_l2_free(AnomalyDetection_6_compbuf, (int) AnomalyDetection_6_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());


  printf("[miniMNV2_2] Errors: %u out of %u \r\n", miniMNV2_2_tot_err, miniMNV2_2_tot_tested);
  printf("[AnomalyDetection_6] Errors: %u out of %u \r\n", AnomalyDetection_6_tot_err, AnomalyDetection_6_tot_tested);

  if (miniMNV2_2_tot_err == 0 && AnomalyDetection_6_tot_err == 0) {
    return 0;
  } else {
    return miniMNV2_2_tot_err + AnomalyDetection_6_tot_err;
  }
  // return miniMNV2_2_tot_err;
}