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
    }

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
  // Copy miniMNV2_6 input data to L2
  for (uint32_t buf = 0; buf < miniMNV2_6_num_inputs; buf++) {
    if ((uint32_t) miniMNV2_6_inputs[buf] >= 0x10000000) {
      memcpy(miniMNV2_6_inputs[buf], miniMNV2_6_testInputVector[buf],
             miniMNV2_6_inputs_bytes[buf]);
    }
  }

  // Copy AnomalyDetection_2 input data to L2
  for (uint32_t buf = 0; buf < AnomalyDetection_2_num_inputs; buf++) {
    if ((uint32_t) AnomalyDetection_2_inputs[buf] >= 0x10000000) {
      memcpy(AnomalyDetection_2_inputs[buf], AnomalyDetection_2_testInputVector[buf],
             AnomalyDetection_2_inputs_bytes[buf]);
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
  // miniMNV2_6 Output Verification
  // -------------------------------------------------------------------
  uint32_t miniMNV2_6_tot_err, miniMNV2_6_tot_tested;
  miniMNV2_6_tot_err = 0;
  miniMNV2_6_tot_tested = 0;
  void *miniMNV2_6_compbuf;
  FloatCompareArgs miniMNV2_6_float_compare_args;
  uint32_t miniMNV2_6_float_error_count = 0;

  for (uint32_t buf = 0; buf < miniMNV2_6_num_outputs; buf++) {
    miniMNV2_6_tot_tested += miniMNV2_6_outputs_bytes[buf] / sizeof(MINIMNV2_6_OUTPUTTYPE);

    if ((uint32_t) miniMNV2_6_outputs[buf] < 0x1000000) {
      miniMNV2_6_compbuf = pi_l2_malloc((int) miniMNV2_6_outputs_bytes[buf]);
      ram_read(miniMNV2_6_compbuf, miniMNV2_6_outputs[buf],
               miniMNV2_6_outputs_bytes[buf]);
    } else {
      miniMNV2_6_compbuf = miniMNV2_6_outputs[buf];
    }

    if (MINIMNV2_6_ISOUTPUTFLOAT) {
      miniMNV2_6_float_error_count = 0;
      miniMNV2_6_float_compare_args.expected = miniMNV2_6_testOutputVector[buf];
      miniMNV2_6_float_compare_args.actual = miniMNV2_6_compbuf;
      miniMNV2_6_float_compare_args.num_elements =
          miniMNV2_6_outputs_bytes[buf] / sizeof(float);
      miniMNV2_6_float_compare_args.output_buf_index = buf;
      miniMNV2_6_float_compare_args.err_count = &miniMNV2_6_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &miniMNV2_6_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      miniMNV2_6_tot_err += miniMNV2_6_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < miniMNV2_6_outputs_bytes[buf] / sizeof(MINIMNV2_6_OUTPUTTYPE); i++) {
        MINIMNV2_6_OUTPUTTYPE expected = ((MINIMNV2_6_OUTPUTTYPE *)miniMNV2_6_testOutputVector[buf])[i];
        MINIMNV2_6_OUTPUTTYPE actual = ((MINIMNV2_6_OUTPUTTYPE *)miniMNV2_6_compbuf)[i];
        int32_t error = expected - actual;
        MINIMNV2_6_OUTPUTTYPE diff = (MINIMNV2_6_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          miniMNV2_6_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) miniMNV2_6_outputs[buf] < 0x1000000) {
      pi_l2_free(miniMNV2_6_compbuf, (int) miniMNV2_6_outputs_bytes[buf]);
    }
  }

  // -------------------------------------------------------------------
  // AnomalyDetection_2 Output Verification
  // -------------------------------------------------------------------
  uint32_t AnomalyDetection_2_tot_err, AnomalyDetection_2_tot_tested;
  AnomalyDetection_2_tot_err = 0;
  AnomalyDetection_2_tot_tested = 0;
  void *AnomalyDetection_2_compbuf;
  FloatCompareArgs AnomalyDetection_2_float_compare_args;
  uint32_t AnomalyDetection_2_float_error_count = 0;

  for (uint32_t buf = 0; buf < AnomalyDetection_2_num_outputs; buf++) {
    AnomalyDetection_2_tot_tested += AnomalyDetection_2_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_2_OUTPUTTYPE);

    if ((uint32_t) AnomalyDetection_2_outputs[buf] < 0x1000000) {
      AnomalyDetection_2_compbuf = pi_l2_malloc((int) AnomalyDetection_2_outputs_bytes[buf]);
      ram_read(AnomalyDetection_2_compbuf, AnomalyDetection_2_outputs[buf],
               AnomalyDetection_2_outputs_bytes[buf]);
    } else {
      AnomalyDetection_2_compbuf = AnomalyDetection_2_outputs[buf];
    }

    if (ANOMALYDETECTION_2_ISOUTPUTFLOAT) {
      AnomalyDetection_2_float_error_count = 0;
      AnomalyDetection_2_float_compare_args.expected = AnomalyDetection_2_testOutputVector[buf];
      AnomalyDetection_2_float_compare_args.actual = AnomalyDetection_2_compbuf;
      AnomalyDetection_2_float_compare_args.num_elements =
          AnomalyDetection_2_outputs_bytes[buf] / sizeof(float);
      AnomalyDetection_2_float_compare_args.output_buf_index = buf;
      AnomalyDetection_2_float_compare_args.err_count = &AnomalyDetection_2_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &AnomalyDetection_2_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      AnomalyDetection_2_tot_err += AnomalyDetection_2_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < AnomalyDetection_2_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_2_OUTPUTTYPE); i++) {
        ANOMALYDETECTION_2_OUTPUTTYPE expected = ((ANOMALYDETECTION_2_OUTPUTTYPE *)AnomalyDetection_2_testOutputVector[buf])[i];
        ANOMALYDETECTION_2_OUTPUTTYPE actual = ((ANOMALYDETECTION_2_OUTPUTTYPE *)AnomalyDetection_2_compbuf)[i];
        int32_t error = expected - actual;
        ANOMALYDETECTION_2_OUTPUTTYPE diff = (ANOMALYDETECTION_2_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          AnomalyDetection_2_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) AnomalyDetection_2_outputs[buf] < 0x1000000) {
      pi_l2_free(AnomalyDetection_2_compbuf, (int) AnomalyDetection_2_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());


  printf("[miniMNV2_6] Errors: %u out of %u \r\n", miniMNV2_6_tot_err, miniMNV2_6_tot_tested);
  printf("[AnomalyDetection_2] Errors: %u out of %u \r\n", AnomalyDetection_2_tot_err, AnomalyDetection_2_tot_tested);

  if (miniMNV2_6_tot_err == 0 && AnomalyDetection_2_tot_err == 0) {
    return 0;
  } else {
    return miniMNV2_6_tot_err + AnomalyDetection_2_tot_err;
  }
  // return miniMNV2_6_tot_err;
}