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
  // Copy VisualWakeWords_heavy_4 input data to L2
  for (uint32_t buf = 0; buf < VisualWakeWords_heavy_4_num_inputs; buf++) {
    if ((uint32_t) VisualWakeWords_heavy_4_inputs[buf] >= 0x10000000) {
      memcpy(VisualWakeWords_heavy_4_inputs[buf], VisualWakeWords_heavy_4_testInputVector[buf],
             VisualWakeWords_heavy_4_inputs_bytes[buf]);
    }
  }

  // Copy AnomalyDetection_heavy_4 input data to L2
  for (uint32_t buf = 0; buf < AnomalyDetection_heavy_4_num_inputs; buf++) {
    if ((uint32_t) AnomalyDetection_heavy_4_inputs[buf] >= 0x10000000) {
      memcpy(AnomalyDetection_heavy_4_inputs[buf], AnomalyDetection_heavy_4_testInputVector[buf],
             AnomalyDetection_heavy_4_inputs_bytes[buf]);
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
  // VisualWakeWords_heavy_4 Output Verification
  // -------------------------------------------------------------------
  uint32_t VisualWakeWords_heavy_4_tot_err, VisualWakeWords_heavy_4_tot_tested;
  VisualWakeWords_heavy_4_tot_err = 0;
  VisualWakeWords_heavy_4_tot_tested = 0;
  void *VisualWakeWords_heavy_4_compbuf;
  FloatCompareArgs VisualWakeWords_heavy_4_float_compare_args;
  uint32_t VisualWakeWords_heavy_4_float_error_count = 0;

  for (uint32_t buf = 0; buf < VisualWakeWords_heavy_4_num_outputs; buf++) {
    VisualWakeWords_heavy_4_tot_tested += VisualWakeWords_heavy_4_outputs_bytes[buf] / sizeof(VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE);

    if ((uint32_t) VisualWakeWords_heavy_4_outputs[buf] < 0x1000000) {
      VisualWakeWords_heavy_4_compbuf = pi_l2_malloc((int) VisualWakeWords_heavy_4_outputs_bytes[buf]);
      ram_read(VisualWakeWords_heavy_4_compbuf, VisualWakeWords_heavy_4_outputs[buf],
               VisualWakeWords_heavy_4_outputs_bytes[buf]);
    } else {
      VisualWakeWords_heavy_4_compbuf = VisualWakeWords_heavy_4_outputs[buf];
    }

    if (VISUALWAKEWORDS_HEAVY_4_ISOUTPUTFLOAT) {
      VisualWakeWords_heavy_4_float_error_count = 0;
      VisualWakeWords_heavy_4_float_compare_args.expected = VisualWakeWords_heavy_4_testOutputVector[buf];
      VisualWakeWords_heavy_4_float_compare_args.actual = VisualWakeWords_heavy_4_compbuf;
      VisualWakeWords_heavy_4_float_compare_args.num_elements =
          VisualWakeWords_heavy_4_outputs_bytes[buf] / sizeof(float);
      VisualWakeWords_heavy_4_float_compare_args.output_buf_index = buf;
      VisualWakeWords_heavy_4_float_compare_args.err_count = &VisualWakeWords_heavy_4_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &VisualWakeWords_heavy_4_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      VisualWakeWords_heavy_4_tot_err += VisualWakeWords_heavy_4_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < VisualWakeWords_heavy_4_outputs_bytes[buf] / sizeof(VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE); i++) {
        VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE expected = ((VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE *)VisualWakeWords_heavy_4_testOutputVector[buf])[i];
        VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE actual = ((VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE *)VisualWakeWords_heavy_4_compbuf)[i];
        int32_t error = expected - actual;
        VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE diff = (VISUALWAKEWORDS_HEAVY_4_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          VisualWakeWords_heavy_4_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) VisualWakeWords_heavy_4_outputs[buf] < 0x1000000) {
      pi_l2_free(VisualWakeWords_heavy_4_compbuf, (int) VisualWakeWords_heavy_4_outputs_bytes[buf]);
    }
  }

  // -------------------------------------------------------------------
  // AnomalyDetection_heavy_4 Output Verification
  // -------------------------------------------------------------------
  uint32_t AnomalyDetection_heavy_4_tot_err, AnomalyDetection_heavy_4_tot_tested;
  AnomalyDetection_heavy_4_tot_err = 0;
  AnomalyDetection_heavy_4_tot_tested = 0;
  void *AnomalyDetection_heavy_4_compbuf;
  FloatCompareArgs AnomalyDetection_heavy_4_float_compare_args;
  uint32_t AnomalyDetection_heavy_4_float_error_count = 0;

  for (uint32_t buf = 0; buf < AnomalyDetection_heavy_4_num_outputs; buf++) {
    AnomalyDetection_heavy_4_tot_tested += AnomalyDetection_heavy_4_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE);

    if ((uint32_t) AnomalyDetection_heavy_4_outputs[buf] < 0x1000000) {
      AnomalyDetection_heavy_4_compbuf = pi_l2_malloc((int) AnomalyDetection_heavy_4_outputs_bytes[buf]);
      ram_read(AnomalyDetection_heavy_4_compbuf, AnomalyDetection_heavy_4_outputs[buf],
               AnomalyDetection_heavy_4_outputs_bytes[buf]);
    } else {
      AnomalyDetection_heavy_4_compbuf = AnomalyDetection_heavy_4_outputs[buf];
    }

    if (ANOMALYDETECTION_HEAVY_4_ISOUTPUTFLOAT) {
      AnomalyDetection_heavy_4_float_error_count = 0;
      AnomalyDetection_heavy_4_float_compare_args.expected = AnomalyDetection_heavy_4_testOutputVector[buf];
      AnomalyDetection_heavy_4_float_compare_args.actual = AnomalyDetection_heavy_4_compbuf;
      AnomalyDetection_heavy_4_float_compare_args.num_elements =
          AnomalyDetection_heavy_4_outputs_bytes[buf] / sizeof(float);
      AnomalyDetection_heavy_4_float_compare_args.output_buf_index = buf;
      AnomalyDetection_heavy_4_float_compare_args.err_count = &AnomalyDetection_heavy_4_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &AnomalyDetection_heavy_4_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      AnomalyDetection_heavy_4_tot_err += AnomalyDetection_heavy_4_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < AnomalyDetection_heavy_4_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE); i++) {
        ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE expected = ((ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE *)AnomalyDetection_heavy_4_testOutputVector[buf])[i];
        ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE actual = ((ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE *)AnomalyDetection_heavy_4_compbuf)[i];
        int32_t error = expected - actual;
        ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE diff = (ANOMALYDETECTION_HEAVY_4_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          AnomalyDetection_heavy_4_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) AnomalyDetection_heavy_4_outputs[buf] < 0x1000000) {
      pi_l2_free(AnomalyDetection_heavy_4_compbuf, (int) AnomalyDetection_heavy_4_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());


  printf("[VisualWakeWords_heavy_4] Errors: %u out of %u \r\n", VisualWakeWords_heavy_4_tot_err, VisualWakeWords_heavy_4_tot_tested);
  printf("[AnomalyDetection_heavy_4] Errors: %u out of %u \r\n", AnomalyDetection_heavy_4_tot_err, AnomalyDetection_heavy_4_tot_tested);

  if (VisualWakeWords_heavy_4_tot_err == 0 && AnomalyDetection_heavy_4_tot_err == 0) {
    return 0;
  } else {
    return VisualWakeWords_heavy_4_tot_err + AnomalyDetection_heavy_4_tot_err;
  }
  // return VisualWakeWords_heavy_4_tot_err;
}