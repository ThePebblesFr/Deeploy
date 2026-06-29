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
  // Copy AnomalyDetection input data to L2
  for (uint32_t buf = 0; buf < AnomalyDetection_num_inputs; buf++) {
    if ((uint32_t) AnomalyDetection_inputs[buf] >= 0x10000000) {
      memcpy(AnomalyDetection_inputs[buf], AnomalyDetection_testInputVector[buf],
             AnomalyDetection_inputs_bytes[buf]);
    }
  }

  // Copy TestRQAdd input data to L2
  for (uint32_t buf = 0; buf < TestRQAdd_num_inputs; buf++) {
    if ((uint32_t) TestRQAdd_inputs[buf] >= 0x10000000) {
      memcpy(TestRQAdd_inputs[buf], TestRQAdd_testInputVector[buf],
             TestRQAdd_inputs_bytes[buf]);
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
  // AnomalyDetection Output Verification
  // -------------------------------------------------------------------
  uint32_t AnomalyDetection_tot_err, AnomalyDetection_tot_tested;
  AnomalyDetection_tot_err = 0;
  AnomalyDetection_tot_tested = 0;
  void *AnomalyDetection_compbuf;
  FloatCompareArgs AnomalyDetection_float_compare_args;
  uint32_t AnomalyDetection_float_error_count = 0;

  for (uint32_t buf = 0; buf < AnomalyDetection_num_outputs; buf++) {
    AnomalyDetection_tot_tested += AnomalyDetection_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_OUTPUTTYPE);

    if ((uint32_t) AnomalyDetection_outputs[buf] < 0x1000000) {
      AnomalyDetection_compbuf = pi_l2_malloc((int) AnomalyDetection_outputs_bytes[buf]);
      ram_read(AnomalyDetection_compbuf, AnomalyDetection_outputs[buf],
               AnomalyDetection_outputs_bytes[buf]);
    } else {
      AnomalyDetection_compbuf = AnomalyDetection_outputs[buf];
    }

    if (ANOMALYDETECTION_ISOUTPUTFLOAT) {
      AnomalyDetection_float_error_count = 0;
      AnomalyDetection_float_compare_args.expected = AnomalyDetection_testOutputVector[buf];
      AnomalyDetection_float_compare_args.actual = AnomalyDetection_compbuf;
      AnomalyDetection_float_compare_args.num_elements =
          AnomalyDetection_outputs_bytes[buf] / sizeof(float);
      AnomalyDetection_float_compare_args.output_buf_index = buf;
      AnomalyDetection_float_compare_args.err_count = &AnomalyDetection_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &AnomalyDetection_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      AnomalyDetection_tot_err += AnomalyDetection_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < AnomalyDetection_outputs_bytes[buf] / sizeof(ANOMALYDETECTION_OUTPUTTYPE); i++) {
        ANOMALYDETECTION_OUTPUTTYPE expected = ((ANOMALYDETECTION_OUTPUTTYPE *)AnomalyDetection_testOutputVector[buf])[i];
        ANOMALYDETECTION_OUTPUTTYPE actual = ((ANOMALYDETECTION_OUTPUTTYPE *)AnomalyDetection_compbuf)[i];
        int32_t error = expected - actual;
        ANOMALYDETECTION_OUTPUTTYPE diff = (ANOMALYDETECTION_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          AnomalyDetection_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) AnomalyDetection_outputs[buf] < 0x1000000) {
      pi_l2_free(AnomalyDetection_compbuf, (int) AnomalyDetection_outputs_bytes[buf]);
    }
  }

  // -------------------------------------------------------------------
  // TestRQAdd Output Verification
  // -------------------------------------------------------------------
  uint32_t TestRQAdd_tot_err, TestRQAdd_tot_tested;
  TestRQAdd_tot_err = 0;
  TestRQAdd_tot_tested = 0;
  void *TestRQAdd_compbuf;
  FloatCompareArgs TestRQAdd_float_compare_args;
  uint32_t TestRQAdd_float_error_count = 0;

  for (uint32_t buf = 0; buf < TestRQAdd_num_outputs; buf++) {
    TestRQAdd_tot_tested += TestRQAdd_outputs_bytes[buf] / sizeof(TESTRQADD_OUTPUTTYPE);

    if ((uint32_t) TestRQAdd_outputs[buf] < 0x1000000) {
      TestRQAdd_compbuf = pi_l2_malloc((int) TestRQAdd_outputs_bytes[buf]);
      ram_read(TestRQAdd_compbuf, TestRQAdd_outputs[buf],
               TestRQAdd_outputs_bytes[buf]);
    } else {
      TestRQAdd_compbuf = TestRQAdd_outputs[buf];
    }

    if (TESTRQADD_ISOUTPUTFLOAT) {
      TestRQAdd_float_error_count = 0;
      TestRQAdd_float_compare_args.expected = TestRQAdd_testOutputVector[buf];
      TestRQAdd_float_compare_args.actual = TestRQAdd_compbuf;
      TestRQAdd_float_compare_args.num_elements =
          TestRQAdd_outputs_bytes[buf] / sizeof(float);
      TestRQAdd_float_compare_args.output_buf_index = buf;
      TestRQAdd_float_compare_args.err_count = &TestRQAdd_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &TestRQAdd_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      TestRQAdd_tot_err += TestRQAdd_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < TestRQAdd_outputs_bytes[buf] / sizeof(TESTRQADD_OUTPUTTYPE); i++) {
        TESTRQADD_OUTPUTTYPE expected = ((TESTRQADD_OUTPUTTYPE *)TestRQAdd_testOutputVector[buf])[i];
        TESTRQADD_OUTPUTTYPE actual = ((TESTRQADD_OUTPUTTYPE *)TestRQAdd_compbuf)[i];
        int32_t error = expected - actual;
        TESTRQADD_OUTPUTTYPE diff = (TESTRQADD_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          TestRQAdd_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) TestRQAdd_outputs[buf] < 0x1000000) {
      pi_l2_free(TestRQAdd_compbuf, (int) TestRQAdd_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());


  printf("[AnomalyDetection] Errors: %u out of %u \r\n", AnomalyDetection_tot_err, AnomalyDetection_tot_tested);
  printf("[TestRQAdd] Errors: %u out of %u \r\n", TestRQAdd_tot_err, TestRQAdd_tot_tested);

  if (AnomalyDetection_tot_err == 0 && TestRQAdd_tot_err == 0) {
    return 0;
  } else {
    return AnomalyDetection_tot_err + TestRQAdd_tot_err;
  }
  // return AnomalyDetection_tot_err;
}