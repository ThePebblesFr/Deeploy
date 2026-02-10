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
  // Copy Model_1 input data to L2
  for (uint32_t buf = 0; buf < Model_1_num_inputs; buf++) {
    if ((uint32_t) Model_1_inputs[buf] >= 0x10000000) {
      memcpy(Model_1_inputs[buf], Model_1_testInputVector[buf],
             Model_1_inputs_bytes[buf]);
    }
  }

  // Copy Model_2 input data to L2
  for (uint32_t buf = 0; buf < Model_2_num_inputs; buf++) {
    if ((uint32_t) Model_2_inputs[buf] >= 0x10000000) {
      memcpy(Model_2_inputs[buf], Model_2_testInputVector[buf],
             Model_2_inputs_bytes[buf]);
    }
  }

#ifndef CI
  printf("Input copied\r\n");
#endif

  pi_cluster_task(&cluster_task, RunNetwork, NULL);
  cluster_task.stack_size = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  ResetTimer();
  StartTimer();
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  StopTimer();

#ifndef CI
  printf("Output:\r\n");
#endif

  // -------------------------------------------------------------------
  // Model_1 Output Verification
  // -------------------------------------------------------------------
  uint32_t Model_1_tot_err, Model_1_tot_tested;
  Model_1_tot_err = 0;
  Model_1_tot_tested = 0;
  void *Model_1_compbuf;
  FloatCompareArgs Model_1_float_compare_args;
  uint32_t Model_1_float_error_count = 0;

  for (uint32_t buf = 0; buf < Model_1_num_outputs; buf++) {
    Model_1_tot_tested += Model_1_outputs_bytes[buf] / sizeof(MODEL_1_OUTPUTTYPE);

    if ((uint32_t) Model_1_outputs[buf] < 0x1000000) {
      Model_1_compbuf = pi_l2_malloc((int) Model_1_outputs_bytes[buf]);
      ram_read(Model_1_compbuf, Model_1_outputs[buf],
               Model_1_outputs_bytes[buf]);
    } else {
      Model_1_compbuf = Model_1_outputs[buf];
    }

    if (MODEL_1_ISOUTPUTFLOAT) {
      Model_1_float_error_count = 0;
      Model_1_float_compare_args.expected = Model_1_testOutputVector[buf];
      Model_1_float_compare_args.actual = Model_1_compbuf;
      Model_1_float_compare_args.num_elements =
          Model_1_outputs_bytes[buf] / sizeof(float);
      Model_1_float_compare_args.output_buf_index = buf;
      Model_1_float_compare_args.err_count = &Model_1_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &Model_1_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      Model_1_tot_err += Model_1_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < Model_1_outputs_bytes[buf] / sizeof(MODEL_1_OUTPUTTYPE); i++) {
        MODEL_1_OUTPUTTYPE expected = ((MODEL_1_OUTPUTTYPE *)Model_1_testOutputVector[buf])[i];
        MODEL_1_OUTPUTTYPE actual = ((MODEL_1_OUTPUTTYPE *)Model_1_compbuf)[i];
        int32_t error = expected - actual;
        MODEL_1_OUTPUTTYPE diff = (MODEL_1_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          Model_1_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) Model_1_outputs[buf] < 0x1000000) {
      pi_l2_free(Model_1_compbuf, (int) Model_1_outputs_bytes[buf]);
    }
  }

  // -------------------------------------------------------------------
  // Model_2 Output Verification
  // -------------------------------------------------------------------
  uint32_t Model_2_tot_err, Model_2_tot_tested;
  Model_2_tot_err = 0;
  Model_2_tot_tested = 0;
  void *Model_2_compbuf;
  FloatCompareArgs Model_2_float_compare_args;
  uint32_t Model_2_float_error_count = 0;

  for (uint32_t buf = 0; buf < Model_2_num_outputs; buf++) {
    Model_2_tot_tested += Model_2_outputs_bytes[buf] / sizeof(MODEL_2_OUTPUTTYPE);

    if ((uint32_t) Model_2_outputs[buf] < 0x1000000) {
      Model_2_compbuf = pi_l2_malloc((int) Model_2_outputs_bytes[buf]);
      ram_read(Model_2_compbuf, Model_2_outputs[buf],
               Model_2_outputs_bytes[buf]);
    } else {
      Model_2_compbuf = Model_2_outputs[buf];
    }

    if (MODEL_2_ISOUTPUTFLOAT) {
      Model_2_float_error_count = 0;
      Model_2_float_compare_args.expected = Model_2_testOutputVector[buf];
      Model_2_float_compare_args.actual = Model_2_compbuf;
      Model_2_float_compare_args.num_elements =
          Model_2_outputs_bytes[buf] / sizeof(float);
      Model_2_float_compare_args.output_buf_index = buf;
      Model_2_float_compare_args.err_count = &Model_2_float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster,
                      &Model_2_float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      Model_2_tot_err += Model_2_float_error_count;
    } else {

      for (uint32_t i = 0;
           i < Model_2_outputs_bytes[buf] / sizeof(MODEL_2_OUTPUTTYPE); i++) {
        MODEL_2_OUTPUTTYPE expected = ((MODEL_2_OUTPUTTYPE *)Model_2_testOutputVector[buf])[i];
        MODEL_2_OUTPUTTYPE actual = ((MODEL_2_OUTPUTTYPE *)Model_2_compbuf)[i];
        int32_t error = expected - actual;
        MODEL_2_OUTPUTTYPE diff = (MODEL_2_OUTPUTTYPE)(error < 0 ? -error : error);

        if (diff) {
          Model_2_tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t) Model_2_outputs[buf] < 0x1000000) {
      pi_l2_free(Model_2_compbuf, (int) Model_2_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());


  printf("[Model_1] Errors: %u out of %u \r\n", Model_1_tot_err, Model_1_tot_tested);
  printf("[Model_2] Errors: %u out of %u \r\n", Model_2_tot_err, Model_2_tot_tested);

  if (Model_1_tot_err == 0 && Model_2_tot_err == 0) {
    return 0;
  } else {
    return Model_1_tot_err + Model_2_tot_err;
  }
  // return Model_1_tot_err;
}