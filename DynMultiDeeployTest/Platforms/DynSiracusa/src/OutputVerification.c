#include "OutputVerification.h"

struct pi_device cluster_dev;

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

uint32_t VerifyModelOutput(OutputVerificationArgs *args, uint32_t *tot_tested) {
  uint32_t tot_err = 0;
  *tot_tested = 0;

  void *compbuf;
  FloatCompareArgs float_compare_args;
  uint32_t float_error_count = 0;
  struct pi_cluster_task cluster_task;

  for (uint32_t buf = 0; buf < args->num_outputs; buf++) {
    *tot_tested += args->outputs_bytes[buf] / args->output_elem_size;

    if ((uint32_t) args->outputs[buf] < 0x1000000) {
      compbuf = pi_l2_malloc((int) args->outputs_bytes[buf]);
      ram_read(compbuf, args->outputs[buf], args->outputs_bytes[buf]);
    } else {
      compbuf = args->outputs[buf];
    }

    if (args->is_output_float) {
      float_error_count = 0;
      float_compare_args.expected = args->testOutputVector[buf];
      float_compare_args.actual = compbuf;
      float_compare_args.num_elements = args->outputs_bytes[buf] / sizeof(float);
      float_compare_args.output_buf_index = buf;
      float_compare_args.err_count = &float_error_count;

      pi_cluster_task(&cluster_task, CompareFloatOnCluster, &float_compare_args);
      cluster_task.stack_size = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      tot_err += float_error_count;
    } else {
      for (uint32_t i = 0; i < args->outputs_bytes[buf] / args->output_elem_size; i++) {
        int32_t expected, actual;

        switch (args->output_elem_size) {
        case 1:
          expected = ((int8_t *) args->testOutputVector[buf])[i];
          actual = ((int8_t *) compbuf)[i];
          break;
        case 2:
          expected = ((int16_t *) args->testOutputVector[buf])[i];
          actual = ((int16_t *) compbuf)[i];
          break;
        default:
          expected = ((int32_t *) args->testOutputVector[buf])[i];
          actual = ((int32_t *) compbuf)[i];
          break;
        }

        int32_t error = expected - actual;
        int32_t diff = error < 0 ? -error : error;

        if (diff) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }

    if ((uint32_t) args->outputs[buf] < 0x1000000) {
      pi_l2_free(compbuf, (int) args->outputs_bytes[buf]);
    }
  }

  return tot_err;
}