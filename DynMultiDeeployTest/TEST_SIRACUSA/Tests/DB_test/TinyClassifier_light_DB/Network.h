
#ifndef __DEEPLOY_HEADER__
#define __DEEPLOY_HEADER__
#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunNetwork();
void InitNetwork();

extern int8_t *TinyClassifier_light_DB_test_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyClassifier_light_DB_test_MEMORYARENA_L1_len = 17472;
extern int8_t *TinyClassifier_light_DB_test_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyClassifier_light_DB_test_MEMORYARENA_L2_len = 288;
extern int8_t *TinyClassifier_light_DB_test_input_0; // L2_init_template
static const uint32_t TinyClassifier_light_DB_test_input_0_len = 256;
extern int8_t *TinyClassifier_light_DB_test_output_0; // L2_init_template
static const uint32_t TinyClassifier_light_DB_test_output_0_len = 10;
static const uint32_t TinyClassifier_light_DB_test_num_inputs = 1;
static const uint32_t TinyClassifier_light_DB_test_num_outputs = 1;
extern void *TinyClassifier_light_DB_test_inputs[1];
extern void *TinyClassifier_light_DB_test_outputs[1];
static const uint32_t TinyClassifier_light_DB_test_inputs_bytes[1] = {256};
static const uint32_t TinyClassifier_light_DB_test_outputs_bytes[1] = {10};
#endif
