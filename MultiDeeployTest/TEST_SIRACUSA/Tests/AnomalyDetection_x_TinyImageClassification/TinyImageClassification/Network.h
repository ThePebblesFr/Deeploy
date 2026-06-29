
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

extern int8_t *TinyImageClassification_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyImageClassification_MEMORYARENA_L1_len = 30792;
extern int8_t *TinyImageClassification_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyImageClassification_MEMORYARENA_L2_len = 65536;
extern float32_t *TinyImageClassification_input_0; // L2_init_template
static const uint32_t TinyImageClassification_input_0_len = 3072;
extern float32_t *TinyImageClassification_output_0; // L2_init_template
static const uint32_t TinyImageClassification_output_0_len = 10;
static const uint32_t TinyImageClassification_num_inputs = 1;
static const uint32_t TinyImageClassification_num_outputs = 1;
extern void *TinyImageClassification_inputs[1];
extern void *TinyImageClassification_outputs[1];
static const uint32_t TinyImageClassification_inputs_bytes[1] = {12288};
static const uint32_t TinyImageClassification_outputs_bytes[1] = {40};
#endif
