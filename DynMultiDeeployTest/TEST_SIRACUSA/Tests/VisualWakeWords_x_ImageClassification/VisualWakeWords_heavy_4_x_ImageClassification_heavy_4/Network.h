
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


extern int8_t *VisualWakeWords_heavy_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t VisualWakeWords_heavy_4_MEMORYARENA_L1_len = 31974;
extern int8_t *VisualWakeWords_heavy_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t VisualWakeWords_heavy_4_MEMORYARENA_L2_len = 73728;
extern int8_t *VisualWakeWords_heavy_4_input_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_4_input_0_len = 27648;
extern int8_t *VisualWakeWords_heavy_4_output_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_4_output_0_len = 2;
static const uint32_t VisualWakeWords_heavy_4_num_inputs = 1;
static const uint32_t VisualWakeWords_heavy_4_num_outputs = 1;
extern void *VisualWakeWords_heavy_4_inputs[1];
extern void *VisualWakeWords_heavy_4_outputs[1];
static const uint32_t VisualWakeWords_heavy_4_inputs_bytes[1] = {27648};
static const uint32_t VisualWakeWords_heavy_4_outputs_bytes[1] = {2};


extern int8_t *ImageClassification_heavy_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t ImageClassification_heavy_4_MEMORYARENA_L1_len = 32000;
extern int8_t *ImageClassification_heavy_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t ImageClassification_heavy_4_MEMORYARENA_L2_len = 65536;
extern int8_t *ImageClassification_heavy_4_input_0; // L2_init_template
static const uint32_t ImageClassification_heavy_4_input_0_len = 3072;
extern int8_t *ImageClassification_heavy_4_output_0; // L2_init_template
static const uint32_t ImageClassification_heavy_4_output_0_len = 10;
static const uint32_t ImageClassification_heavy_4_num_inputs = 1;
static const uint32_t ImageClassification_heavy_4_num_outputs = 1;
extern void *ImageClassification_heavy_4_inputs[1];
extern void *ImageClassification_heavy_4_outputs[1];
static const uint32_t ImageClassification_heavy_4_inputs_bytes[1] = {3072};
static const uint32_t ImageClassification_heavy_4_outputs_bytes[1] = {10};
#endif
