
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


extern int8_t *VisualWakeWords_heavy_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t VisualWakeWords_heavy_6_MEMORYARENA_L1_len = 48000;
extern int8_t *VisualWakeWords_heavy_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t VisualWakeWords_heavy_6_MEMORYARENA_L2_len = 73728;
extern int8_t *VisualWakeWords_heavy_6_input_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_6_input_0_len = 27648;
extern int8_t *VisualWakeWords_heavy_6_output_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_6_output_0_len = 2;
static const uint32_t VisualWakeWords_heavy_6_num_inputs = 1;
static const uint32_t VisualWakeWords_heavy_6_num_outputs = 1;
extern void *VisualWakeWords_heavy_6_inputs[1];
extern void *VisualWakeWords_heavy_6_outputs[1];
static const uint32_t VisualWakeWords_heavy_6_inputs_bytes[1] = {27648};
static const uint32_t VisualWakeWords_heavy_6_outputs_bytes[1] = {2};


extern int8_t *AnomalyDetection_small_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_small_2_MEMORYARENA_L1_len = 1938;
extern int8_t *AnomalyDetection_small_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_small_2_MEMORYARENA_L2_len = 642;
extern int8_t *AnomalyDetection_small_2_input_0; // L2_init_template
static const uint32_t AnomalyDetection_small_2_input_0_len = 640;
extern int8_t *AnomalyDetection_small_2_output_0; // L2_init_template
static const uint32_t AnomalyDetection_small_2_output_0_len = 2;
static const uint32_t AnomalyDetection_small_2_num_inputs = 1;
static const uint32_t AnomalyDetection_small_2_num_outputs = 1;
extern void *AnomalyDetection_small_2_inputs[1];
extern void *AnomalyDetection_small_2_outputs[1];
static const uint32_t AnomalyDetection_small_2_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_small_2_outputs_bytes[1] = {2};
#endif
