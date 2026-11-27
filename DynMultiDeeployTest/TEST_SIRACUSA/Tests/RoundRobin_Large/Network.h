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
void RunNetwork_VisualWakeWords_heavy();
void InitNetwork_VisualWakeWords_heavy();

extern int8_t *VisualWakeWords_heavy_MEMORYARENA_L1; // L2_init_template
static const uint32_t VisualWakeWords_heavy_MEMORYARENA_L1_len = 65522;
extern int8_t *VisualWakeWords_heavy_MEMORYARENA_L2; // L2_init_template
static const uint32_t VisualWakeWords_heavy_MEMORYARENA_L2_len = 73728;
extern int8_t *VisualWakeWords_heavy_input_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_input_0_len = 27648;
extern int8_t *VisualWakeWords_heavy_output_0; // L2_init_template
static const uint32_t VisualWakeWords_heavy_output_0_len = 2;
static const uint32_t VisualWakeWords_heavy_num_inputs = 1;
static const uint32_t VisualWakeWords_heavy_num_outputs = 1;
extern void *VisualWakeWords_heavy_inputs[1];
extern void *VisualWakeWords_heavy_outputs[1];
static const uint32_t VisualWakeWords_heavy_inputs_bytes[1] = {27648};
static const uint32_t VisualWakeWords_heavy_outputs_bytes[1] = {2};

void RunNetwork_AnomalyDetection_heavy();
void InitNetwork_AnomalyDetection_heavy();

extern int8_t *AnomalyDetection_heavy_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_heavy_MEMORYARENA_L1_len = 62944;
extern int8_t *AnomalyDetection_heavy_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_heavy_MEMORYARENA_L2_len = 768;
extern int8_t *AnomalyDetection_heavy_input_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_input_0_len = 640;
extern int8_t *AnomalyDetection_heavy_output_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_output_0_len = 2;
static const uint32_t AnomalyDetection_heavy_num_inputs = 1;
static const uint32_t AnomalyDetection_heavy_num_outputs = 1;
extern void *AnomalyDetection_heavy_inputs[1];
extern void *AnomalyDetection_heavy_outputs[1];
static const uint32_t AnomalyDetection_heavy_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_heavy_outputs_bytes[1] = {2};
#endif
