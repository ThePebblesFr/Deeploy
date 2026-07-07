
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


extern int8_t *miniMNV2_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t miniMNV2_6_MEMORYARENA_L1_len = 7712;
extern int8_t *miniMNV2_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t miniMNV2_6_MEMORYARENA_L2_len = 4096;
extern uint8_t *miniMNV2_6_input_0; // L2_init_template
static const uint32_t miniMNV2_6_input_0_len = 768;
extern int8_t *miniMNV2_6_output_0; // L2_init_template
static const uint32_t miniMNV2_6_output_0_len = 10;
static const uint32_t miniMNV2_6_num_inputs = 1;
static const uint32_t miniMNV2_6_num_outputs = 1;
extern void *miniMNV2_6_inputs[1];
extern void *miniMNV2_6_outputs[1];
static const uint32_t miniMNV2_6_inputs_bytes[1] = {768};
static const uint32_t miniMNV2_6_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_2_MEMORYARENA_L1_len = 15472;
extern int8_t *AnomalyDetection_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_2_MEMORYARENA_L2_len = 768;
extern uint8_t *AnomalyDetection_2_input_0; // L2_init_template
static const uint32_t AnomalyDetection_2_input_0_len = 640;
extern int8_t *AnomalyDetection_2_output_0; // L2_init_template
static const uint32_t AnomalyDetection_2_output_0_len = 640;
static const uint32_t AnomalyDetection_2_num_inputs = 1;
static const uint32_t AnomalyDetection_2_num_outputs = 1;
extern void *AnomalyDetection_2_inputs[1];
extern void *AnomalyDetection_2_outputs[1];
static const uint32_t AnomalyDetection_2_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_2_outputs_bytes[1] = {640};
#endif
