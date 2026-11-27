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

void RunNetworks_s_s_4_4();
void RunNetworks_s_l_2_6();
void RunNetworks_l_s_6_2();
void RunNetworks_l_l_4_4();
void InitNetworks_s_s_4_4();
void InitNetworks_s_l_2_6();
void InitNetworks_l_s_6_2();
void InitNetworks_l_l_4_4();




extern int8_t *TinyClassifier_light_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyClassifier_light_4_MEMORYARENA_L1_len = 8736;
extern int8_t *TinyClassifier_light_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyClassifier_light_4_MEMORYARENA_L2_len = 288;
extern int8_t *TinyClassifier_light_4_input_0; // L2_init_template
static const uint32_t TinyClassifier_light_4_input_0_len = 256;
extern int8_t *TinyClassifier_light_4_output_0; // L2_init_template
static const uint32_t TinyClassifier_light_4_output_0_len = 10;
static const uint32_t TinyClassifier_light_4_num_inputs = 1;
static const uint32_t TinyClassifier_light_4_num_outputs = 1;
extern void *TinyClassifier_light_4_inputs[1];
extern void *TinyClassifier_light_4_outputs[1];
static const uint32_t TinyClassifier_light_4_inputs_bytes[1] = {256};
static const uint32_t TinyClassifier_light_4_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_light_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_light_4_MEMORYARENA_L1_len = 31792;
extern int8_t *AnomalyDetection_light_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_light_4_MEMORYARENA_L2_len = 704;
extern int8_t *AnomalyDetection_light_4_input_0; // L2_init_template
static const uint32_t AnomalyDetection_light_4_input_0_len = 640;
extern int8_t *AnomalyDetection_light_4_output_0; // L2_init_template
static const uint32_t AnomalyDetection_light_4_output_0_len = 640;
static const uint32_t AnomalyDetection_light_4_num_inputs = 1;
static const uint32_t AnomalyDetection_light_4_num_outputs = 1;
extern void *AnomalyDetection_light_4_inputs[1];
extern void *AnomalyDetection_light_4_outputs[1];
static const uint32_t AnomalyDetection_light_4_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_light_4_outputs_bytes[1] = {640};




extern int8_t *TinyClassifier_light_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyClassifier_light_2_MEMORYARENA_L1_len = 8736;
extern int8_t *TinyClassifier_light_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyClassifier_light_2_MEMORYARENA_L2_len = 288;
extern int8_t *TinyClassifier_light_2_input_0; // L2_init_template
static const uint32_t TinyClassifier_light_2_input_0_len = 256;
extern int8_t *TinyClassifier_light_2_output_0; // L2_init_template
static const uint32_t TinyClassifier_light_2_output_0_len = 10;
static const uint32_t TinyClassifier_light_2_num_inputs = 1;
static const uint32_t TinyClassifier_light_2_num_outputs = 1;
extern void *TinyClassifier_light_2_inputs[1];
extern void *TinyClassifier_light_2_outputs[1];
static const uint32_t TinyClassifier_light_2_inputs_bytes[1] = {256};
static const uint32_t TinyClassifier_light_2_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_heavy_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_heavy_6_MEMORYARENA_L1_len = 46896;
extern int8_t *AnomalyDetection_heavy_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_heavy_6_MEMORYARENA_L2_len = 896;
extern int8_t *AnomalyDetection_heavy_6_input_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_6_input_0_len = 640;
extern int8_t *AnomalyDetection_heavy_6_output_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_6_output_0_len = 640;
static const uint32_t AnomalyDetection_heavy_6_num_inputs = 1;
static const uint32_t AnomalyDetection_heavy_6_num_outputs = 1;
extern void *AnomalyDetection_heavy_6_inputs[1];
extern void *AnomalyDetection_heavy_6_outputs[1];
static const uint32_t AnomalyDetection_heavy_6_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_heavy_6_outputs_bytes[1] = {640};




extern int8_t *TinyClassifier_heavy_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyClassifier_heavy_6_MEMORYARENA_L1_len = 34176;
extern int8_t *TinyClassifier_heavy_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyClassifier_heavy_6_MEMORYARENA_L2_len = 384;
extern int8_t *TinyClassifier_heavy_6_input_0; // L2_init_template
static const uint32_t TinyClassifier_heavy_6_input_0_len = 256;
extern int8_t *TinyClassifier_heavy_6_output_0; // L2_init_template
static const uint32_t TinyClassifier_heavy_6_output_0_len = 10;
static const uint32_t TinyClassifier_heavy_6_num_inputs = 1;
static const uint32_t TinyClassifier_heavy_6_num_outputs = 1;
extern void *TinyClassifier_heavy_6_inputs[1];
extern void *TinyClassifier_heavy_6_outputs[1];
static const uint32_t TinyClassifier_heavy_6_inputs_bytes[1] = {256};
static const uint32_t TinyClassifier_heavy_6_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_light_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_light_2_MEMORYARENA_L1_len = 15248;
extern int8_t *AnomalyDetection_light_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_light_2_MEMORYARENA_L2_len = 704;
extern int8_t *AnomalyDetection_light_2_input_0; // L2_init_template
static const uint32_t AnomalyDetection_light_2_input_0_len = 640;
extern int8_t *AnomalyDetection_light_2_output_0; // L2_init_template
static const uint32_t AnomalyDetection_light_2_output_0_len = 640;
static const uint32_t AnomalyDetection_light_2_num_inputs = 1;
static const uint32_t AnomalyDetection_light_2_num_outputs = 1;
extern void *AnomalyDetection_light_2_inputs[1];
extern void *AnomalyDetection_light_2_outputs[1];
static const uint32_t AnomalyDetection_light_2_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_light_2_outputs_bytes[1] = {640};




extern int8_t *TinyClassifier_heavy_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t TinyClassifier_heavy_4_MEMORYARENA_L1_len = 29936;
extern int8_t *TinyClassifier_heavy_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t TinyClassifier_heavy_4_MEMORYARENA_L2_len = 384;
extern int8_t *TinyClassifier_heavy_4_input_0; // L2_init_template
static const uint32_t TinyClassifier_heavy_4_input_0_len = 256;
extern int8_t *TinyClassifier_heavy_4_output_0; // L2_init_template
static const uint32_t TinyClassifier_heavy_4_output_0_len = 10;
static const uint32_t TinyClassifier_heavy_4_num_inputs = 1;
static const uint32_t TinyClassifier_heavy_4_num_outputs = 1;
extern void *TinyClassifier_heavy_4_inputs[1];
extern void *TinyClassifier_heavy_4_outputs[1];
static const uint32_t TinyClassifier_heavy_4_inputs_bytes[1] = {256};
static const uint32_t TinyClassifier_heavy_4_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_heavy_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_heavy_4_MEMORYARENA_L1_len = 31792;
extern int8_t *AnomalyDetection_heavy_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_heavy_4_MEMORYARENA_L2_len = 896;
extern int8_t *AnomalyDetection_heavy_4_input_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_4_input_0_len = 640;
extern int8_t *AnomalyDetection_heavy_4_output_0; // L2_init_template
static const uint32_t AnomalyDetection_heavy_4_output_0_len = 640;
static const uint32_t AnomalyDetection_heavy_4_num_inputs = 1;
static const uint32_t AnomalyDetection_heavy_4_num_outputs = 1;
extern void *AnomalyDetection_heavy_4_inputs[1];
extern void *AnomalyDetection_heavy_4_outputs[1];
static const uint32_t AnomalyDetection_heavy_4_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_heavy_4_outputs_bytes[1] = {640};
#endif
