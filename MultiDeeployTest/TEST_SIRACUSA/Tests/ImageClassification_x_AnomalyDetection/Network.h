
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


extern int8_t *ImageClassification_MEMORYARENA_L1; // L2_init_template
static const uint32_t ImageClassification_MEMORYARENA_L1_len = 43768;
extern int8_t *ImageClassification_MEMORYARENA_L2; // L2_init_template
static const uint32_t ImageClassification_MEMORYARENA_L2_len = 49152;
extern uint8_t *ImageClassification_input_0; // L2_init_template
static const uint32_t ImageClassification_input_0_len = 3072;
extern int8_t *ImageClassification_output_0; // L2_init_template
static const uint32_t ImageClassification_output_0_len = 10;
static const uint32_t ImageClassification_num_inputs = 1;
static const uint32_t ImageClassification_num_outputs = 1;
extern void *ImageClassification_inputs[1];
extern void *ImageClassification_outputs[1];
static const uint32_t ImageClassification_inputs_bytes[1] = {3072};
static const uint32_t ImageClassification_outputs_bytes[1] = {10};


extern int8_t *AnomalyDetection_MEMORYARENA_L1; // L2_init_template
static const uint32_t AnomalyDetection_MEMORYARENA_L1_len = 19856;
extern int8_t *AnomalyDetection_MEMORYARENA_L2; // L2_init_template
static const uint32_t AnomalyDetection_MEMORYARENA_L2_len = 768;
extern uint8_t *AnomalyDetection_input_0; // L2_init_template
static const uint32_t AnomalyDetection_input_0_len = 640;
extern int8_t *AnomalyDetection_output_0; // L2_init_template
static const uint32_t AnomalyDetection_output_0_len = 640;
static const uint32_t AnomalyDetection_num_inputs = 1;
static const uint32_t AnomalyDetection_num_outputs = 1;
extern void *AnomalyDetection_inputs[1];
extern void *AnomalyDetection_outputs[1];
static const uint32_t AnomalyDetection_inputs_bytes[1] = {640};
static const uint32_t AnomalyDetection_outputs_bytes[1] = {640};
#endif
