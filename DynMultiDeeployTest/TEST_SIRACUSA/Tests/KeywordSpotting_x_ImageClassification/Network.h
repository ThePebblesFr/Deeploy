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




extern int8_t *KeywordSpotting_small_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t KeywordSpotting_small_4_MEMORYARENA_L1_len = 3288;
extern int8_t *KeywordSpotting_small_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t KeywordSpotting_small_4_MEMORYARENA_L2_len = 2000;
extern int8_t *KeywordSpotting_small_4_input_0; // L2_init_template
static const uint32_t KeywordSpotting_small_4_input_0_len = 490;
extern int8_t *KeywordSpotting_small_4_output_0; // L2_init_template
static const uint32_t KeywordSpotting_small_4_output_0_len = 12;
static const uint32_t KeywordSpotting_small_4_num_inputs = 1;
static const uint32_t KeywordSpotting_small_4_num_outputs = 1;
extern void *KeywordSpotting_small_4_inputs[1];
extern void *KeywordSpotting_small_4_outputs[1];
static const uint32_t KeywordSpotting_small_4_inputs_bytes[1] = {490};
static const uint32_t KeywordSpotting_small_4_outputs_bytes[1] = {12};


extern int8_t *ImageClassification_small_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t ImageClassification_small_4_MEMORYARENA_L1_len = 23392;
extern int8_t *ImageClassification_small_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t ImageClassification_small_4_MEMORYARENA_L2_len = 24576;
extern int8_t *ImageClassification_small_4_input_0; // L2_init_template
static const uint32_t ImageClassification_small_4_input_0_len = 3072;
extern int8_t *ImageClassification_small_4_output_0; // L2_init_template
static const uint32_t ImageClassification_small_4_output_0_len = 10;
static const uint32_t ImageClassification_small_4_num_inputs = 1;
static const uint32_t ImageClassification_small_4_num_outputs = 1;
extern void *ImageClassification_small_4_inputs[1];
extern void *ImageClassification_small_4_outputs[1];
static const uint32_t ImageClassification_small_4_inputs_bytes[1] = {3072};
static const uint32_t ImageClassification_small_4_outputs_bytes[1] = {10};




extern int8_t *KeywordSpotting_small_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t KeywordSpotting_small_2_MEMORYARENA_L1_len = 3288;
extern int8_t *KeywordSpotting_small_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t KeywordSpotting_small_2_MEMORYARENA_L2_len = 2000;
extern int8_t *KeywordSpotting_small_2_input_0; // L2_init_template
static const uint32_t KeywordSpotting_small_2_input_0_len = 490;
extern int8_t *KeywordSpotting_small_2_output_0; // L2_init_template
static const uint32_t KeywordSpotting_small_2_output_0_len = 12;
static const uint32_t KeywordSpotting_small_2_num_inputs = 1;
static const uint32_t KeywordSpotting_small_2_num_outputs = 1;
extern void *KeywordSpotting_small_2_inputs[1];
extern void *KeywordSpotting_small_2_outputs[1];
static const uint32_t KeywordSpotting_small_2_inputs_bytes[1] = {490};
static const uint32_t KeywordSpotting_small_2_outputs_bytes[1] = {12};


extern int8_t *ImageClassification_heavy_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t ImageClassification_heavy_6_MEMORYARENA_L1_len = 48000;
extern int8_t *ImageClassification_heavy_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t ImageClassification_heavy_6_MEMORYARENA_L2_len = 65536;
extern int8_t *ImageClassification_heavy_6_input_0; // L2_init_template
static const uint32_t ImageClassification_heavy_6_input_0_len = 3072;
extern int8_t *ImageClassification_heavy_6_output_0; // L2_init_template
static const uint32_t ImageClassification_heavy_6_output_0_len = 10;
static const uint32_t ImageClassification_heavy_6_num_inputs = 1;
static const uint32_t ImageClassification_heavy_6_num_outputs = 1;
extern void *ImageClassification_heavy_6_inputs[1];
extern void *ImageClassification_heavy_6_outputs[1];
static const uint32_t ImageClassification_heavy_6_inputs_bytes[1] = {3072};
static const uint32_t ImageClassification_heavy_6_outputs_bytes[1] = {10};




extern int8_t *KeywordSpotting_heavy_6_MEMORYARENA_L1; // L2_init_template
static const uint32_t KeywordSpotting_heavy_6_MEMORYARENA_L1_len = 26304;
extern int8_t *KeywordSpotting_heavy_6_MEMORYARENA_L2; // L2_init_template
static const uint32_t KeywordSpotting_heavy_6_MEMORYARENA_L2_len = 16000;
extern int8_t *KeywordSpotting_heavy_6_input_0; // L2_init_template
static const uint32_t KeywordSpotting_heavy_6_input_0_len = 490;
extern int8_t *KeywordSpotting_heavy_6_output_0; // L2_init_template
static const uint32_t KeywordSpotting_heavy_6_output_0_len = 12;
static const uint32_t KeywordSpotting_heavy_6_num_inputs = 1;
static const uint32_t KeywordSpotting_heavy_6_num_outputs = 1;
extern void *KeywordSpotting_heavy_6_inputs[1];
extern void *KeywordSpotting_heavy_6_outputs[1];
static const uint32_t KeywordSpotting_heavy_6_inputs_bytes[1] = {490};
static const uint32_t KeywordSpotting_heavy_6_outputs_bytes[1] = {12};


extern int8_t *ImageClassification_small_2_MEMORYARENA_L1; // L2_init_template
static const uint32_t ImageClassification_small_2_MEMORYARENA_L1_len = 15995;
extern int8_t *ImageClassification_small_2_MEMORYARENA_L2; // L2_init_template
static const uint32_t ImageClassification_small_2_MEMORYARENA_L2_len = 24576;
extern int8_t *ImageClassification_small_2_input_0; // L2_init_template
static const uint32_t ImageClassification_small_2_input_0_len = 3072;
extern int8_t *ImageClassification_small_2_output_0; // L2_init_template
static const uint32_t ImageClassification_small_2_output_0_len = 10;
static const uint32_t ImageClassification_small_2_num_inputs = 1;
static const uint32_t ImageClassification_small_2_num_outputs = 1;
extern void *ImageClassification_small_2_inputs[1];
extern void *ImageClassification_small_2_outputs[1];
static const uint32_t ImageClassification_small_2_inputs_bytes[1] = {3072};
static const uint32_t ImageClassification_small_2_outputs_bytes[1] = {10};




extern int8_t *KeywordSpotting_heavy_4_MEMORYARENA_L1; // L2_init_template
static const uint32_t KeywordSpotting_heavy_4_MEMORYARENA_L1_len = 26304;
extern int8_t *KeywordSpotting_heavy_4_MEMORYARENA_L2; // L2_init_template
static const uint32_t KeywordSpotting_heavy_4_MEMORYARENA_L2_len = 16000;
extern int8_t *KeywordSpotting_heavy_4_input_0; // L2_init_template
static const uint32_t KeywordSpotting_heavy_4_input_0_len = 490;
extern int8_t *KeywordSpotting_heavy_4_output_0; // L2_init_template
static const uint32_t KeywordSpotting_heavy_4_output_0_len = 12;
static const uint32_t KeywordSpotting_heavy_4_num_inputs = 1;
static const uint32_t KeywordSpotting_heavy_4_num_outputs = 1;
extern void *KeywordSpotting_heavy_4_inputs[1];
extern void *KeywordSpotting_heavy_4_outputs[1];
static const uint32_t KeywordSpotting_heavy_4_inputs_bytes[1] = {490};
static const uint32_t KeywordSpotting_heavy_4_outputs_bytes[1] = {12};


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
