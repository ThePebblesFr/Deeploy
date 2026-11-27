
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
#endif
