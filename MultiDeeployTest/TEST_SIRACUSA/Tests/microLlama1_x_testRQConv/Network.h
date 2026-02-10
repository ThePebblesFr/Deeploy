
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


extern int8_t *microLlama1_MEMORYARENA_L1;
static const uint32_t microLlama1_MEMORYARENA_L1_len = 18752;
extern int8_t *microLlama1_MEMORYARENA_L2;
static const uint32_t microLlama1_MEMORYARENA_L2_len = 3664;
extern int8_t *microLlama1_input_0;
static const uint32_t microLlama1_input_0_len = 64;
extern uint8_t *microLlama1_input_1;
static const uint32_t microLlama1_input_1_len = 32;
extern int8_t *microLlama1_input_2;
static const uint32_t microLlama1_input_2_len = 16;
extern int8_t *microLlama1_input_3;
static const uint32_t microLlama1_input_3_len = 64;
extern int8_t *microLlama1_input_4;
static const uint32_t microLlama1_input_4_len = 64;
extern int8_t *microLlama1_input_5;
static const uint32_t microLlama1_input_5_len = 64;
extern int8_t *microLlama1_input_6;
static const uint32_t microLlama1_input_6_len = 64;
extern int8_t *microLlama1_input_7;
static const uint32_t microLlama1_input_7_len = 64;
extern int8_t *microLlama1_input_8;
static const uint32_t microLlama1_input_8_len = 64;
extern int8_t *microLlama1_input_9;
static const uint32_t microLlama1_input_9_len = 64;
extern int8_t *microLlama1_input_10;
static const uint32_t microLlama1_input_10_len = 64;
extern int8_t *microLlama1_input_11;
static const uint32_t microLlama1_input_11_len = 64;
extern int8_t *microLlama1_input_12;
static const uint32_t microLlama1_input_12_len = 64;
extern int8_t *microLlama1_input_13;
static const uint32_t microLlama1_input_13_len = 64;
extern int8_t *microLlama1_input_14;
static const uint32_t microLlama1_input_14_len = 64;
extern int8_t *microLlama1_input_15;
static const uint32_t microLlama1_input_15_len = 64;
extern int8_t *microLlama1_input_16;
static const uint32_t microLlama1_input_16_len = 64;
extern int8_t *microLlama1_input_17;
static const uint32_t microLlama1_input_17_len = 64;
extern int8_t *microLlama1_input_18;
static const uint32_t microLlama1_input_18_len = 64;
extern int8_t *microLlama1_output_0;
static const uint32_t microLlama1_output_0_len = 64;
extern int8_t *microLlama1_output_1;
static const uint32_t microLlama1_output_1_len = 128;
extern int8_t *microLlama1_output_2;
static const uint32_t microLlama1_output_2_len = 128;
extern int8_t *microLlama1_output_3;
static const uint32_t microLlama1_output_3_len = 128;
extern int8_t *microLlama1_output_4;
static const uint32_t microLlama1_output_4_len = 128;
extern int8_t *microLlama1_output_5;
static const uint32_t microLlama1_output_5_len = 128;
extern int8_t *microLlama1_output_6;
static const uint32_t microLlama1_output_6_len = 128;
extern int8_t *microLlama1_output_7;
static const uint32_t microLlama1_output_7_len = 128;
extern int8_t *microLlama1_output_8;
static const uint32_t microLlama1_output_8_len = 128;
extern int8_t *microLlama1_output_9;
static const uint32_t microLlama1_output_9_len = 128;
extern int8_t *microLlama1_output_10;
static const uint32_t microLlama1_output_10_len = 128;
extern int8_t *microLlama1_output_11;
static const uint32_t microLlama1_output_11_len = 128;
extern int8_t *microLlama1_output_12;
static const uint32_t microLlama1_output_12_len = 128;
extern int8_t *microLlama1_output_13;
static const uint32_t microLlama1_output_13_len = 128;
extern int8_t *microLlama1_output_14;
static const uint32_t microLlama1_output_14_len = 128;
extern int8_t *microLlama1_output_15;
static const uint32_t microLlama1_output_15_len = 128;
extern int8_t *microLlama1_output_16;
static const uint32_t microLlama1_output_16_len = 128;
static const uint32_t microLlama1_num_inputs = 19;
static const uint32_t microLlama1_num_outputs = 17;
extern void *microLlama1_inputs[19];
extern void *microLlama1_outputs[17];
static const uint32_t microLlama1_inputs_bytes[19] = {64, 32, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
static const uint32_t microLlama1_outputs_bytes[17] = {64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};


extern int8_t *testRQConv_MEMORYARENA_L1;
static const uint32_t testRQConv_MEMORYARENA_L1_len = 13060;
extern int8_t *testRQConv_MEMORYARENA_L2;
static const uint32_t testRQConv_MEMORYARENA_L2_len = 12744;
extern int8_t *testRQConv_input_0;
static const uint32_t testRQConv_input_0_len = 4096;
extern int8_t *testRQConv_output_0;
static const uint32_t testRQConv_output_0_len = 6372;
static const uint32_t testRQConv_num_inputs = 1;
static const uint32_t testRQConv_num_outputs = 1;
extern void *testRQConv_inputs[1];
extern void *testRQConv_outputs[1];
static const uint32_t testRQConv_inputs_bytes[1] = {4096};
static const uint32_t testRQConv_outputs_bytes[1] = {6372};
#endif
