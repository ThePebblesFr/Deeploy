
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


extern int8_t *Adder_MEMORYARENA_L1; // L2_init_template
static const uint32_t Adder_MEMORYARENA_L1_len = 750;
extern int8_t *Adder_MEMORYARENA_L2; // L2_init_template
static const uint32_t Adder_MEMORYARENA_L2_len = 750;
extern int8_t *Adder_input_0; // L2_init_template
static const uint32_t Adder_input_0_len = 125;
extern int8_t *Adder_input_1; // L2_init_template
static const uint32_t Adder_input_1_len = 125;
extern int32_t *Adder_output_0; // L2_init_template
static const uint32_t Adder_output_0_len = 125;
static const uint32_t Adder_num_inputs = 2;
static const uint32_t Adder_num_outputs = 1;
extern void *Adder_inputs[2];
extern void *Adder_outputs[1];
static const uint32_t Adder_inputs_bytes[2] = {125, 125};
static const uint32_t Adder_outputs_bytes[1] = {500};


extern int8_t *testRQConv_MEMORYARENA_L1; // L2_init_template
static const uint32_t testRQConv_MEMORYARENA_L1_len = 13060;
extern int8_t *testRQConv_MEMORYARENA_L2; // L2_init_template
static const uint32_t testRQConv_MEMORYARENA_L2_len = 12744;
extern int8_t *testRQConv_input_0; // L2_init_template
static const uint32_t testRQConv_input_0_len = 4096;
extern int8_t *testRQConv_output_0; // L2_init_template
static const uint32_t testRQConv_output_0_len = 6372;
static const uint32_t testRQConv_num_inputs = 1;
static const uint32_t testRQConv_num_outputs = 1;
extern void *testRQConv_inputs[1];
extern void *testRQConv_outputs[1];
static const uint32_t testRQConv_inputs_bytes[1] = {4096};
static const uint32_t testRQConv_outputs_bytes[1] = {6372};
#endif
