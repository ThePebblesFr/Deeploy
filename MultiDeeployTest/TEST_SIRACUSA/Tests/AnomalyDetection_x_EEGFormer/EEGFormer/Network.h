
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

extern int8_t *EEGFormer_MEMORYARENA_L1; // L2_init_template
static const uint32_t EEGFormer_MEMORYARENA_L1_len = 31946;
extern int8_t *EEGFormer_MEMORYARENA_L2; // L2_init_template
static const uint32_t EEGFormer_MEMORYARENA_L2_len = 38016;
extern uint8_t *EEGFormer_input_0; // L2_init_template
static const uint32_t EEGFormer_input_0_len = 1056;
extern int8_t *EEGFormer_output_0; // L2_init_template
static const uint32_t EEGFormer_output_0_len = 1056;
static const uint32_t EEGFormer_num_inputs = 1;
static const uint32_t EEGFormer_num_outputs = 1;
extern void *EEGFormer_inputs[1];
extern void *EEGFormer_outputs[1];
static const uint32_t EEGFormer_inputs_bytes[1] = {1056};
static const uint32_t EEGFormer_outputs_bytes[1] = {1056};
#endif
