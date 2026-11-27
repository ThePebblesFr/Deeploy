
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

extern int8_t *miniMNV2-- profileToJSON_MEMORYARENA_L1; // L2_init_template
static const uint32_t miniMNV2-- profileToJSON_MEMORYARENA_L1_len = 7712;
extern int8_t *miniMNV2-- profileToJSON_MEMORYARENA_L2; // L2_init_template
static const uint32_t miniMNV2-- profileToJSON_MEMORYARENA_L2_len = 4096;
extern uint8_t *miniMNV2-- profileToJSON_input_0; // L2_init_template
static const uint32_t miniMNV2-- profileToJSON_input_0_len = 768;
extern int8_t *miniMNV2-- profileToJSON_output_0; // L2_init_template
static const uint32_t miniMNV2-- profileToJSON_output_0_len = 10;
static const uint32_t miniMNV2-- profileToJSON_num_inputs = 1;
static const uint32_t miniMNV2-- profileToJSON_num_outputs = 1;
extern void *miniMNV2-- profileToJSON_inputs[1];
extern void *miniMNV2-- profileToJSON_outputs[1];
static const uint32_t miniMNV2-- profileToJSON_inputs_bytes[1] = {768};
static const uint32_t miniMNV2-- profileToJSON_outputs_bytes[1] = {10};
#endif
