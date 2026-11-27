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

