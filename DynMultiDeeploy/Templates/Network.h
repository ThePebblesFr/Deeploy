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

void RunNetworks_2_6();
void RunNetworks_4_4();
void RunNetworks_6_2();
void InitNetworks_2_6();
void InitNetworks_4_4();
void InitNetworks_6_2();

