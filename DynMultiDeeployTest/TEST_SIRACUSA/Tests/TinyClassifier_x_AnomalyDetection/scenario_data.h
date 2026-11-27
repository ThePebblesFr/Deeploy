#ifndef __SCENARIO_DATA_H__
#define __SCENARIO_DATA_H__

#define DELTA_T 100
#define TASK1_INPUT_SIZE 640
#define TASK2_INPUT_SIZE 256

#define TASK1_ERROR_THRESHOLD_LIGHT 50000
#define TASK1_ERROR_THRESHOLD_HEAVY 50000
#define TASK2_MARGIN_THRESHOLD 1
#define ENERGY_THRESHOLD_NJ 1000000000000ULL

#define PHI_ESTIMATE_LOW_ENERGY_NW 274600ULL
#define PHI_ESTIMATE_CRITICAL_TASK1_NW 216000ULL
#define PHI_ESTIMATE_CRITICAL_TASK2_NW 295500ULL
#define PHI_ESTIMATE_BOTH_CRITICAL_NW 276000ULL

#define TASK1_SCENARIO_HEX_FILE "task1_scenario.hex"
#define TASK2_SCENARIO_HEX_FILE "task2_scenario.hex"

static const uint8_t operator_order[DELTA_T] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

#endif
