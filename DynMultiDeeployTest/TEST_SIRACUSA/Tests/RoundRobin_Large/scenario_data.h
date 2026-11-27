#ifndef __SCENARIO_DATA_H__
#define __SCENARIO_DATA_H__

#define DELTA_T 100
#define TASK1_INPUT_SIZE 640
#define TASK2_INPUT_SIZE 27648

#define TASK1_MARGIN_THRESHOLD_LIGHT 1
#define TASK1_MARGIN_THRESHOLD_HEAVY 1
#define TASK2_MARGIN_THRESHOLD 1
#define ENERGY_THRESHOLD_NJ 100000ULL

#define PHI_ESTIMATE_LOW_ENERGY_NW 274600ULL
#define PHI_ESTIMATE_CRITICAL_TASK1_NW 216000ULL
#define PHI_ESTIMATE_CRITICAL_TASK2_NW 295500ULL
#define PHI_ESTIMATE_BOTH_CRITICAL_NW 276000ULL

#define TASK1_SCENARIO_HEX_FILE "task1_scenario.hex"
#define TASK2_SCENARIO_HEX_FILE "task2_scenario.hex"

static const uint8_t operator_order[DELTA_T] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#endif
