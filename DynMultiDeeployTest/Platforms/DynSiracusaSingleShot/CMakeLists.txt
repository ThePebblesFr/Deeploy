# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

# Sibling of ../DynSiracusa, for the per-tick-relaunch DB workaround (see the
# plan file's "Step 2 follow-up" section): reuses DynSiracusa's
# CycleCounter.c/OutputVerification.c as-is (cluster_dev's actual definition +
# MAINSTACKSIZE/SLAVESTACKSIZE, no network-specific symbols) and its inc/ dir
# (Network.h-adjacent generated headers, scenario_data.h), same pattern
# RoundRobin/PrioritySequential already use to reuse DynSiracusa's plumbing
# without duplicating it. Only src/dynmultideeploytest_singleshot.c differs.
file(GLOB_RECURSE SOURCES
    ../Siracusa/src/CycleCounter.c
    ../DynSiracusa/src/OutputVerification.c
    src/dynmultideeploytest_singleshot.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../DynSiracusa/inc)
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../Siracusa/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId} "siracusa")

link_compile_dump(${TESTNAME})
