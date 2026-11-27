# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

# Reuses the exact same Network.c/.h Round-Robin-Large already generates (same two
# full-cluster deployments, same InitNetwork_*/RunNetwork_* symbols) -- only the
# main-loop dispatch logic differs, so only that .c file is platform-specific here.
file(GLOB_RECURSE SOURCES
    ../Siracusa/src/CycleCounter.c
    ../DynSiracusa/src/OutputVerification.c
    src/priority_sequential.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../DynSiracusa/inc)
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../Siracusa/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId} "siracusa")

link_compile_dump(${TESTNAME})
