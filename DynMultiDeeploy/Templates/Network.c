#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Network.h"

// ===== GLOBALS START =====

// ===== GLOBALS END =====

// No CORES MAP section: each config's combined closure (see
// MultiDeeploy/CodeFusion/SourceFusion.py) bakes its own core-range boundaries in
// as literal constants, computed from that config's own dedicated core counts --
// there is no shared cores_map/nb_dedicated_cores global left to declare here.
// (This used to be a single hardcoded value shared by every mode once fused --
// a real bug, since only whichever mode's core split happened to match this
// placeholder actually got the right core assignment. Fixed by removing the
// shared array entirely rather than trying to keep it in sync per mode.)

// ===== SUBGROUP BARRIERS START =====

#include "hal/eu/eu_v3.h"
#include "archi/eu/eu_v3.h"

static inline uint32_t subgroup_core_mask(uint32_t group_start, uint32_t group_end)
{
    uint32_t mask = 0;

    for (uint32_t c = group_start; c <= group_end; c++) {
        for (volatile int i = 0; i < 8; i++) {
            asm volatile("nop");
        }
        // printf("Adding core %u to subgroup barrier mask (bit %08x)\n", c, bit);
        mask |= (1u << c);
    }

    return mask;
}

static inline void subgroup_barrier_init(uint32_t barrier_id,
                                         uint32_t group_start,
                                         uint32_t group_end)
{
    uint32_t mask = subgroup_core_mask(group_start, group_end);
    eu_bar_setup(eu_bar_addr(barrier_id), mask);
}

static inline void subgroup_barrier_wait(uint32_t barrier_id, uint32_t core_id)
{
    // printf("Core %u waiting on barrier %u\n", core_id, barrier_id);
    // Deliberately does NOT clear PULP_HW_BAR_EVENT from the mask after use
    // (a fix attempted and reverted -- see git history/session notes): PULP-OS's
    // OWN pi_cl_team_barrier()/pi_cl_team_fork() join-wait (pos_team_offload_wait)
    // call eu_bar_trig_wait_clr() DIRECTLY, without ever arming
    // PULP_HW_BAR_EVENT themselves -- they rely on it staying permanently
    // enabled once armed here. Clearing it after use breaks the very next
    // pi_cl_team_fork's join-wait (used to dispatch every Combined_closure_*),
    // which then hangs forever since nothing re-arms the bit for it. Confirmed
    // by tracing pi_cl_team_barrier()'s implementation
    // (pulp_hal/.../pos/implem/pe.h) after a mask-clear "fix" here caused
    // exactly that hang.
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    eu_bar_trig_wait_clr(eu_bar_addr(barrier_id));
    // printf("Core %u passed barrier %u\n", core_id, barrier_id);
}

// ===== SUBGROUP BARRIERS END =====

// ===== FUNCTIONS START =====

// ===== FUNCTIONS END =====

// ===== RUNNETWORKS START =====

// ===== RUNNETWORKS END =====

// ===== INITNETWORKS START =====

// ===== INITNETWORKS END =====