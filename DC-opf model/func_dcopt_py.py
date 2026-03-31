"""
Pure Python port of func_dcopt.m, add_branchCapacity.m, and load2disp.

No MATLAB/Octave dependency — uses scipy.optimize.linprog for DC-OPF solving
with the dual-simplex method (matching the patched MATPOWER qps_ot.m solver).
Compatible with MATPOWER .m case files via the included parser.

Usage:
    from func_dcopt_py import load_case, add_branch_capacity, func_dcopt

    ppc = load_case('case14')  # or load_case('/path/to/case_ACTIVSg2000.m')
    ppc = add_branch_capacity(ppc, alpha=2.0)

    system_state = np.zeros(nb + nl)
    system_state[0] = 1  # fail bus 1
    blackout_size, flag = func_dcopt(system_state, ppc)
"""

import numpy as np
import re
from copy import deepcopy
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, csr_array, csc_array, bmat as sparse_bmat
from scipy.sparse.csgraph import connected_components
from scipy.optimize._linprog_highs import _highs_wrapper
from scipy.optimize._highspy._core import (
    _Highs, HighsLp, ObjSense, MatrixFormat, HighsModelStatus,
)

from pypower.rundcpf import rundcpf
from pypower.ppoption import ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, BASE_KV, REF
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, APF
from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, RATE_A, BR_STATUS, PF as BR_PF, PT as BR_PT
from pypower.idx_cost import MODEL, STARTUP, SHUTDOWN, NCOST, COST, POLYNOMIAL


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _parse_matpower_matrix(text, field_name):
    """Extract a numeric matrix from MATPOWER .m file text."""
    pattern = rf'mpc\.{field_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)
    rows = []
    for line in block.strip().split('\n'):
        line = line.split('%')[0].strip().rstrip(';')
        if not line:
            continue
        vals = line.split()
        if vals:
            rows.append([float(v) for v in vals])
    return np.array(rows, dtype=float) if rows else None


def load_case(case_name_or_path):
    """
    Load a MATPOWER case, returning a PYPOWER-compatible dict.

    Args:
        case_name_or_path: Either a built-in PYPOWER case name (e.g., 'case14')
            or a path to a MATPOWER .m file.

    Returns:
        ppc dict with keys: version, baseMVA, bus, gen, branch, gencost
    """
    path = Path(case_name_or_path)
    if path.exists() and path.suffix == '.m':
        # Parse MATPOWER .m file
        with open(path, 'r') as f:
            text = f.read()
        bus = _parse_matpower_matrix(text, 'bus')
        gen = _parse_matpower_matrix(text, 'gen')
        branch = _parse_matpower_matrix(text, 'branch')
        gencost = _parse_matpower_matrix(text, 'gencost')
        m = re.search(r'mpc\.baseMVA\s*=\s*(\d+\.?\d*)', text)
        baseMVA = float(m.group(1)) if m else 100.0

        ppc = {
            'version': '2',
            'baseMVA': baseMVA,
            'bus': bus,
            'gen': gen,
            'branch': branch,
        }
        if gencost is not None:
            ppc['gencost'] = gencost
        return ppc
    else:
        # Try built-in PYPOWER case
        try:
            mod = __import__(f'pypower.{case_name_or_path}',
                             fromlist=[case_name_or_path])
            return getattr(mod, case_name_or_path)()
        except (ImportError, AttributeError):
            raise FileNotFoundError(
                f"Case '{case_name_or_path}' not found as file or PYPOWER built-in"
            )


# ---------------------------------------------------------------------------
# Branch capacity (port of add_branchCapacity.m)
# ---------------------------------------------------------------------------

def add_branch_capacity(ppc, alpha=2.0):
    """
    Set branch thermal limits based on DC power flow results.

    If all RATE_A values are zero, runs a DC power flow and sets
    RATE_A = alpha * 0.5 * |PF - PT| for each branch.

    Args:
        ppc: PYPOWER case dict
        alpha: capacity scaling factor (default 2.0)

    Returns:
        ppc with updated branch RATE_A values
    """
    ppc = deepcopy(ppc)
    if np.all(ppc['branch'][:, RATE_A] == 0):
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        results = rundcpf(ppc, ppopt)
        res_branch = results[0]['branch']
        ppc['branch'][:, RATE_A] = (
            alpha * 0.5 * np.abs(res_branch[:, BR_PF] - res_branch[:, BR_PT])
        )
    # PYPOWER treats RATE_A=0 as zero capacity (unlike MATPOWER where 0=unlimited)
    # Replace any remaining zeros with a large value
    zero_rate = ppc['branch'][:, RATE_A] == 0
    if np.any(zero_rate):
        ppc['branch'][zero_rate, RATE_A] = 9999.0
    return ppc


# ---------------------------------------------------------------------------
# load2disp (port of MATPOWER's load2disp.m)
# ---------------------------------------------------------------------------

def load2disp(ppc):
    """
    Convert fixed loads to negative dispatchable generators.

    For each bus with PD > 0, creates a dispatchable "generator" with
    negative Pg representing the load, allowing the OPF to shed load
    at a high cost (value of lost load).

    Returns:
        ppc with modified bus, gen, gencost tables
    """
    ppc = deepcopy(ppc)
    bus = ppc['bus']
    gen = ppc['gen']
    gencost = ppc.get('gencost')

    # Ensure gen has at least 21 columns (PYPOWER standard)
    if gen.shape[1] < 21:
        gen = np.hstack([gen, np.zeros((gen.shape[0], 21 - gen.shape[1]))])

    # Find load buses (only positive loads; negative PD are generation injections
    # that should not be converted to dispatchable loads — doing so creates
    # PMIN = -PD > 0 > PMAX = 0, making the OPF infeasible)
    load_idx = np.where(bus[:, PD] > 0)[0]
    nld = len(load_idx)

    if nld == 0:
        ppc['gen'] = gen
        return ppc

    ng = gen.shape[0]

    # Create dispatchable load generators (negative generation)
    new_gen = np.zeros((nld, gen.shape[1]))
    new_gen[:, GEN_BUS] = bus[load_idx, BUS_I]
    new_gen[:, PG] = -bus[load_idx, PD]
    new_gen[:, QG] = -bus[load_idx, QD]
    new_gen[:, QMAX] = 0
    new_gen[:, QMIN] = -bus[load_idx, QD]
    new_gen[:, VG] = bus[load_idx, VM]
    new_gen[:, MBASE] = bus[load_idx, BASE_KV] if np.any(bus[load_idx, BASE_KV]) else 100.0
    new_gen[:, GEN_STATUS] = 1
    new_gen[:, PMAX] = 0
    new_gen[:, PMIN] = -bus[load_idx, PD]

    # Append to gen table
    ppc['gen'] = np.vstack([gen, new_gen])

    # Zero out original loads
    ppc['bus'][load_idx, PD] = 0
    ppc['bus'][load_idx, QD] = 0

    # Create gencost for dispatchable loads (high cost = value of lost load)
    # Cost = VOLL * Pg. Since Pg is negative for loads, the OPF is incentivized
    # to serve load (large negative cost) rather than shed it (cost = 0).
    voll = 5000.0
    new_gencost = np.zeros((nld, 7))
    new_gencost[:, MODEL] = POLYNOMIAL
    new_gencost[:, STARTUP] = 0
    new_gencost[:, SHUTDOWN] = 0
    new_gencost[:, NCOST] = 2  # linear cost: c1*Pg + c0
    new_gencost[:, COST] = voll  # linear coefficient
    new_gencost[:, COST + 1] = 0  # constant term

    if gencost is not None:
        # Pad gencost to same number of columns if needed
        nc = max(gencost.shape[1], new_gencost.shape[1])
        if gencost.shape[1] < nc:
            gencost = np.hstack([gencost, np.zeros((gencost.shape[0], nc - gencost.shape[1]))])
        if new_gencost.shape[1] < nc:
            new_gencost = np.hstack([new_gencost, np.zeros((nld, nc - new_gencost.shape[1]))])
        ppc['gencost'] = np.vstack([gencost, new_gencost])
    else:
        ppc['gencost'] = new_gencost

    return ppc


# ---------------------------------------------------------------------------
# DC-OPF solver using scipy.optimize.linprog (dual-simplex)
# ---------------------------------------------------------------------------

def _solve_dcopf(mpc):
    """
    Solve DC optimal power flow using scipy.optimize.linprog.

    Formulation (all in p.u. on baseMVA):
        Decision variables: x = [Pg (ng); Va (nb)]

        Minimize: c_pg' * Pg

        Subject to:
            - Power balance at each bus: Cg * Pg - Bbus * Va = 0
            - Branch flow limits: -RATE_A <= Bf * Va <= RATE_A
            - Reference bus angle: Va_ref = 0
            - Generator limits: Pmin <= Pg <= Pmax
            - Va unbounded

    This formulation handles disconnected islands naturally since each
    island has its own independent power balance constraints.

    Returns:
        dict with 'success', 'gen' (updated generator table with Pg)
    """
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']
    gencost = mpc['gencost']
    baseMVA = mpc.get('baseMVA', 100.0)

    nb = bus.shape[0]
    ng_total = gen.shape[0]
    nl = branch.shape[0]

    if nb == 0 or ng_total == 0 or nl == 0:
        return {'success': False}

    # Bus numbering: external -> internal (0-indexed)
    bus_ids = bus[:, BUS_I].astype(int)
    bus_map = {int(bid): i for i, bid in enumerate(bus_ids)}

    # Branch susceptance
    x_br = branch[:, BR_X].copy()
    x_br[x_br == 0] = 1e-6
    tap = branch[:, TAP].copy()
    tap[tap == 0] = 1.0
    b_branch = 1.0 / (x_br * tap)

    f_bus = np.array([bus_map[int(b)] for b in branch[:, F_BUS]])
    t_bus = np.array([bus_map[int(b)] for b in branch[:, T_BUS]])

    # Bf: branch-bus susceptance matrix (nl x nb)
    Bf = np.zeros((nl, nb))
    for l in range(nl):
        Bf[l, f_bus[l]] = b_branch[l]
        Bf[l, t_bus[l]] = -b_branch[l]

    # Bbus: bus susceptance matrix (nb x nb)
    Bbus = np.zeros((nb, nb))
    for l in range(nl):
        f, t = f_bus[l], t_bus[l]
        b = b_branch[l]
        Bbus[f, f] += b
        Bbus[f, t] -= b
        Bbus[t, f] -= b
        Bbus[t, t] += b

    # Cg: generator-bus connection matrix (nb x ng_total)
    Cg = np.zeros((nb, ng_total))
    for g in range(ng_total):
        g_bus_idx = bus_map.get(int(gen[g, GEN_BUS]))
        if g_bus_idx is not None:
            Cg[g_bus_idx, g] = 1.0

    # Reference bus (one per island)
    ref_buses = _find_ref_buses(bus, branch, bus_map)

    # --- Decision variables: x = [Pg (ng_total), Va (nb)] ---
    n_vars = ng_total + nb

    # Objective: min c_pg' * Pg + 0 * Va
    c_obj = np.zeros(n_vars)
    for g in range(ng_total):
        if gencost[g, NCOST] >= 2:
            c_obj[g] = gencost[g, COST]

    # Bounds
    pg_min = gen[:, PMIN] / baseMVA
    pg_max = gen[:, PMAX] / baseMVA
    bounds = (
        [(pg_min[g], pg_max[g]) for g in range(ng_total)] +
        [(-np.inf, np.inf) if i not in ref_buses else (0, 0) for i in range(nb)]
    )

    # Equality: Cg * Pg - Bbus * Va = 0  (power balance at each bus)
    A_eq = np.hstack([Cg, -Bbus])
    b_eq = np.zeros(nb)

    # Inequality: -RATE_A <= Bf * Va <= RATE_A
    rate_a = branch[:, RATE_A] / baseMVA
    rate_a[rate_a == 0] = 1e10  # unlimited

    # [0 | Bf] * x <= rate_a  and  [0 | -Bf] * x <= rate_a
    zeros_ng = np.zeros((nl, ng_total))
    A_ub = np.vstack([
        np.hstack([zeros_ng, Bf]),
        np.hstack([zeros_ng, -Bf]),
    ])
    b_ub = np.concatenate([rate_a, rate_a])

    # --- Solve with dual-simplex (matching patched qps_ot.m) ---
    try:
        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs-ds',
            options={'disp': False, 'presolve': True},
        )
    except Exception:
        result = type('R', (), {'success': False})()

    if not result.success:
        # Fallback to interior-point
        try:
            result = linprog(
                c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs-ipm',
                options={'disp': False, 'presolve': True},
            )
        except Exception:
            return {'success': False}

    if not result.success:
        return {'success': False}

    # Extract Pg results (scale back to MW)
    pg_result = result.x[:ng_total] * baseMVA

    gen_result = gen.copy()
    gen_result[:, PG] = pg_result

    return {'success': True, 'gen': gen_result}


def _find_ref_buses(bus, branch, bus_map):
    """Find one reference bus per connected island."""
    nb = bus.shape[0]
    bus_ids = bus[:, BUS_I].astype(int)

    # Build adjacency
    adj = [[] for _ in range(nb)]
    for row in branch:
        f = bus_map.get(int(row[F_BUS]))
        t = bus_map.get(int(row[T_BUS]))
        if f is not None and t is not None:
            adj[f].append(t)
            adj[t].append(f)

    # BFS to find islands, pick ref bus for each
    visited = [False] * nb
    ref_buses = set()
    for start in range(nb):
        if visited[start]:
            continue
        # Find all buses in this island
        island = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            island.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        # Pick existing REF bus, or first bus in island
        ref_found = False
        for i in island:
            if bus[i, BUS_TYPE] == REF:
                ref_buses.add(i)
                ref_found = True
                break
        if not ref_found:
            ref_buses.add(island[0])

    return ref_buses


# ---------------------------------------------------------------------------
# Precomputed DC-OPF solver (fast repeated calls)
# ---------------------------------------------------------------------------

class DcopfPrecomputed:
    """
    Precomputed DC-OPF solver for fast repeated evaluation.

    Precomputes all matrices and data structures once from a base case,
    then each __call__ only slices out failed components and solves the LP
    via the HiGHS C++ wrapper directly (bypassing scipy.linprog overhead).

    Typical speedup: ~100x for large cases (e.g., ACTIVSg2000: 710ms -> ~7ms).
    """

    def __init__(self, ppc0):
        """
        Precompute all LP structures from the base case.

        Args:
            ppc0: PYPOWER case dict with branch capacities already set
                  (from add_branch_capacity).
        """
        # Apply load2disp once
        mpc = load2disp(ppc0)
        baseMVA = mpc.get('baseMVA', 100.0)

        bus = mpc['bus']
        gen = mpc['gen']
        branch = mpc['branch']
        gencost = mpc['gencost']

        nb = bus.shape[0]
        ng = ppc0['gen'].shape[0]  # original generators
        ng_total = gen.shape[0]    # original + dispatchable loads
        nl = branch.shape[0]

        self.nb = nb
        self.ng = ng
        self.ng_total = ng_total
        self.nl = nl
        self.baseMVA = baseMVA

        # Store bus/branch IDs for state mapping
        self.bus_dic = ppc0['bus'][:, BUS_I].astype(int).copy()
        self.branch_f = ppc0['branch'][:, F_BUS].astype(int).copy()
        self.branch_t = ppc0['branch'][:, T_BUS].astype(int).copy()

        # Generator-to-bus index mapping (original generators only)
        gen_bus_ids = ppc0['gen'][:, GEN_BUS].astype(int)
        bus_id_to_idx = {int(bid): i for i, bid in enumerate(self.bus_dic)}
        self.gen_idx = np.array([bus_id_to_idx[int(g)] for g in gen_bus_ids])

        # Bus numbering for matrices
        bus_ids = bus[:, BUS_I].astype(int)
        bus_map = {int(bid): i for i, bid in enumerate(bus_ids)}

        # Original ref bus type
        self.bus_type = bus[:, BUS_TYPE].copy()

        # Total demand (from dispatchable loads)
        # Zero out original generator costs and set Pmin=0
        gencost[:ng, :] = 0
        gencost[:ng, MODEL] = POLYNOMIAL
        gencost[:ng, NCOST] = 2
        gen[:ng, PMIN] = 0
        gen[:ng, GEN_STATUS] = 1

        self.totpf = -np.sum(gen[ng:, PG])

        # Generator bus IDs for the full gen table (orig + disp loads)
        self.gen_bus_ids_full = gen[:, GEN_BUS].astype(int).copy()
        # Map from full gen index to bus internal index
        self.gen_bus_idx = np.array(
            [bus_map.get(int(g), -1) for g in self.gen_bus_ids_full])

        # Branch susceptance
        x_br = branch[:, BR_X].copy()
        x_br[x_br == 0] = 1e-6
        tap = branch[:, TAP].copy()
        tap[tap == 0] = 1.0
        b_branch = 1.0 / (x_br * tap)

        f_bus = np.array([bus_map[int(b)] for b in branch[:, F_BUS]])
        t_bus = np.array([bus_map[int(b)] for b in branch[:, T_BUS]])
        self.f_bus = f_bus
        self.t_bus = t_bus
        self.b_branch = b_branch

        # Build sparse Bf (nl x nb) — each row is one branch
        row_bf = np.concatenate([np.arange(nl), np.arange(nl)])
        col_bf = np.concatenate([f_bus, t_bus])
        data_bf = np.concatenate([b_branch, -b_branch])
        self.Bf = csr_array((data_bf, (row_bf, col_bf)), shape=(nl, nb))

        # Build sparse Cg (nb x ng_total)
        valid = self.gen_bus_idx >= 0
        self.Cg = csr_array(
            (np.ones(valid.sum()), (self.gen_bus_idx[valid], np.where(valid)[0])),
            shape=(nb, ng_total))

        # Cost vector: c = [gencost for Pg | 0 for Va]
        c_obj = np.zeros(ng_total + nb)
        for g in range(ng_total):
            if gencost[g, NCOST] >= 2:
                c_obj[g] = gencost[g, COST]
        self.c_obj = c_obj

        # Bounds template (in p.u.)
        pg_min = gen[:, PMIN] / baseMVA
        pg_max = gen[:, PMAX] / baseMVA
        lb = np.concatenate([pg_min, np.full(nb, -1e20)])
        ub = np.concatenate([pg_max, np.full(nb, 1e20)])
        self.lb_template = lb
        self.ub_template = ub

        # Store original generator Pmax in p.u. for scaling
        self.orig_gen_pmax = gen[:ng, PMAX].copy() / baseMVA

        # Rate A in p.u.
        rate_a = branch[:, RATE_A] / baseMVA
        rate_a[rate_a == 0] = 1e10
        self.rate_a = rate_a

        # Precompute branch endpoint bus IDs for fast removal
        self.branch_f_bus_id = branch[:, F_BUS].astype(int)
        self.branch_t_bus_id = branch[:, T_BUS].astype(int)

        # Persistent HiGHS solver (avoids option validation overhead per call)
        self._highs = _Highs()
        self._highs.setOptionValue('output_flag', False)
        self._highs.setOptionValue('presolve', 'on')

    def __call__(self, system_state):
        """
        Compute blackout size for a given system state.

        Args:
            system_state: 1D array of length (nb + nl).

        Returns:
            (blackout_size, flag) same as func_dcopt.
        """
        nb, ng, ng_total, nl = self.nb, self.ng, self.ng_total, self.nl

        # Parse state
        bus_state = system_state[:nb]
        branch_state = system_state[nb:nb + nl]
        gen_state = bus_state[self.gen_idx]

        # Identify removed buses/branches
        removed_bus_mask = bus_state == 1
        removed_bus_set = set(self.bus_dic[removed_bus_mask].tolist())

        removed_branch_mask = branch_state == 1
        # Also remove branches connected to removed buses
        if removed_bus_set:
            br_f_removed = np.isin(self.branch_f_bus_id, list(removed_bus_set))
            br_t_removed = np.isin(self.branch_t_bus_id, list(removed_bus_set))
            removed_branch_mask = removed_branch_mask | br_f_removed | br_t_removed

        keep_bus = ~removed_bus_mask
        keep_branch = ~removed_branch_mask

        # Generators/loads to keep: those not at failed buses
        gen_at_removed = np.isin(self.gen_bus_ids_full.astype(int),
                                  list(removed_bus_set)) if removed_bus_set else \
                         np.zeros(ng_total, dtype=bool)
        keep_gen = ~gen_at_removed

        n_keep_bus = int(keep_bus.sum())
        n_keep_branch = int(keep_branch.sum())
        n_keep_gen = int(keep_gen.sum())

        if n_keep_bus == 0 or n_keep_gen == 0 or n_keep_branch == 0:
            return 100.0, 0

        # Map surviving bus indices to local 0-based
        surv_buses = np.where(keep_bus)[0]
        bus_local_map = np.full(nb, -1, dtype=int)
        bus_local_map[surv_buses] = np.arange(n_keep_bus)

        # --- Build constraint matrix from surviving components ---

        # Bf_surv: surviving branch rows, remapped to local bus indices
        surv_br_idx = np.where(keep_branch)[0]
        f_local = bus_local_map[self.f_bus[keep_branch]]
        t_local = bus_local_map[self.t_bus[keep_branch]]
        b_surv = self.b_branch[keep_branch]

        row_bf = np.concatenate([np.arange(n_keep_branch),
                                 np.arange(n_keep_branch)])
        col_bf = np.concatenate([f_local, t_local])
        data_bf = np.concatenate([b_surv, -b_surv])
        Bf_surv = csr_array((data_bf, (row_bf, col_bf)),
                            shape=(n_keep_branch, n_keep_bus))

        # Bbus_surv from surviving branches: Bbus = Bf^T @ diag(1/b) @ Bf
        inv_b = 1.0 / b_surv
        Bf_scaled = csr_array((data_bf * np.concatenate([inv_b, inv_b]),
                               (row_bf, col_bf)),
                              shape=(n_keep_branch, n_keep_bus))
        Bbus_surv = Bf_surv.T @ Bf_scaled

        # Cg_surv: surviving generators mapped to local bus indices
        surv_gen_idx = np.where(keep_gen)[0]
        surv_gen_bus_local = bus_local_map[self.gen_bus_idx[keep_gen]]
        valid = surv_gen_bus_local >= 0
        Cg_surv = csr_array(
            (np.ones(valid.sum()),
             (surv_gen_bus_local[valid], np.where(valid)[0])),
            shape=(n_keep_bus, n_keep_gen))

        # Combined constraint matrix (CSC for HiGHS):
        # Rows: [0..n_keep_branch-1] ub_pos, [n_keep_branch..2*n_keep_branch-1] ub_neg,
        #       [2*n_keep_branch..2*n_keep_branch+n_keep_bus-1] eq
        # Columns: [0..n_keep_gen-1] Pg, [n_keep_gen..n_keep_gen+n_keep_bus-1] Va
        zeros_br_gen = csr_array((n_keep_branch, n_keep_gen))
        A_combined = sparse_bmat([
            [zeros_br_gen,  Bf_surv],
            [zeros_br_gen, -Bf_surv],
            [Cg_surv,      -Bbus_surv],
        ], format='csc')

        # LHS/RHS
        rate_surv = self.rate_a[keep_branch]
        lhs = np.concatenate([np.full(n_keep_branch, -1e20),
                              np.full(n_keep_branch, -1e20),
                              np.zeros(n_keep_bus)])
        rhs = np.concatenate([rate_surv, rate_surv, np.zeros(n_keep_bus)])

        # Bounds with generator scaling
        lb = self.lb_template[np.concatenate([np.where(keep_gen)[0],
                                              ng_total + surv_buses])].copy()
        ub = self.ub_template[np.concatenate([np.where(keep_gen)[0],
                                              ng_total + surv_buses])].copy()

        # Scale original generator Pmax by (1 - gen_state)
        orig_kept = keep_gen[:ng]
        n_orig_kept = int(orig_kept.sum())
        if n_orig_kept > 0:
            gs = gen_state[orig_kept]
            ub[:n_orig_kept] = self.orig_gen_pmax[orig_kept] * (1 - gs)
            lb[:n_orig_kept] = 0.0

        # Cost
        c_sub = self.c_obj[np.concatenate([np.where(keep_gen)[0],
                                           ng_total + surv_buses])]

        # Reference buses (one per island)
        if n_keep_branch > 0:
            adj_data = np.ones(2 * n_keep_branch)
            adj_row = np.concatenate([f_local, t_local])
            adj_col = np.concatenate([t_local, f_local])
            adj = csr_array((adj_data, (adj_row, adj_col)),
                            shape=(n_keep_bus, n_keep_bus))
            n_islands, labels = connected_components(adj, directed=False)

            surv_bus_types = self.bus_type[surv_buses]
            va_offset = n_keep_gen
            for island_id in range(n_islands):
                island_local = np.where(labels == island_id)[0]
                ref_found = False
                for li in island_local:
                    if surv_bus_types[li] == REF:
                        lb[va_offset + li] = 0.0
                        ub[va_offset + li] = 0.0
                        ref_found = True
                        break
                if not ref_found:
                    lb[va_offset + island_local[0]] = 0.0
                    ub[va_offset + island_local[0]] = 0.0

        # Solve via persistent HiGHS instance (no option validation overhead)
        n_vars = n_keep_gen + n_keep_bus
        n_cons = 2 * n_keep_branch + n_keep_bus

        lp = HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_cons
        lp.col_cost_ = c_sub.astype(np.float64)
        lp.col_lower_ = lb
        lp.col_upper_ = ub
        lp.row_lower_ = lhs
        lp.row_upper_ = rhs
        lp.a_matrix_.format_ = MatrixFormat.kColwise
        lp.a_matrix_.start_ = A_combined.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A_combined.indices.astype(np.int32)
        lp.a_matrix_.value_ = A_combined.data.astype(np.float64)
        lp.sense_ = ObjSense.kMinimize

        h = self._highs
        h.clearModel()
        h.passModel(lp)
        h.run()

        status = h.getModelStatus()
        if status != HighsModelStatus.kOptimal:
            return 100.0, 0

        x = np.array(h.getSolution().col_value)

        # Extract Pg for surviving original generators
        pg_surviving = x[:n_orig_kept] * self.baseMVA
        realpf = np.sum(pg_surviving)

        blackout_size = 100.0 * round(
            abs(realpf - self.totpf) / self.totpf * 1e8) / 1e8

        return blackout_size, 1


# ---------------------------------------------------------------------------
# func_dcopt (port of func_dcopt.m)
# ---------------------------------------------------------------------------

def func_dcopt(system_state, ppc0):
    """
    Compute the relative blackout size (%) given a system state vector.

    Port of func_dcopt.m — uses scipy.optimize.linprog (dual-simplex)
    for the DC-OPF, matching the patched qps_ot.m solver.

    Args:
        system_state: 1D array of length (nb + nl).
            - system_state[0:nb]: bus states (1 = failed, 0 = operational,
              fractional values for partial generator degradation)
            - system_state[nb:nb+nl]: branch states (1 = failed, 0 = operational)
        ppc0: base PYPOWER case dict (with branch capacities already set)

    Returns:
        blackout_size: float, percentage of total load not served
        flag: int, 1 if OPF converged, 0 otherwise
    """
    # Basic network info
    bus_dic = ppc0['bus'][:, BUS_I].copy()
    gen_dic = ppc0['gen'][:, GEN_BUS].copy()
    branch_dic = ppc0['branch'][:, [F_BUS, T_BUS]].copy()

    nb = len(bus_dic)
    ng = len(gen_dic)
    nl = len(branch_dic)

    # Parse system state
    bus_state = system_state[:nb].copy()
    branch_state = system_state[nb:nb + nl].copy()

    # Generator state derived from bus state
    gen_idx = np.array([np.where(bus_dic == g)[0][0] for g in gen_dic])
    gen_state = bus_state[gen_idx]

    # Convert fixed loads to dispatchable loads
    mpc = load2disp(ppc0)

    # No cost for original generators
    mpc['gencost'][:ng, :] = 0
    mpc['gencost'][:ng, MODEL] = POLYNOMIAL
    mpc['gencost'][:ng, NCOST] = 2

    # Zero out shunt conductance and undispatchable loads
    mpc['bus'][:, PD] = 0
    mpc['bus'][:, GS] = 0
    mpc['gen'][:ng, PMIN] = 0
    mpc['gen'][:ng, GEN_STATUS] = 1  # all generators in service

    # Total real power demand (sum of negative Pg from dispatchable loads)
    totpf = -np.sum(mpc['gen'][ng:, PG])

    # --- Remove failed buses ---
    removed_bus_idx = np.where(bus_state == 1)[0]
    removed_bus_no = bus_dic[removed_bus_idx]

    # Delete bus rows
    mpc['bus'] = np.delete(mpc['bus'], removed_bus_idx, axis=0)

    # --- Update generator capacity for partial degradation ---
    mpc['gen'][:ng, QMAX] *= (1 - gen_state)
    mpc['gen'][:ng, QMIN] *= (1 - gen_state)
    mpc['gen'][:ng, PMAX] *= (1 - gen_state)
    mpc['gen'][:ng, PMIN] *= (1 - gen_state)

    # Remove generators/loads at failed buses
    gen_at_removed = np.isin(mpc['gen'][:, GEN_BUS], removed_bus_no)
    mpc['gen'] = mpc['gen'][~gen_at_removed]
    mpc['gencost'] = mpc['gencost'][~gen_at_removed]

    # --- Remove failed branches ---
    removed_branch_idx = np.where(branch_state == 1)[0]
    br_from_removed = np.isin(branch_dic[:, 0], removed_bus_no)
    br_to_removed = np.isin(branch_dic[:, 1], removed_bus_no)
    all_removed_br = np.zeros(nl, dtype=bool)
    all_removed_br[removed_branch_idx] = True
    all_removed_br |= br_from_removed | br_to_removed

    mpc['branch'] = mpc['branch'][~all_removed_br]

    # --- Run DC-OPF ---
    results = _solve_dcopf(mpc)

    if not results['success']:
        return 100.0, 0

    # --- Compute blackout size ---
    n_surviving_gen = int(np.sum(gen_state != 1))

    surviving_gen_status = mpc['gen'][:n_surviving_gen, GEN_STATUS]
    surviving_gen_pg = results['gen'][:n_surviving_gen, PG]
    realpf = np.dot(surviving_gen_status, surviving_gen_pg)

    blackout_size = 100.0 * round(abs(realpf - totpf) / totpf * 1e8) / 1e8

    return blackout_size, 1


# ---------------------------------------------------------------------------
# Main: test against known Octave results
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    # Load case14 from MATPOWER .m file for consistency
    matpower_case14 = '/mnt/c/Projects/matpower8.1/data/case14.m'
    ppc0 = load_case(matpower_case14)
    ppc0 = add_branch_capacity(ppc0, alpha=2.0)

    nb = ppc0['bus'].shape[0]
    nl = ppc0['branch'].shape[0]
    print(f"Loaded case14: {nb} buses, {ppc0['gen'].shape[0]} generators, {nl} branches\n")

    tests = [
        ("All operational", {}),
        ("Branch 3 failed", {nb + 2: 1}),
        ("Branches 1,2 failed", {nb + 0: 1, nb + 1: 1}),
        ("Bus 1 failed", {0: 1}),
        ("Bus 1 + branch 3", {0: 1, nb + 2: 1}),
        ("Branches 3,8,10", {nb + 2: 1, nb + 7: 1, nb + 9: 1}),
        ("5 branches failed", {nb + 0: 1, nb + 1: 1, nb + 2: 1, nb + 3: 1, nb + 4: 1}),
        ("Buses 1,2 failed", {0: 1, 1: 1}),
        ("Buses 1,2,3 failed", {0: 1, 1: 1, 2: 1}),
        ("6 branches (8-13)", {nb + 7: 1, nb + 8: 1, nb + 9: 1, nb + 10: 1, nb + 11: 1, nb + 12: 1}),
        ("Gens at 40% cap", {0: 0.6, 1: 0.6}),
    ]

    for name, failures in tests:
        ss = np.zeros(nb + nl)
        for idx, val in failures.items():
            ss[idx] = val
        t0 = time.time()
        bs, flag = func_dcopt(ss, ppc0)
        dt = time.time() - t0
        print(f"  {name:30s}  blackout={bs:7.4f}%  flag={flag}  ({dt:.3f}s)")
