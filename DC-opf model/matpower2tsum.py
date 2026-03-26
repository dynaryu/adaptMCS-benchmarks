"""
Convert a MATPOWER case file (.m) to TSUM input files (nodes.json, edges.json, probs.json).

Usage:
    python matpower2tsum.py <case_file.m> [--output_dir <dir>] [--pf_branch <float>] [--include_bus_failures]

Example:
    python matpower2tsum.py /mnt/c/Projects/matpower8.1/data/case14.m --output_dir ./case14_tsum
    python matpower2tsum.py /mnt/c/Projects/matpower8.1/data/case14.m --output_dir ./case14_tsum_bus --include_bus_failures
"""

import re
import json
import argparse
import numpy as np
from pathlib import Path


def parse_matpower_matrix(text, field_name):
    """Extract a matrix from MATPOWER .m file text for a given field (e.g., 'mpc.bus')."""
    # Match pattern: mpc.field = [ ... ];
    pattern = rf'mpc\.{field_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    block = match.group(1)
    rows = []
    for line in block.strip().split('\n'):
        line = line.split('%')[0].strip().rstrip(';')  # remove comments and trailing semicolons
        if not line:
            continue
        vals = line.split()
        if vals:
            rows.append([float(v) for v in vals])
    return np.array(rows) if rows else None


def parse_matpower_case(filepath):
    """Parse a MATPOWER .m case file and return bus, gen, branch matrices."""
    with open(filepath, 'r') as f:
        text = f.read()

    bus = parse_matpower_matrix(text, 'bus')
    gen = parse_matpower_matrix(text, 'gen')
    branch = parse_matpower_matrix(text, 'branch')

    if bus is None or branch is None:
        raise ValueError(f"Could not parse bus/branch data from {filepath}")

    # Extract baseMVA
    m = re.search(r'mpc\.baseMVA\s*=\s*(\d+\.?\d*)', text)
    baseMVA = float(m.group(1)) if m else 100.0

    return {'bus': bus, 'gen': gen, 'branch': branch, 'baseMVA': baseMVA}


def layout_buses(bus_data):
    """
    Generate x, y coordinates for buses using a simple force-directed-like layout.
    Uses bus voltage angle and magnitude as initial positions.
    """
    n = len(bus_data)
    # Use voltage angle (col 8) as x and voltage magnitude (col 7) as y
    angles = bus_data[:, 8]  # Va in degrees
    vm = bus_data[:, 7]      # Vm in p.u.

    # Convert to Cartesian-like coordinates
    angles_rad = np.radians(angles)
    x = vm * np.cos(angles_rad)
    y = vm * np.sin(angles_rad)

    # Normalize to [0, 10] range
    if x.max() != x.min():
        x = 10 * (x - x.min()) / (x.max() - x.min())
    if y.max() != y.min():
        y = 10 * (y - y.min()) / (y.max() - y.min())

    return x, y


def get_bus_type_label(bus_type_code, bus_id, gen_buses, pd):
    """Map MATPOWER bus type to a TSUM node type string."""
    # MATPOWER bus types: 1=PQ (load), 2=PV (generator), 3=ref (slack)
    if bus_type_code == 3:
        return "source"
    elif bus_type_code == 2 or bus_id in gen_buses:
        return "source"
    elif pd > 0:
        return "output"
    else:
        return "transmission"


def create_nodes(mpc, include_bus_failures=False):
    """Create TSUM nodes.json from MATPOWER bus data.

    When include_bus_failures=True, each bus is split into an external node
    (bus{id}) and an internal node (bus{id}_int). A virtual edge connects
    them to represent bus failure. All branches attach to _int nodes.
    """
    bus = mpc['bus']
    gen = mpc['gen']
    gen_buses = set(gen[:, 0].astype(int)) if gen is not None else set()

    x_coords, y_coords = layout_buses(bus)

    nodes = {}
    for i, row in enumerate(bus):
        bus_id = int(row[0])
        bus_type = int(row[1])
        pd = row[2]  # real power demand (MW)

        node_type = get_bus_type_label(bus_type, bus_id, gen_buses, pd)
        x = round(float(x_coords[i]), 4)
        y = round(float(y_coords[i]), 4)

        if include_bus_failures:
            # External node: carries bus identity and type info
            ext_node = {"x": x, "y": y, "type": node_type}
            if bus_id in gen_buses:
                gen_row = gen[gen[:, 0] == bus_id][0]
                ext_node["capacity"] = float(gen_row[8])
                ext_node["unit"] = "MW"
            elif pd > 0:
                ext_node["capacity"] = float(pd)
                ext_node["unit"] = "MW"
            nodes[f"bus{bus_id}"] = ext_node

            # Internal node: branches connect here
            nodes[f"bus{bus_id}_int"] = {
                "x": round(x + 0.15, 4),
                "y": round(y + 0.15, 4),
                "type": "transmission",
            }
        else:
            node = {"x": x, "y": y, "type": node_type}
            if bus_id in gen_buses:
                gen_row = gen[gen[:, 0] == bus_id][0]
                node["capacity"] = float(gen_row[8])
                node["unit"] = "MW"
            elif pd > 0:
                node["capacity"] = float(pd)
                node["unit"] = "MW"
            nodes[f"bus{bus_id}"] = node

    return nodes


def create_edges(mpc, include_bus_failures=False):
    """Create TSUM edges.json from MATPOWER branch data.

    When include_bus_failures=True, branches connect to bus{id}_int nodes,
    and a virtual edge (vbus{id}) connects bus{id} <-> bus{id}_int for
    each bus.
    """
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']

    edges = {}

    # Virtual edges for bus failures
    if include_bus_failures:
        gen_buses = set(gen[:, 0].astype(int)) if gen is not None else set()
        for row in bus:
            bus_id = int(row[0])
            is_gen = bus_id in gen_buses
            edge = {
                "from": f"bus{bus_id}",
                "to": f"bus{bus_id}_int",
                "directed": False,
                "component_type": "generator_bus" if is_gen else "ordinary_bus",
            }
            edges[f"vbus{bus_id}"] = edge

    # Branch edges
    node_suffix = "_int" if include_bus_failures else ""
    for i, row in enumerate(branch):
        fbus = int(row[0])
        tbus = int(row[1])

        edge = {
            "from": f"bus{fbus}{node_suffix}",
            "to": f"bus{tbus}{node_suffix}",
            "directed": False,
        }

        rate_a = row[5]
        if rate_a > 0:
            edge["capacity"] = float(rate_a)
            edge["unit"] = "MVA"

        edges[f"br{i+1}"] = edge

    return edges


def create_probs(mpc, pf_branch=0.01, include_bus_failures=False):
    """
    Create TSUM probs.json.

    Branch components are binary:
      state 0 = failed, state 1 = operational

    When include_bus_failures=True, bus components are added:
      - Generator buses: 4-state (from main_DCPF.py probabilistic model)
          state 0 = removed          (p=0.01)
          state 1 = 40% capacity     (p=0.19)
          state 2 = 80% capacity     (p=0.30)
          state 3 = fully operational (p=0.50)
      - Ordinary buses: binary
          state 0 = failed (p=0.01), state 1 = operational (p=0.99)
    """
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']

    probs = {}

    # Bus failure probabilities
    if include_bus_failures:
        gen_buses = set(gen[:, 0].astype(int)) if gen is not None else set()
        for row in bus:
            bus_id = int(row[0])
            if bus_id in gen_buses:
                # Multi-state generator bus (from main_DCPF.py distrGen)
                # CDF thresholds: [0.01, 0.2, 0.5, 1.0]
                # capacity states: [1(removed), 0.6(40%cap), 0.2(80%cap), 0(full)]
                probs[f"vbus{bus_id}"] = {
                    "0": {"p": 0.01},   # removed
                    "1": {"p": 0.19},   # 40% capacity
                    "2": {"p": 0.30},   # 80% capacity
                    "3": {"p": 0.50},   # fully operational
                }
            else:
                # Binary ordinary bus (from main_DCPF.py distrOrdinaryBus)
                probs[f"vbus{bus_id}"] = {
                    "0": {"p": 0.01},
                    "1": {"p": 0.99},
                }

    # Branch failure probabilities
    for i in range(len(branch)):
        probs[f"br{i+1}"] = {
            "0": {"p": pf_branch},
            "1": {"p": round(1.0 - pf_branch, 10)},
        }

    return probs


def main():
    parser = argparse.ArgumentParser(
        description='Convert a MATPOWER case file to TSUM input files.')
    parser.add_argument('case_file', type=str,
                        help='Path to MATPOWER .m case file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <case_name>_tsum)')
    parser.add_argument('--pf_branch', type=float, default=0.01,
                        help='Branch failure probability (default: 0.01)')
    parser.add_argument('--include_bus_failures', action='store_true',
                        help='Model bus failures as virtual edges (multi-state for generators)')
    args = parser.parse_args()

    case_path = Path(args.case_file)
    if not case_path.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Parse the MATPOWER case
    mpc = parse_matpower_case(case_path)

    nb = len(mpc['bus'])
    ng = len(mpc['gen']) if mpc['gen'] is not None else 0
    nl = len(mpc['branch'])
    print(f"Parsed {case_path.name}: {nb} buses, {ng} generators, {nl} branches")
    if args.include_bus_failures:
        print(f"  Bus failures enabled: {nb} virtual edges ({ng} multi-state generators)")

    # Create output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        case_name = case_path.stem
        out_dir = case_path.parent / f"{case_name}_tsum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate the three TSUM input files
    bus_fail = args.include_bus_failures
    nodes = create_nodes(mpc, include_bus_failures=bus_fail)
    edges = create_edges(mpc, include_bus_failures=bus_fail)
    probs = create_probs(mpc, pf_branch=args.pf_branch, include_bus_failures=bus_fail)

    for name, data in [('nodes.json', nodes), ('edges.json', edges), ('probs.json', probs)]:
        out_path = out_dir / name
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"  Written: {out_path}")

    print(f"\nTSUM input files created in: {out_dir}")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}, Components with probs: {len(probs)}")


if __name__ == '__main__':
    main()
