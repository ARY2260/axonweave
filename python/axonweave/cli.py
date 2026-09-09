from __future__ import annotations
import argparse
from .data.installer import install_male_cns

def main(argv=None):
    p = argparse.ArgumentParser(prog="axonweave")
    sub = p.add_subparsers(dest="cmd", required=True)
    substrate = sub.add_parser("substrate", help="manage biological substrates")
    ss = substrate.add_subparsers(dest="action", required=True)
    install = ss.add_parser("install", help="install a versioned substrate")
    install.add_argument("name")
    install.add_argument("--root", default=None)
    install.add_argument("--include-stats", action="store_true")
    install.add_argument("--include-synapses", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "substrate" and args.action == "install":
        if args.name != "male-cns:v1.0":
            raise SystemExit(f"AXW101: unsupported substrate {args.name!r}; available: male-cns:v1.0")
        brain = install_male_cns(args.root, args.include_synapses, args.include_stats)
        print(f"Installed {args.name}: {brain.n_neurons} neurons, {brain.graph.n_edges} graph edges")
