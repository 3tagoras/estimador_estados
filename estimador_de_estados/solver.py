"""
solver.py
Load a power system test file (PSS/E .raw) from a local folder,
convert it to a pandapower net, and solve the power flow.
"""
import argparse
from pathlib import Path

import andes
import pandapower as pp
from andes.interop.pandapower import to_pandapower


def convert_raw_to_pandapower(raw_path, output_path=None, run_powerflow=True):
    """
    Convert a PSS/E RAW file into a pandapower network.

    Parameters
    ----------
    raw_path : str
        Path to the PSS/E RAW file.

    output_path : str or None
        Optional output file for the pandapower JSON network.

    run_powerflow : bool
        Whether to run a power flow after conversion.
    """

    raw_path = Path(raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_path}")

    print("=" * 70)
    print("Loading PSS/E RAW file with ANDES...")
    print("=" * 70)

    # Load RAW into ANDES
    ss = andes.load(
        str(raw_path),
        setup=False,
        no_output=True,
        default_config=True,
    )

    print("Setting up ANDES system...")
    ss.setup()

    print("=" * 50)
    print("Converting ANDES system to pandapower...")
    print("=" * 50)

    # Convert to pandapower
    net = to_pandapower(ss)

    print("Conversion completed successfully.")

    print()
    print("Network Summary")
    print("-" * 70)
    print(f"Buses        : {len(net.bus)}")
    print(f"Lines        : {len(net.line)}")
    print(f"Transformers : {len(net.trafo)}")
    print(f"Loads        : {len(net.load)}")
    print(f"Generators   : {len(net.gen)}")
    print(f"Ext Grids    : {len(net.ext_grid)}")

    if run_powerflow:
        print()
        print("=" * 70)
        print("Running pandapower power flow...")
        print("=" * 70)

        try:
            pp.runpp(net)
            if net.converged:
                print("Power flow converged successfully.")

            print()
            pp.diagnostic(net)
            print("Bus Voltages")
            print("-" * 70)
            print(net.res_bus[["vm_pu", "va_degree"]].head())

        except Exception as e:
            print(f"Power flow failed: {e}")

    # Save network
    if output_path:
        output_path = Path(output_path)

        print()
        print("=" * 70)
        print(f"Saving pandapower network to: {output_path}")
        print("=" * 70)

        pp.to_json(net, str(output_path))

        print("Saved successfully.")

    return net


if __name__ == "__main__":
    """ parser = argparse.ArgumentParser(
        description="Convert PSS/E RAW files to pandapower"
    )

    parser.add_argument(
        "raw_file",
        help="Path to the PSS/E RAW file"
    )

    parser.add_argument(
        "--output",
        default="network.json",
        help="Output pandapower JSON file"
    )

    parser.add_argument(
        "--no-pf",
        action="store_true",
        help="Skip power flow calculation"
    )

    args = parser.parse_args() """

    

    convert_raw_to_pandapower(
        raw_path= r"C:\Users\DELL\Documents\Areas\MCD\Proyecto\estimador_estados\data\interim\case69.raw",# args.raw_file,
        output_path= r"C:\Users\DELL\Documents\Areas\MCD\Proyecto\estimador_estados\data\processed\case69.json",# args.output,
        run_powerflow=not False,# args.no_pf,
    )