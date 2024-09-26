from opengert.RT.process import SimulationConfig, SimulationRunner
import argparse

def main(args):
    # Create the configuration dictionary from parsed arguments
    config = {
        "freq": args.freq,
        "synthetic_array": args.synthetic_array,
        "tx_array_pattern": args.tx_array_pattern,
        "tx_array_pol": args.tx_array_pol,
        "rx_array_pattern": args.rx_array_pattern,
        "rx_array_pol": args.rx_array_pol,
        "tx_loc": args.tx_loc,
        "max_depth": args.max_depth,
        "edge_diffraction": args.edge_diffraction,
        "num_samples": args.num_samples,
        "batch_size_cir": args.batch_size_cir,
        "target_num_cirs": args.target_num_cirs,
        "max_gain_db": args.max_gain_db,
        "min_gain_db": args.min_gain_db,
        "min_dist": args.min_dist,
        "max_dist": args.max_dist,
        "delay_bins": args.delay_bins,
        "subcarrier_spacing": args.subcarrier_spacing,
        "geo_data_dir": args.geo_data_dir,
        "xml_filename": args.xml_filename,
        "save_dir": args.save_dir,
    }

    # Initialize and run the simulation using the config
    simulation_config = SimulationConfig(config)
    simulation_config.save_config()

    runner = SimulationRunner(config)
    runner.run_simulation()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Generate channel impulse responses with Sionna RT.")

    # Add arguments corresponding to the configuration parameters
    parser.add_argument("--freq", type=float, default=1.8e9, help="Frequency in Hz")
    parser.add_argument("--synthetic_array", type=bool, default=True, help="Use synthetic array (True/False)")
    parser.add_argument("--tx_array_pattern", type=str, default="tr38901", help="Transmit array pattern")
    parser.add_argument("--tx_array_pol", type=str, default="V", help="Transmit array polarization")
    parser.add_argument("--rx_array_pattern", type=str, default="dipole", help="Receive array pattern")
    parser.add_argument("--rx_array_pol", type=str, default="V", help="Receive array polarization")
    parser.add_argument("--tx_loc", type=float, nargs=3, default=[-1178, -310, 91.0], help="Transmitter location (3 values: x, y, z)")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum simulation depth")
    parser.add_argument("--edge_diffraction", type=bool, default=True, help="Use edge diffraction (True/False)")
    parser.add_argument("--num_samples", type=float, default=1e7, help="Number of samples")
    parser.add_argument("--batch_size_cir", type=int, default=50, help="Batch size for CIR")
    parser.add_argument("--target_num_cirs", type=int, default=100, help="Target number of CIRs")
    parser.add_argument("--max_gain_db", type=float, default=0, help="Maximum gain in dB")
    parser.add_argument("--min_gain_db", type=float, default=-140, help="Minimum gain in dB")
    parser.add_argument("--min_dist", type=float, default=10, help="Minimum distance")
    parser.add_argument("--max_dist", type=float, default=1300, help="Maximum distance")
    parser.add_argument("--delay_bins", type=int, default=250, help="Number of delay bins")
    parser.add_argument("--subcarrier_spacing", type=float, default=15e3, help="Subcarrier spacing in Hz")
    parser.add_argument("--geo_data_dir", type=str, default="/home/gtpropagation/Documents/stadik/data/", help="Directory for geographic data")
    parser.add_argument("--xml_filename", type=str, default="example_scene.xml", help="Directory for geographic data")
    parser.add_argument("--save_dir", type=str, default="./cir_data/", help="Directory to save output")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
