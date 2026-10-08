"""Velocity-estimation entry point.

A velocity estimator is not included. Computing velocities requires calibrated
coordinates, a frame rate, filtering settings, and boundary conventions.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder")
    parser.add_argument("--json_folder")
    parser.add_argument("--output_json")
    parser.parse_args()
    parser.exit(
        2,
        "Velocity estimation is not implemented. "
        "Provide an estimator with specified coordinate units, frame rate, "
        "filtering settings, and boundary conventions before use. "
        "See 1_keypoints/readme.md. No output was written.\n",
    )


if __name__ == "__main__":
    main()
