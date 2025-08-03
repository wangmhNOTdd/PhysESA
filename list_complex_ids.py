import pickle
import argparse
import os
import sys
from torch_geometric.data import Data

# Add project root to sys.path to allow importing Stage2Dataset if needed for unpickling
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'experiments', 'stage2'))
# It might be necessary to import the class of the pickled objects
# even if it's not directly used in this script.
try:
    from train_stage2 import Stage2Dataset
except ImportError:
    print("Warning: Could not import Stage2Dataset. Unpickling might fail if the .pkl file contains custom classes.")


def list_ids_from_pkl(pkl_path):
    """
    Loads a .pkl file containing a list of PyG Data objects and prints their complex_id.
    """
    if not os.path.exists(pkl_path):
        print(f"Error: File not found at {pkl_path}")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data_list = pickle.load(f)
    except Exception as e:
        print(f"Error loading or unpickling file: {e}")
        return

    if not isinstance(data_list, list):
        print(f"Error: Expected a list of data objects, but found type {type(data_list)}.")
        return

    print(f"--- Complex IDs in {os.path.basename(pkl_path)} ---")
    for i, data in enumerate(data_list):
        complex_id = getattr(data, 'complex_id', 'N/A')
        print(f"{i+1: >4}: {complex_id}")
    print(f"--- Total complexes: {len(data_list)} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List complex IDs from a .pkl dataset file.")
    parser.add_argument("pkl_file", type=str, help="Path to the .pkl file (e.g., experiments/stage2/test.pkl).")
    
    args = parser.parse_args()
    
    list_ids_from_pkl(args.pkl_file)