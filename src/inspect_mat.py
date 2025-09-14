import scipy.io
import os
import numpy as np

def inspect_mat_file(file_path):
    """
    Loads a .mat file and prints its structure in a human-readable format.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return

    print(f"\n{'='*80}")
    print(f"--- Inspecting File: {os.path.basename(file_path)} ---")
    print(f"{'='*80}")

    try:
        # Load the .mat file
        mat = scipy.io.loadmat(file_path)

        # 1. Print top-level keys
        print("\n[1. Top-Level Keys]")
        print("--------------------")
        print(f"Found keys: {list(mat.keys())}")
        
        # Find the main data key (not the headers)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        print(f"==> Identified main data key as: '{data_key}'")

        # 2. Inspect the main data structure
        print(f"\n[2. Structure of '{data_key}']")
        print("---------------------------")
        # The data is usually nested like this: mat['B0005'][0, 0]
        main_struct = mat[data_key][0, 0]
        print(f"Type: {type(main_struct)}")
        print(f"Available fields (dtype.names): {main_struct.dtype.names}")

        # 3. Inspect the 'cycle' field
        if 'cycle' in main_struct.dtype.names:
            print("\n[3. 'cycle' Field Details]")
            print("-------------------------")
            all_cycles = main_struct['cycle'][0]
            print(f"Number of cycles found: {len(all_cycles)}")

            # 4. Inspect a few individual cycles to see their structure
            print("\n[4. Sample Cycle Structures]")
            print("----------------------------")
            for i in [0, 5, -1]: # Look at the first, a middle, and the last cycle
                if abs(i) >= len(all_cycles): continue
                
                print(f"\n--- Cycle #{i} ---")
                cycle_struct = all_cycles[i]
                print(f"Available fields: {cycle_struct.dtype.names}")
                
                if 'type' in cycle_struct.dtype.names:
                    print(f"  - type: {cycle_struct['type'][0]}")

                if 'data' in cycle_struct.dtype.names:
                    print("  - 'data' field found. Inspecting its sub-fields:")
                    data_sub_struct = cycle_struct['data'][0, 0]
                    print(f"    Available sub-fields: {data_sub_struct.dtype.names}")

                    # Check for our target fields
                    if 'Capacity' in data_sub_struct.dtype.names:
                        print(f"    ==> 'Capacity' FOUND!")
                    if 'Voltage_measured' in data_sub_struct.dtype.names:
                        print(f"    ==> 'Voltage_measured' FOUND!")
                    if 'Current_measured' in data_sub_struct.dtype.names:
                        print(f"    ==> 'Current_measured' FOUND!")
                    if 'Temperature_measured' in data_sub_struct.dtype.names:
                        print(f"    ==> 'Temperature_measured' FOUND!")
        else:
            print("ERROR: 'cycle' field not found in main structure.")

    except Exception as e:
        print(f"An error occurred during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Adjust this path to match your project structure exactly.
    # It should point to ONE of the .mat files.
    # We are going up one level from `src` to the project root, then down into the data folder.
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_to_inspect = os.path.join(
        project_root, 
        'data', 
        '5. Battery Data Set', 
        '1. BatteryAgingARC-FY08Q4', 
        'B0005.mat'
    )
    
    inspect_mat_file(file_to_inspect)