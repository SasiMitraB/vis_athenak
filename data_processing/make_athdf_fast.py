# A parallelized script for converting a collection of .bin files to .athdf/.xdmf files
# Running this file : 
# # Basic run (will automatically use most of your CPU cores)
## ---- python make_athdf_fast.py "my_data_prefix_" my_data_prefix can be replaced with 'KH_hydro_w' or prefix common in your bin file. 

# If you want to limit it to exactly 4 cores:
## ---- python make_athdf_fast.py "my_data_prefix_" -c 4 
import os
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm # Optional, but gives a beautiful progress bar

# AthenaK modules
import bin_convert

# We moved the conversion logic into its own function so the workers can execute it
def convert_single_file(fname):
    athdf_name = fname.replace(".bin", ".athdf")
    xdmf_name = athdf_name + ".xdmf"
    
    # Read, format, and write
    filedata = bin_convert.read_binary(fname)
    bin_convert.write_athdf(athdf_name, filedata)
    bin_convert.write_xdmf_for(xdmf_name, os.path.basename(athdf_name), filedata)
    
    return fname

def main(**kwargs):
    # Get the root name for the files
    files = glob.glob(kwargs['file_stem'] + '*.bin')
    if len(files) < 1:
        print(f"No files found with stem {kwargs['file_stem']}")
        quit()

    total = len(files)
    
    # Determine how many cores to use. Default to all available minus 1 (to keep your PC responsive)
    max_workers = kwargs.get('cores', max(1, os.cpu_count() - 1))
    
    print(f"Found {total} files. Starting parallel conversion using {max_workers} CPU cores...")

    # ProcessPoolExecutor spins up separate Python processes to bypass the GIL
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the conversion function to our list of files, and wrap it in tqdm for a progress bar
        list(tqdm(executor.map(convert_single_file, files), total=total))
        
    print("Conversion complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_stem', help='path to files, excluding .#.bin')
    parser.add_argument('-c', '--cores', type=int, default=os.cpu_count() - 1,
                        help='Number of CPU cores to use (default: All cores - 1)')
    args = parser.parse_args()
    main(**vars(args))