
"""
convert_meshes.py
Process the .obj files and make them waterproof and simplified.
"""
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

GRASPDIR = "/home/danial/Documents/acronym/grasps"
OBJDIR = "/home/danial/Documents/acronym/meshes/"
MANIFOLD_PATH = "/home/danial/Documents/Manifold/build/manifold"
SIMPLIFY_PATH = "/home/danial/Documents/Manifold/build/simplify"

## Grab the object file names from the grasp directory
hashes = []
for fname in os.listdir(GRASPDIR):
    tokens = fname.split("_")
    assert(len(tokens) == 3)
    hashes.append(tokens[1])

## Define function to process a single file
def process_hash(h):
    """Process a single object file by calling a subshell with the mesh processing script.

    Args:
        h (string): the hash denoting the file type
    """
    obj = OBJDIR + 'models/' + h + ".obj"

    if not os.path.isfile(obj):
        return
    
    # Waterproof the object
    temp_name = f"temp.{h}.watertight.obj"
    completed = subprocess.run(["timeout", "-sKILL", "30", MANIFOLD_PATH, obj, temp_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if completed.returncode != 0:
        print(f"Skipping object (manifold failed): {h}")
        return
            
    # Simplify the object
    outfile = OBJDIR + "simplified/" + h + ".obj"
    completed = subprocess.run([SIMPLIFY_PATH, "-i", temp_name, "-o", outfile, "-m", "-r", "0.02"],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if completed.returncode != 0:
        print(f"Skipping object (simplify failed): {h}")
        
## Issue the commands in a multiprocessing pool     
with Pool(cpu_count()-3) as p:
    examples = list(
        tqdm(
            p.imap_unordered(process_hash, hashes),
            total=len(hashes)
        )
    )
