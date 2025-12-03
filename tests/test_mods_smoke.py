import os
from neuron import h

# Your exact SUFFIX names from your .mod files
MECHANISMS = [
    "NaCh_nmb",
    "KHT_nmb",
    "KLT_nmb",
    "KA",
    "Ih_nmb",
    "Leak",
]

# Directories where compiled mechanisms (arm64) will be located
MOD_PARENT_DIRS = [
    "optimization",
    "optimization/3D_and_bar_graphs",
]

def test_load_and_insert_all_mechanisms():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    # 1) Load compiled mechanism libraries
    for parent in MOD_PARENT_DIRS:
        mech_dir = os.path.join(repo_root, parent, "arm64")

        if not os.path.isdir(mech_dir):
            raise RuntimeError(f"Compiled directory not found: {mech_dir}")

        candidates = [
            os.path.join(mech_dir, "libnrnmech.dylib"),   # macOS
            os.path.join(mech_dir, "libnrnmech.so"),      # fallback
        ]

        lib = None
        for c in candidates:
            if os.path.exists(c):
                lib = c
                break

        if lib is None:
            raise RuntimeError(
                f"Could not find compiled library in {mech_dir}: looked for .dylib/.so")

        print(f"Loading mechanisms from: {lib}")
        h.nrn_load_dll(lib)

    # 2) Insert each mechanism into a Section
    soma = h.Section(name="soma")

    for mech in MECHANISMS:
        print(f"Testing mechanism: {mech}")
        try:
            soma.insert(mech)
        except Exception as e:
            raise AssertionError(f"Failed to insert mechanism '{mech}': {e}")

    print("All mechanisms inserted successfully.")
