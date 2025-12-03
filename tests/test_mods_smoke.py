import os
from neuron import h

# Mechanisms from your .mod files
MECHANISMS = [
    "NaCh_nmb",
    "KHT_nmb",
    "KLT_nmb",
    "KA",
    "Ih_nmb",
    "Leak",
]

# Directories (relative to repo root) that contain mod files and libnrnmech.dylib
MOD_DIRS = [
    "optimization/mod",
    "optimization/3D_and_bar_graphs/mod",
]


def _load_all_libraries(repo_root: str):
    """
    For each mod directory, load the libnrnmech.dylib produced by nrnivmodl.
    """
    for rel in MOD_DIRS:
        mod_dir = os.path.join(repo_root, rel)
        print(f"Checking mod dir: {mod_dir}")
        if not os.path.isdir(mod_dir):
            raise RuntimeError(f"MOD directory not found: {mod_dir}")

        # NEURON is linking ./libnrnmech.dylib directly in this directory
        lib_path = os.path.join(mod_dir, "libnrnmech.dylib")
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"Expected library not found: {lib_path}. "
                f"Did nrnivmodl mod run in {mod_dir}?"
            )

        print(f"Loading mechanisms from: {lib_path}")
        h.nrn_load_dll(lib_path)


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:
      1. Load all compiled NEURON mechanism libraries from the two mod dirs.
      2. Try inserting each known SUFFIX into a test Section.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    # 1) Load compiled NEURON libraries
    _load_all_libraries(repo_root)

    # 2) Try inserting each mechanism
    soma = h.Section(name="soma")

    for mech in MECHANISMS:
        print(f"Testing mechanism: {mech}")
        try:
            soma.insert(mech)
        except Exception as e:
            raise AssertionError(f"Failed to insert mechanism '{mech}': {e}")

    print("All mechanisms inserted successfully.")
