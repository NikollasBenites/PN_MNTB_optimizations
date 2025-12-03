import os
from neuron import h

# Mechanisms from your .mod files (exact SUFFIX names, case-sensitive)
MECHANISMS = [
    "NaCh_nmb",    # nach_nmb.mod
    "HT_dth_nmb",  # kht_dth_nmb.mod
    "LT_dth",      # klt_dth.mod
    "ka",          # ka.mod
    "IH_nmb",      # ih_nmb.mod
    "leak",        # leak.mod
]


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:

      1. Load ONE compiled NEURON mechanism library from optimization/mod.
      2. Try inserting each known mechanism SUFFIX into a test Section.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    # Candidate locations for the library produced by:
    #   cd optimization/mod
    #   nrnivmodl
    candidates = [
        os.path.join(repo_root, "optimization", "mod", "arm64", "libnrnmech.dylib"),
        os.path.join(repo_root, "optimization", "mod", "arm64", ".libs", "libnrnmech.so"),
        # Fallbacks if layout ever changes:
        os.path.join(repo_root, "optimization", "mod", "libnrnmech.dylib"),
        os.path.join(repo_root, "optimization", "mod", ".libs", "libnrnmech.so"),
    ]

    print("Looking for NEURON mechanism library in:")
    for c in candidates:
        print("  ", c)

    lib = None
    for c in candidates:
        if os.path.exists(c):
            lib = c
            break

    if lib is None:
        raise RuntimeError(
            "Could not find libnrnmech.{dylib,so} under optimization/mod.\n"
            "Check that the 'Compile MOD files (macOS)' step ran 'nrnivmodl' "
            "successfully in optimization/mod."
        )

    print(f"Loading mechanisms from: {lib}")
    h.nrn_load_dll(lib)

    # Try inserting each mechanism into a dummy soma
    soma = h.Section(name="soma")

    for mech in MECHANISMS:
        print(f"Testing mechanism: {mech}")
        try:
            soma.insert(mech)
        except Exception as e:
            raise AssertionError(f"Failed to insert mechanism '{mech}': {e}")

    print("All mechanisms inserted successfully.")
