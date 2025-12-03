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


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:

      1. Load ONE compiled NEURON mechanism library from optimization/mod.
      2. Try inserting each known mechanism SUFFIX into a test Section.

    We only need to load one libnrnmech; compiling in the other folder is
    already checked by the 'Compile MOD files' step.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    # Candidate locations for the library produced by:
    #   cd optimization/mod
    #   nrnivmodl
    #
    # From the CI log we know NEURON is creating:
    #   optimization/mod/arm64/libnrnmech.dylib
    #   optimization/mod/arm64/.libs/libnrnmech.so
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
