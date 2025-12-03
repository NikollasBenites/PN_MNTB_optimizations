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


def _find_all_libs(repo_root: str):
    """
    Walk the repo tree and find all NEURON mechanism libraries created by nrnivmodl.
    We look for libnrnmech.dylib (mac) or libnrnmech.so (unix) anywhere under repo_root.
    """
    libs = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        for fname in filenames:
            if fname.startswith("libnrnmech") and (
                fname.endswith(".dylib") or fname.endswith(".so")
            ):
                full = os.path.join(dirpath, fname)
                libs.append(full)
    return libs


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:
      1. Find all libnrnmech.* produced by nrnivmodl in the repo.
      2. Load each with h.nrn_load_dll().
      3. Try inserting each known mechanism SUFFIX into a test Section.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    libs = _find_all_libs(repo_root)
    print("Found NEURON mechanism libraries:")
    for lib in libs:
        print("  ", lib)

    if not libs:
        raise RuntimeError(
            "No NEURON mechanism libraries (libnrnmech.dylib/.so) found in repo.\n"
            "Check that the 'Compile MOD files (macOS)' step is running 'nrnivmodl' "
            "and that it succeeds."
        )

    # 1) Load each library
    for lib in libs:
        print(f"Loading mechanisms from: {lib}")
        h.nrn_load_dll(lib)

    # 2) Try inserting each mechanism into a dummy section
    soma = h.Section(name="soma")

    for mech in MECHANISMS:
        print(f"Testing mechanism: {mech}")
        try:
            soma.insert(mech)
        except Exception as e:
            raise AssertionError(f"Failed to insert mechanism '{mech}': {e}")

    print("All mechanisms inserted successfully.")
