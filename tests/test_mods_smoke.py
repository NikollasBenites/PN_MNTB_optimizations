import os
from neuron import h

# Mechanisms from your .mod files
MECHANISMS = [
    "NaCh_nmb",
    "KHT_nmb",
    "KLT_nmb",
    "KA",
    "Ih_nmb",   # if this later fails, we can adjust case to 'IH_nmb'
    "Leak",
]


def _find_libs_by_directory(repo_root: str):
    """
    Walk the repo tree and find NEURON mechanism libraries created by nrnivmodl.
    For each directory, keep only one library (prefer .dylib over .so).
    Returns a dict: {dirpath: libpath}.
    """
    libs_by_dir = {}

    for dirpath, dirnames, filenames in os.walk(repo_root):
        has_dylib = "libnrnmech.dylib" in filenames
        has_so = "libnrnmech.so" in filenames

        chosen = None
        if has_dylib:
            chosen = os.path.join(dirpath, "libnrnmech.dylib")
        elif has_so:
            chosen = os.path.join(dirpath, "libnrnmech.so")

        if chosen is not None:
            libs_by_dir[dirpath] = chosen

    return libs_by_dir


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:
      1. Find NEURON mechanism libraries (one per directory).
      2. Load each with h.nrn_load_dll(), ignoring 'already exists' duplicates.
      3. Try inserting each known mechanism SUFFIX into a test Section.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    libs_by_dir = _find_libs_by_directory(repo_root)
    libs = list(libs_by_dir.values())

    print("Found NEURON mechanism libraries (one per directory):")
    for lib in libs:
        print("  ", lib)

    if not libs:
        raise RuntimeError(
            "No NEURON mechanism libraries (libnrnmech.dylib/.so) found in repo.\n"
            "Check that the 'Compile MOD files (macOS)' step is running 'nrnivmodl' "
            "and that it succeeds."
        )

    # 1) Load each library.
    # If NEURON says 'user defined name already exists', we just skip that lib.
    for lib in libs:
        print(f"Loading mechanisms from: {lib}")
        try:
            h.nrn_load_dll(lib)
        except RuntimeError as e:
            msg = str(e)
            if "user defined name already exists" in msg or "already exists" in msg:
                print(f"Skipping {lib} (mechanisms already loaded)")
                continue
            # Any other error is real
            raise

    # 2) Try inserting each mechanism into a dummy section
    soma = h.Section(name="soma")

    for mech in MECHANISMS:
        print(f"Testing mechanism: {mech}")
        try:
            soma.insert(mech)
        except Exception as e:
            raise AssertionError(f"Failed to insert mechanism '{mech}': {e}")

    print("All mechanisms inserted successfully.")
