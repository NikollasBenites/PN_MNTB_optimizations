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

# Parent directories that contain a "mod" folder and where you run nrnivmodl mod
MOD_PARENT_DIRS = [
    "optimization",
    "optimization/3D_and_bar_graphs",
]


def _find_mech_dir(parent_dir: str) -> str:
    """
    Given a parent directory (e.g., 'optimization'),
    return the path to the compiled mechanisms directory.

    On your local M1/M2 Mac:
        optimization/arm64
    On GitHub's Intel macOS runners:
        optimization/x86_64
    """
    candidates = ["arm64", "x86_64"]

    for sub in candidates:
        candidate = os.path.join(parent_dir, sub)
        if os.path.isdir(candidate):
            print(f"Using compiled mechanisms directory: {candidate}")
            return candidate

    raise RuntimeError(
        f"Could not find compiled mechanisms dir under {parent_dir} "
        f"(tried: {', '.join(candidates)})"
    )


def _load_all_libraries(repo_root: str):
    """
    For each parent dir, locate arm64 or x86_64, then load libnrnmech.
    """
    for parent in MOD_PARENT_DIRS:
        parent_abs = os.path.join(repo_root, parent)
        mech_dir = _find_mech_dir(parent_abs)

        # Typical library names
        lib_candidates = [
            os.path.join(mech_dir, "libnrnmech.dylib"),  # macOS
            os.path.join(mech_dir, "libnrnmech.so"),     # generic Unix
        ]

        lib_path = None
        for path in lib_candidates:
            if os.path.exists(path):
                lib_path = path
                break

        if lib_path is None:
            raise RuntimeError(
                f"Could not find compiled library in {mech_dir} "
                f"(tried: {', '.join(lib_candidates)})"
            )

        print(f"Loading mechanisms from: {lib_path}")
        h.nrn_load_dll(lib_path)


def test_load_and_insert_all_mechanisms():
    """
    Smoke test:
      1. Load all compiled NEURON mechanism libraries.
      2. Try inserting each known SUFFIX into a test Section.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Repo root:", repo_root)

    # 1) Load compiled mechanism libraries (arm64 or x86_64)
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
