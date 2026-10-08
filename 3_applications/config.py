"""User-editable analysis groups.

Only the distributed example session is configured here. Add verified session IDs
for your own dataset; this example does not define the paper's experimental cohorts.
"""
from pathlib import Path


group_configs = {
    "example": ["240419_C11M4"],
    "240419_C11M4": ["240419_C11M4"],
}


def resolve_groups(names):
    """Resolve explicit group names, deduplicate, and preserve script sort order."""
    if not names:
        raise ValueError("Select at least one group configuration from config.py.")
    unknown = [name for name in names if name not in group_configs]
    if unknown:
        raise ValueError(
            f"Unknown group configuration(s): {', '.join(unknown)}. "
            "Add verified session lists to 3_applications/config.py first."
        )
    groups = sorted({session for name in names for session in group_configs[name]})
    if not groups:
        raise ValueError("The selected group configuration contains no sessions.")
    return groups


def reduction_paths(folder, names, downscale, model_name, n_neigh, min_dist=0.1):
    """Shared filenames for the dimension-reduction writer and clustering reader."""
    resolve_groups(names)
    if downscale != 1:
        raise ValueError("Only downscale=1 is supported for per-frame HDF5 output.")
    if n_neigh < 2:
        raise ValueError("n_neigh must be at least 2.")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must lie between 0 and 1.")
    suffix = f"{'_'.join(names)}_ds{downscale}_{model_name}_{n_neigh}"
    if min_dist != 0.1:
        suffix += f"_{min_dist}"
    folder = Path(folder)
    return (
        folder / f"embedding_umap_des_{suffix}.npy",
        folder / f"linkage_matrix_des_{suffix}.npy",
    )
