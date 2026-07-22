"""SLIC-style clustering using only RGB and XYZ distances.

The assignment distance is exactly

    D^2 = ||dRGB||^2 + (compactness / S)^2 ||dXYZ||^2

No UV/image-coordinate term is used.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

Array = np.ndarray


def mesh_surface_spatial_step(
    vertices: Array,
    triangle_vertex_indices: Array,
    n_segments: int,
) -> float:
    """Return S = sqrt(mesh_surface_area / n_segments).

    This is the 3D-surface analogue of the usual SLIC grid interval
    S = sqrt(number_of_pixels / number_of_superpixels).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangle_vertex_indices, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangle_vertex_indices must have shape (M, 3).")
    if n_segments < 1:
        raise ValueError("n_segments must be positive.")

    triangle_vertices = vertices[triangles]
    edge_1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    surface_area = 0.5 * np.linalg.norm(
        np.cross(edge_1, edge_2), axis=1
    ).sum()

    if not np.isfinite(surface_area) or surface_area <= 0.0:
        raise ValueError("The mesh surface area must be positive.")

    return float(np.sqrt(surface_area / n_segments))


def _random_sample_indices(
    number_of_points: int,
    maximum_sample_size: int,
    rng: np.random.Generator,
) -> Array:
    if number_of_points <= maximum_sample_size:
        return np.arange(number_of_points, dtype=np.int64)

    return rng.choice(
        number_of_points,
        size=maximum_sample_size,
        replace=False,
    ).astype(np.int64)


def _estimate_spatial_step(
    xyz_points: Array,
    n_segments: int,
    rng: np.random.Generator,
    sample_size: int = 100_000,
) -> float:
    """Estimate S from point density when mesh area is unavailable."""
    number_of_points = len(xyz_points)
    sample_indices = _random_sample_indices(
        number_of_points,
        min(sample_size, number_of_points),
        rng,
    )
    sample_xyz = xyz_points[sample_indices].astype(np.float64, copy=False)
    number_of_sampled_points = len(sample_xyz)

    if number_of_sampled_points < 2:
        return 1.0

    tree = cKDTree(sample_xyz) # permite procurar rapidamente os vizinhos espaciais de cada ponto
    number_of_neighbors = min(16, number_of_sampled_points)

    try:
        neighbor_distances, _ = tree.query(
            sample_xyz,
            k=number_of_neighbors,
            workers=-1,
        )
    except TypeError:  # Compatibility with older SciPy versions.
        neighbor_distances, _ = tree.query(
            sample_xyz,
            k=number_of_neighbors,
        )

    if neighbor_distances.ndim == 1:
        neighbor_distances = neighbor_distances[:, None]

    bounding_box_diagonal = float(
        np.linalg.norm(np.ptp(sample_xyz, axis=0))
    )
    distance_epsilon = max(bounding_box_diagonal, 1.0) * 1.0e-12

    positive_distances = np.where(
        neighbor_distances > distance_epsilon,
        neighbor_distances,
        np.inf,
    )
    nearest_positive_distance = positive_distances.min(axis=1)
    nearest_positive_distance = nearest_positive_distance[
        np.isfinite(nearest_positive_distance)
    ]

    if nearest_positive_distance.size == 0:
        if bounding_box_diagonal <= 0.0:
            raise ValueError("XYZ points have zero spatial extent.")
        return bounding_box_diagonal / np.sqrt(n_segments)

    median_sample_spacing = float(np.median(nearest_positive_distance))

    # For points sampled from a 2D surface, nearest-neighbor spacing scales as
    # 1/sqrt(number_of_points). Therefore this estimates the expected linear
    # spacing of one of n_segments surface regions.
    spatial_step = median_sample_spacing * np.sqrt(
        number_of_sampled_points / float(n_segments)
    )

    if not np.isfinite(spatial_step) or spatial_step <= 0.0:
        spatial_step = bounding_box_diagonal / np.sqrt(n_segments)

    return max(spatial_step, np.finfo(np.float32).eps)


def _initialize_centers_from_xyz_voxels(
    rgb_points: Array,
    xyz_points: Array,
    n_segments: int,
    rng: np.random.Generator,
    maximum_sample_size: int = 250_000,
) -> tuple[Array, Array]:
    """Create spatially distributed initial centers without using UV."""
    number_of_points = len(xyz_points)
    requested_sample_size = min(
        number_of_points,
        maximum_sample_size,
        max(10 * n_segments, 50_000),
    )
    sample_indices = _random_sample_indices(
        number_of_points,
        requested_sample_size,
        rng,
    )
    sample_xyz = xyz_points[sample_indices]

    xyz_minimum = sample_xyz.min(axis=0)
    xyz_extent = sample_xyz.max(axis=0) - xyz_minimum
    xyz_extent[xyz_extent <= 0.0] = 1.0

    # A surface is intrinsically two-dimensional, so sqrt(K) cells per axis
    # is a useful first grid resolution. Increase it until enough occupied
    # cells exist to initialize all centers.
    grid_resolution = max(2, int(np.ceil(np.sqrt(n_segments))))
    candidate_indices: Optional[Array] = None

    for _ in range(8):
        voxel_coordinates = np.floor(
            (sample_xyz - xyz_minimum)
            / xyz_extent
            * grid_resolution
        ).astype(np.int64)
        np.clip(
            voxel_coordinates,
            0,
            grid_resolution - 1,
            out=voxel_coordinates,
        )

        voxel_keys = (
            voxel_coordinates[:, 0]
            + grid_resolution
            * (
                voxel_coordinates[:, 1]
                + grid_resolution * voxel_coordinates[:, 2]
            )
        )

        _, first_in_voxel = np.unique(voxel_keys, return_index=True)
        candidate_indices = sample_indices[first_in_voxel]

        if len(candidate_indices) >= n_segments:
            break

        grid_resolution = int(np.ceil(grid_resolution * 1.5))

    assert candidate_indices is not None

    if len(candidate_indices) >= n_segments:
        selected_indices = rng.choice(
            candidate_indices,
            size=n_segments,
            replace=False,
        )
    else:
        selected_indices_list = candidate_indices.tolist()
        number_missing = n_segments - len(selected_indices_list)

        remaining_pool = np.setdiff1d(
            np.arange(number_of_points, dtype=np.int64),
            candidate_indices,
            assume_unique=False,
        )
        selected_indices_list.extend(
            rng.choice(
                remaining_pool,
                size=number_missing,
                replace=False,
            ).tolist()
        )
        selected_indices = np.asarray(
            selected_indices_list,
            dtype=np.int64,
        )

    return (
        rgb_points[selected_indices].astype(np.float64),
        xyz_points[selected_indices].astype(np.float64),
    )


def _query_tree(tree: cKDTree, features: Array) -> Array:
    try:
        _, labels = tree.query(features, k=1, workers=-1)
    except TypeError:  # Compatibility with older SciPy versions.
        _, labels = tree.query(features, k=1)
    return labels.astype(np.int32, copy=False)


def run_rgb_xyz_slic(
    rgb: Array,
    xyz: Array,
    valid_mask: Optional[Array],
    n_segments: int,
    compactness: float,
    max_num_iter: int,
    spatial_step: Optional[float] = None,
    convergence_tol: float = 1.0e-4,
    chunk_size: int = 250_000,
    random_state: int = 0,
    verbose: bool = True,
) -> Array:
    """Cluster RGB+XYZ points using a SLIC-style Lloyd iteration.

    The only assignment distance is

        D^2 = ||RGB_i - RGB_c||^2
              + (compactness / S)^2 ||XYZ_i - XYZ_c||^2.

    No UV, image row, or image column is included in either the feature vector
    or the assignment distance.

    Parameters
    ----------
    rgb, xyz:
        Either arrays with shape (H, W, 3), or point arrays with shape (N, 3).
        RGB should normally be in [0, 1]. Integer RGB is converted to [0, 1].
    valid_mask:
        For image-shaped inputs, a boolean array with shape (H, W). For point
        inputs, a boolean array with shape (N,), or None to use every point.
    n_segments:
        Requested number of clusters.
    compactness:
        Geometry strength in the requested distance formula.
    max_num_iter:
        Maximum number of assignment/update iterations.
    spatial_step:
        The S term in the distance. For a surface mesh, the recommended value
        is sqrt(mesh_surface_area / n_segments), obtainable with
        mesh_surface_spatial_step(). If None, S is estimated from XYZ density.

    Returns
    -------
    labels:
        Shape (H, W) for image inputs or (N,) for point inputs. Invalid entries
        are -1; valid labels are contiguous integers starting at zero.
    """
    rgb_input = np.asarray(rgb)
    xyz_input = np.asarray(xyz)

    if rgb_input.shape != xyz_input.shape:
        raise ValueError("rgb and xyz must have the same shape.")
    if rgb_input.ndim not in (2, 3) or rgb_input.shape[-1] != 3:
        raise ValueError(
            "rgb and xyz must have shape (N, 3) or (H, W, 3)."
        )
    if n_segments < 1:
        raise ValueError("n_segments must be positive.")
    if compactness < 0.0:
        raise ValueError("compactness must be non-negative.")
    if max_num_iter < 1:
        raise ValueError("max_num_iter must be positive.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    leading_shape = rgb_input.shape[:-1]
    if valid_mask is None:
        mask = np.ones(leading_shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != leading_shape:
            raise ValueError(
                f"valid_mask must have shape {leading_shape}, received "
                f"{mask.shape}."
            )

    mask &= np.all(np.isfinite(rgb_input), axis=-1)
    mask &= np.all(np.isfinite(xyz_input), axis=-1)

    number_of_points = int(np.count_nonzero(mask))
    if number_of_points == 0:
        return np.full(leading_shape, -1, dtype=np.int32)

    number_of_clusters = min(int(n_segments), number_of_points)

    rgb_float = rgb_input.astype(np.float32, copy=False)
    if np.issubdtype(rgb_input.dtype, np.integer):
        rgb_float = rgb_float / float(np.iinfo(rgb_input.dtype).max)

    valid_rgb = rgb_float[mask]
    if valid_rgb.min() < 0.0 or valid_rgb.max() > 1.0:
        raise ValueError(
            "Floating-point RGB must be in [0, 1]. Normalize it before "
            "calling run_rgb_xyz_slic()."
        )

    rgb_points = np.ascontiguousarray(valid_rgb, dtype=np.float32)
    xyz_points = np.ascontiguousarray(xyz_input[mask], dtype=np.float32)

    rng = np.random.default_rng(random_state)

    if spatial_step is None:
        spatial_step_value = _estimate_spatial_step(
            xyz_points,
            number_of_clusters,
            rng,
        )
    else:
        spatial_step_value = float(spatial_step)

    if not np.isfinite(spatial_step_value) or spatial_step_value <= 0.0:
        raise ValueError("spatial_step must be finite and positive.")

    spatial_scale = float(compactness) / spatial_step_value

    if verbose:
        print(
            "RGB+XYZ SLIC: "
            f"points={number_of_points:,}, "
            f"clusters={number_of_clusters:,}, "
            f"S={spatial_step_value:.6g}, "
            f"compactness/S={spatial_scale:.6g}"
        )

    center_rgb, center_xyz = _initialize_centers_from_xyz_voxels(
        rgb_points,
        xyz_points,
        number_of_clusters,
        rng,
    )

    point_labels = np.empty(number_of_points, dtype=np.int32)

    for iteration in range(max_num_iter):
        old_center_features = np.concatenate(
            [center_rgb, spatial_scale * center_xyz],
            axis=1,
        )
        center_tree = cKDTree(old_center_features)

        counts = np.zeros(number_of_clusters, dtype=np.int64)
        rgb_sums = np.zeros((number_of_clusters, 3), dtype=np.float64)
        xyz_sums = np.zeros((number_of_clusters, 3), dtype=np.float64)

        for start in range(0, number_of_points, chunk_size):
            stop = min(start + chunk_size, number_of_points)

            chunk_features = np.empty((stop - start, 6), dtype=np.float64)
            chunk_features[:, :3] = rgb_points[start:stop]
            chunk_features[:, 3:] = (
                spatial_scale * xyz_points[start:stop]
            )

            chunk_labels = _query_tree(center_tree, chunk_features)
            point_labels[start:stop] = chunk_labels

            counts += np.bincount(
                chunk_labels,
                minlength=number_of_clusters,
            )

            for channel in range(3):
                rgb_sums[:, channel] += np.bincount(
                    chunk_labels,
                    weights=rgb_points[start:stop, channel],
                    minlength=number_of_clusters,
                )
                xyz_sums[:, channel] += np.bincount(
                    chunk_labels,
                    weights=xyz_points[start:stop, channel],
                    minlength=number_of_clusters,
                )

        nonempty = counts > 0
        center_rgb[nonempty] = (
            rgb_sums[nonempty] / counts[nonempty, None]
        )
        center_xyz[nonempty] = (
            xyz_sums[nonempty] / counts[nonempty, None]
        )

        empty_clusters = np.flatnonzero(~nonempty)
        if empty_clusters.size:
            replacement_points = rng.choice(
                number_of_points,
                size=empty_clusters.size,
                replace=False,
            )
            center_rgb[empty_clusters] = rgb_points[replacement_points]
            center_xyz[empty_clusters] = xyz_points[replacement_points]

        new_center_features = np.concatenate(
            [center_rgb, spatial_scale * center_xyz],
            axis=1,
        )
        mean_center_shift = float(
            np.mean(
                np.linalg.norm(
                    new_center_features - old_center_features,
                    axis=1,
                )
            )
        )

        if verbose:
            print(
                f"  iteration {iteration + 1}: "
                f"mean center shift={mean_center_shift:.6g}, "
                f"empty clusters={empty_clusters.size}"
            )

        if mean_center_shift <= convergence_tol:
            break

    # Final assignment using the final center locations.
    final_center_features = np.concatenate(
        [center_rgb, spatial_scale * center_xyz],
        axis=1,
    )
    final_tree = cKDTree(final_center_features)

    for start in range(0, number_of_points, chunk_size):
        stop = min(start + chunk_size, number_of_points)
        chunk_features = np.empty((stop - start, 6), dtype=np.float64)
        chunk_features[:, :3] = rgb_points[start:stop]
        chunk_features[:, 3:] = spatial_scale * xyz_points[start:stop]
        point_labels[start:stop] = _query_tree(final_tree, chunk_features)

    # Remove labels belonging to centers that ended empty.
    _, contiguous_labels = np.unique(point_labels, return_inverse=True)

    output = np.full(leading_shape, -1, dtype=np.int32)
    output[mask] = contiguous_labels.astype(np.int32)
    return output
