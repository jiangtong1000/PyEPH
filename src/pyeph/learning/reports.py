"""Numerical validation against declared labels, without material-accuracy claims."""

import numpy as np


def error_metrics(prediction, target):
    """Return elementwise RMSE and maximum absolute error, including complex data.

    Shape equality is required: broadcasting must not conceal missing labels.
    Each complex element contributes its squared modulus to the mean.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError("prediction and target must have identical nonempty shapes")
    if any(a.dtype.kind not in "iufc" or not np.isfinite(a).all()
           for a in (prediction, target)):
        raise ValueError("predictions and targets must be finite numerical arrays")
    dtype = np.result_type(prediction.dtype, target.dtype, np.float64)
    error = np.abs(prediction.astype(dtype) - target.astype(dtype))
    if not np.isfinite(error).all():
        raise ValueError("prediction errors exceed the finite numerical range")
    scale = float(np.max(error))
    rmse = scale * float(np.sqrt(np.mean((error / scale)**2))) if scale else 0.
    return dict(rmse=rmse, max_abs=scale)


def validation_report(predictions, targets, splits, *, geometry_ids, groups, units, scope):
    """Record whole-family held-out errors with sample identities and target units.

    All target names, samples and groups must participate explicitly. This
    report records numerical errors, not acceptance thresholds or uncertainty
    calibration. A provider chooses appropriate physical qualification gates.
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("validation scope must be nonempty")
    if not predictions or set(predictions) != set(targets) or set(units) != set(targets):
        raise ValueError("predictions, targets and units must name the same nonempty targets")
    if any(not isinstance(unit, str) or not unit.strip() for unit in units.values()):
        raise ValueError("each target must declare its units")
    geometry_ids, groups = np.asarray(geometry_ids), np.asarray(groups)
    if (geometry_ids.ndim != 1 or not geometry_ids.size or groups.shape != geometry_ids.shape
            or geometry_ids.dtype.kind not in "US" or groups.dtype.kind not in "US"
            or any(not s.strip() for a in (geometry_ids, groups) for s in a.astype(str))
            or len(set(geometry_ids.astype(str))) != geometry_ids.size):
        raise ValueError("geometry IDs must be unique and groups must label every geometry")
    geometry_ids, groups = geometry_ids.astype(str), groups.astype(str)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must specify train, validation and test")
    samples = len(geometry_ids)
    indices = {name: np.asarray(index) for name, index in splits.items()}
    for index in indices.values():
        if (index.ndim != 1 or not index.size or index.dtype.kind not in "iu"
                or np.any(index < 0) or np.any(index >= samples)):
            raise ValueError("split indices must be nonempty bounded integer arrays")
    if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(samples)):
        raise ValueError("splits must cover each geometry exactly once")
    seen = set()
    for index in indices.values():
        family = set(groups[index].astype(str))
        if seen & family:
            raise ValueError("structural families must not leak across splits")
        seen |= family
    for name in targets:
        if np.asarray(targets[name]).ndim == 0 or len(targets[name]) != samples:
            raise ValueError(f"target {name} must have one row per geometry")
        error_metrics(predictions[name], targets[name])
    return dict(
        schema="pyeph.label_validation.v1", scope=scope, units=dict(units),
        splits={name: dict(samples=len(index), geometry_ids=geometry_ids[index].tolist(),
                           groups=sorted(set(groups[index].astype(str))),
                           errors={key: error_metrics(np.asarray(predictions[key])[index],
                                                      np.asarray(targets[key])[index])
                                   for key in targets})
                for name, index in indices.items()},
    )
