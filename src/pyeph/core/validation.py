"""Optional host model checks at supplied parameters and actual coordinates."""

import numpy as np


def validate_model_at(model, params, q, *, batch=False):
    """Run a model's concrete preflight, or its existing geometry-only check.

    ``validate_at(params, q, *, batch=False)`` is optional. A model supplying it
    owns geometry checks too, and may evaluate an entire native batch at once.
    This host operation does not enter a compiled dynamics step or certify the
    provider at future coordinates. It introduces no required model protocol.
    Parameter-only validation remains part of Problem construction/update.
    """
    validate = getattr(model, "validate_at", None)
    if callable(validate):
        validate(params, q, batch=batch)
        return
    geometry = getattr(model, "validate_geometry", None)
    if callable(geometry):
        values = np.asarray(q)
        if batch:
            for value in values:
                geometry(value)
        else:
            geometry(values)
