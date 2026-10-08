"""Phenotype prediction is not provided by the natural-language interface."""

def predictor(*args, **kwargs):
    raise NotImplementedError(
        "NLUI phenotype prediction is unavailable in this release. "
        "Use the separate Stage 3 research workflow with its documented cohort inputs."
    )
