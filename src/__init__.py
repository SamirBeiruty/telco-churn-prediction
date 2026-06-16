"""Telco customer churn prediction package.

A small, production-style package that mirrors the analysis in
`notebooks/01_churn_analysis.ipynb`:

    data.py      -> load + clean the raw CSV
    features.py  -> engineer features and build the preprocessing transformer
    model.py     -> define candidate models and assemble full sklearn Pipelines
    evaluate.py  -> metrics + diagnostic plots
    pipeline.py  -> command-line entry point that runs everything end to end
"""

__all__ = ["config", "data", "features", "model", "evaluate", "pipeline"]
