"""Augmentation source loaders. Each module registers itself on import."""

from afro_health_qa.data.sources import (  # noqa: F401
    ghananlp_khaya,
    medmcqa_translated,
    sunbird_salt,
    synthetic_qa,
    who_factsheets,
)
