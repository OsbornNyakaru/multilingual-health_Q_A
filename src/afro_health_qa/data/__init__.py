"""Data loading, language ID, splitting, and augmentation."""

from afro_health_qa.data.language_id import detect_lang
from afro_health_qa.data.load import load_competition_data
from afro_health_qa.data.split import stratified_split

__all__ = ["detect_lang", "load_competition_data", "stratified_split"]
