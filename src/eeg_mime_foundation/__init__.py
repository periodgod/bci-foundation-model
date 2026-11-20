"""EEG/time-series preparation framework; engineering checks are not research results."""

from .model import ModelConfig, SignalModel, count_parameters

__all__ = ["ModelConfig", "SignalModel", "count_parameters"]
