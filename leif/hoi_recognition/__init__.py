"""
Module 4: Human-Object Interaction (HOI) Recognition.
Recognizes interactions: holding, touching, opening, closing, placing, moving, using.
Integrates Torchvision R3D-18 Video Action Recognition model.
"""

from leif.hoi_recognition.spatial_relations import SpatialRelationAnalyzer
from leif.hoi_recognition.action_recognizer import VideoActionRecognizer
from leif.hoi_recognition.hoi_classifier import HOIClassifier

__all__ = [
    "SpatialRelationAnalyzer",
    "VideoActionRecognizer",
    "HOIClassifier",
]
