"""
Initialize the src package
"""

from .model import SpeechToTextTransformer
from .dataset import TextTokenizer, AudioPreprocessor, SpeechToTextDataset
from .utils import calculate_wer, calculate_cer, LabelSmoothedCrossEntropy

__version__ = '1.0.0'
__all__ = [
    'SpeechToTextTransformer',
    'TextTokenizer',
    'AudioPreprocessor',
    'SpeechToTextDataset',
    'calculate_wer',
    'calculate_cer',
    'LabelSmoothedCrossEntropy'
]
