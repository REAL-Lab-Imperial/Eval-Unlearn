import logging

_log = logging.getLogger(__name__)

try:
    from .asr_p4d.metric import ASRP4D
except Exception as e:
    _log.warning("Could not register ASRP4D: %s", e)

try:
    from .asr_i2p.metric import ASRMetric
except Exception as e:
    _log.warning("Could not register ASRMetric: %s", e)

try:
    from .fid.metric import FIDMetric
except Exception as e:
    _log.warning("Could not register FIDMetric: %s", e)

try:
    from .err.metric import ERRMetric
except Exception as e:
    _log.warning("Could not register ERRMetric: %s", e)

try:
    from .tifa.metric import TIFAMetric
except Exception as e:
    _log.warning("Could not register TIFAMetric: %s", e)

try:
    from .clip_score.metric import CLIPScoreMetric
except Exception as e:
    _log.warning("Could not register CLIPScoreMetric: %s", e)

try:
    from .ua_ira.metric import UAIRAMetric
except Exception as e:
    _log.warning("Could not register UAIRAMetric: %s", e)

try:
    from .asr_ring_a_bell.metric import ASRRingABellMetric
except Exception as e:
    _log.warning("Could not register ASRRingABellMetric: %s", e)

try:
    from .asr_mma_diffusion.metric import MMADiffusionMetric
except Exception as e:
    _log.warning("Could not register MMADiffusionMetric: %s", e)

__all__ = [
    name for name in [
        "ASRP4D",
        "ASRMetric",
        "FIDMetric",
        "ERRMetric",
        "TIFAMetric",
        "CLIPScoreMetric",
        "UAIRAMetric",
        "ASRRingABellMetric",
        "MMADiffusionMetric",
    ]
    if name in globals()
]
