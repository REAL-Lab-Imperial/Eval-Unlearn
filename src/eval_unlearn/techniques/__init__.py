from .sld.wrapper import SLDTechnique
from .uce.wrapper import UCETechnique
from .concept_steerers.wrapper import ConceptSteerersTechnique
from .free_run.wrapper import FreeRunTechnique
from .mace.wrapper import MACETechnique

try:
    from .saeuron.wrapper import SAeUronTechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register saeuron: %s", e)

try:
    from .esd.wrapper import ESDTechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register esd: %s", e)

try:
    from .SAFREE.wrapper import SAFREETechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register safree: %s", e)

try:
    from .advunlearn.wrapper import AdvUnlearnTechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register advunlearn: %s", e)

try:
    from .cogfd.wrapper import CoGFDTechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register cogfd: %s", e)

try:
    from .ssd.wrapper import SSDTechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register ssd: %s", e)

try:
    from .ca.wrapper import CATechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register ca: %s", e)

try:
    from .trasce.wrapper import TraSCETechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register trasce: %s", e)

__all__ = [
    name
    for name in [
        "SLDTechnique",
        "UCETechnique",
        "ConceptSteerersTechnique",
        "FreeRunTechnique",
        "MACETechnique",
        "SAeUronTechnique",
        "ESDTechnique",
        "SAFREETechnique",
        "AdvUnlearnTechnique",
        "CoGFDTechnique",
        "SSDTechnique",
        "TraSCETechnique",
        "CATechnique",
    ]
    if name in globals()
]
