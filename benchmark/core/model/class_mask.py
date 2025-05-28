import numpy as np

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.core.model.image import CV2Image, CV2Mask


class ClassMask(PydanticFrozen):
    image: CV2Image
    intensity: np.uint8

    def get_value(self) -> CV2Mask:
        return self.image == self.intensity
