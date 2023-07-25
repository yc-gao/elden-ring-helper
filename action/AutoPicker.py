from PIL import ImageGrab
import numpy as np
import cv2 as cv

from .CommonAction import CommonAction
from .SpriteClassifier import SpriteClassifier


class AutoPicker(CommonAction):
    def __init__(self, period, post_handler, bbox, *args, **kwargs):
        super().__init__(period)
        self.post_handler = post_handler
        self.bbox = bbox
        self.sprite = SpriteClassifier(*args, **kwargs)

    def run(self):
        img = np.array(ImageGrab.grab(self.bbox))
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        res = self.sprite.classify(img_gray)
        if res is not None:
            self.post_handler(res, img)
