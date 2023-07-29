import math

import cv2 as cv


class SpriteClassifier:
    def __init__(self, sprite, size=(1, 1)):
        if isinstance(sprite, str):
            temp = cv.imread(sprite)
            sprite = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        self.sprite = sprite

        h, w = self.sprite.shape[:2]
        self.dx = w / size[0]
        self.dy = h / size[1]

    def classify(self, img_gray, threshold=0.9):
        h, w = img_gray.shape[:2]
        res = cv.matchTemplate(self.sprite, img_gray, cv.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        if max_val > threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center_point = (top_left[0] + w / 2, top_left[1] + h / 2)
            x = math.floor(center_point[0] / self.dx)
            y = math.floor(center_point[1] / self.dy)
            return (max_val, (x, y), top_left, bottom_right, center_point)
