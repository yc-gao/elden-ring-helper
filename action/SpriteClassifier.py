import math

import cv2 as cv


class SpriteClassifier:
    def __init__(self, threshold, sprite, width=1, height=1):
        self.threshold = threshold

        if isinstance(sprite, str):
            temp = cv.imread(sprite)
            sprite = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        self.sprite = sprite

        self.width = width
        self.height = height
        h, w = self.sprite.shape[:2]
        self.dx = w / width
        self.dy = h / height

    def classify(self, img_gray):
        h, w = img_gray.shape[:2]
        res = cv.matchTemplate(self.sprite, img_gray, cv.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        if max_val > self.threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center_point = (top_left[0] + w / 2, top_left[1] + h / 2)
            x = math.floor(center_point[0] / self.dx)
            y = math.floor(center_point[1] / self.dy)
            return (max_val, (x, y), top_left, bottom_right)
