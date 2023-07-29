import math

import cv2 as cv


class SpriteDetector:
    def __init__(self, temp):
        if isinstance(temp, str):
            temp = cv.imread(temp)
            temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
        self.temp = temp
        self.h, self.w = temp.shape[:2]

    def detect(self, sprite, size=(1, 1), threshold=0.9):
        if isinstance(sprite, str):
            sprite = cv.imread(sprite)
            sprite = cv.cvtColor(sprite, cv.COLOR_BGR2GRAY)
        h, w = sprite.shape[:2]
        dx, dy = w / size[0], h / size[1]
        
        res = cv.matchTemplate(sprite, self.temp, cv.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        if max_val > self.threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
            center_point = (top_left[0] + self.w / 2, top_left[1] + self.h / 2)
            x = math.floor(center_point[0] / dx)
            y = math.floor(center_point[1] / dy)
            return (max_val, (x, y), top_left, bottom_right, center_point)
