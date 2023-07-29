import argparse

import cv2 as cv
from matplotlib import pyplot as plt

from action import SpriteClassifier


def imread(fname):
    img = cv.imread(fname)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sprite')
    parser.add_argument('-s', '--size')
    parser.add_argument('fname')
    args = parser.parse_args()

    sprite = imread(args.sprite)
    width, height = map(int, args.size.split('x'))
    cls = SpriteClassifier(sprite, (width, height))

    img = imread(args.fname)
    res = cls.classify(img)

    if res is not None:
        print(res)
        cv.rectangle(sprite, res[2], res[3], 0, 2)
        plt.imshow(sprite, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
