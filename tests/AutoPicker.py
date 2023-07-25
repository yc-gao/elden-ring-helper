import argparse

import cv2 as cv

from action import AutoPicker


def imread(fname):
    img = cv.imread(fname)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sprite')
    parser.add_argument('-s', '--size')
    parser.add_argument('--bbox')
    args = parser.parse_args()

    sprite = imread(args.sprite)
    bbox = tuple(map(int, args.bbox.split(',')))
    width, height = map(int, args.size.split('x'))

    def post_handler(res, *args):
        print(res)
    picker = AutoPicker(0.1, post_handler, bbox, 0.9,
                        sprite, width, height)
    picker.start()
    picker.join()


if __name__ == '__main__':
    main()
