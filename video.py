from __future__ import print_function

import cv2
import numpy as np


class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg=None):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv2.imread(bg, 1)
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv2.resize(self.bg, self.frame_size)

        self.noise = float(noise)

    def read(self):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv2.randn(noise, np.zeros(3), np.ones(3) * 255 * self.noise)
            buf = cv2.add(buf, noise, dtype=cv2.CV_8UC3)
        return True, buf


def create_capture(source=0):
    source = str(source).strip()
    chunks = source.split(':')

    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = int(chunks[0])
    return cv2.VideoCapture(source)
