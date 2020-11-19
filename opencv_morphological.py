from enum import auto, Enum
from typing import Tuple

import cv2 as cv
import numpy as np


class Shape(Enum):
    rock = auto()
    paper = auto()
    scissors = auto()

    none = auto()


def detect(img: np.ndarray) -> Tuple[np.ndarray, Shape]:
    # img = cv.resize(img, (720, 480))
    img = cv.resize(img, None, fx=0.5, fy=0.5)
    img: np.ndarray

    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # R, G, B = cv.split(img)
    # R = clahe.apply(R)
    # G = clahe.apply(G)
    # B = clahe.apply(B)
    # img = cv.merge((R,G,B))

    rn = cv.blur(img, (5, 5))
    # cv.imshow("1. blur", rn)

    hsv = cv.cvtColor(rn, cv.COLOR_BGR2HSV)
    lower = np.array([0, 30, 90], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    skin_region_hsv = cv.inRange(hsv, lower, upper)

    blurred = cv.medianBlur(skin_region_hsv, 5)

    kernel = np.ones((8, 8), np.uint8)
    morphed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
    morphed = cv.medianBlur(morphed, 5)

    contours, hierarchy = cv.findContours(morphed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    cv.drawContours(img, [contours], -1, (0, 255, 0), 2)

    hull = cv.convexHull(contours)
    cv.drawContours(img, [hull], -1, (0, 0, 0), 2)
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)

    acutes = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])

        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= np.pi / 2:
            acutes += 1
            cv.circle(img, start, 4, [0, 0, 255], -1)

    fingers = acutes + 1 if acutes > 0 else acutes

    if not fingers:
        _shape = Shape.rock
    elif fingers == 2:
        _shape = Shape.scissors
    elif fingers == 5:
        _shape = Shape.paper
    else:
        _shape = Shape.none

    return img, _shape


camera = cv.VideoCapture(0)

while camera.isOpened():
    _, cap = camera.read()
    # noinspection PyBroadException
    try:
        cap, shape = detect(cap)
        cv.putText(
            cap,
            str(shape).split(".")[1],
            (0, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
            cv.LINE_AA,
        )
    except Exception:
        pass

    cv.imshow("2020 Kanggo Storm - Rock Paper Scissors!", cap)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
else:
    im = cv.imread("images/rock.jpg")
    import time

    start_time = time.time()
    im, sh = detect(im)
    cv.putText(
        im,
        str(sh).split(".")[1],
        (0, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
        cv.LINE_AA,
    )
    end_time = time.time()
    cv.imshow("2020 Kanggo Storm - Rock Paper Scissors!", im)
    cv.waitKey()

    print("Detect {} in {}s".format(sh, end_time - start_time))

    raise IOError("No camera available")

camera.release()
cv.destroyAllWindows()
