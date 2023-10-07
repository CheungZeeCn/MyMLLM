
import cv2
import json
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np

default_color_li = [
    "#FF0000",  # red
    "#FFFF00",  # yellow
    "#0000FF",  # blue
    "#D2691E",  # chocolate
    "#00FF00",  # lime
    "#FF00FF",  # fuchsia
    "#FF1493",  # deeppink
    "#FFA500",  # orange
    "#00FFFF",  # cyan
    "#DC143C",  # crimson
    "#000000"   # black
]

ttc_font = None
def setup(ttc_path, size=12):
    global ttc_font
    ttc_font = ImageFont.truetype(ttc_path, size)


def pil_draw_line(img, a, b, color='black'):
    shape = [a, b]
    img1 = ImageDraw.Draw(img)
    img1.line(shape, fill=color, width=0)
    return img

def pil_draw_box(img, coord, color='black'):
    img1 = ImageDraw.Draw(img)
    img1.rectangle(xy=tuple(coord),
                   outline=color,
                   width=1)
    return img