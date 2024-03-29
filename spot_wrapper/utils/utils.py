import subprocess

import cv2
import numpy as np


def say(text):
    try:
        text = text.replace("_", " ")
        text = f'"{text}"'
        cmd = f"/usr/bin/festival -b '(voice_cmu_us_slt_arctic_hts)' '(SayText {text})'"
        subprocess.Popen(cmd, shell=True)
    except:
        pass
    print(f'Saying: "{text}"')


def resize_to_tallest(imgs, widest=False, hstack=False):
    dim_idx = 1 if widest else 0
    tallest = max([i.shape[dim_idx] for i in imgs])
    for idx, i in enumerate(imgs):
        curr_dim = i.shape[dim_idx]
        other_dim = i.shape[1 - dim_idx]
        if curr_dim != tallest:
            new_dim = int(other_dim * (tallest / curr_dim))
            new_dims = (new_dim, tallest) if widest else (tallest, new_dim)
            imgs[idx] = cv2.resize(i, new_dims)
    if hstack:
        return np.hstack(imgs)
    return imgs


def inflate_erode(mask, size=50):
    mask_copy = mask.copy()
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy > 0] = 255
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy < 255] = 0

    return mask_copy


def erode_inflate(mask, size=20):
    mask_copy = mask.copy()
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy < 255] = 0
    mask_copy = cv2.blur(mask_copy, (size, size))
    mask_copy[mask_copy > 0] = 255

    return mask_copy


def contour_mask(mask):
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    max_area = 0
    max_index = 0
    for idx, c in enumerate(cnt):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_index = idx
    cv2.drawContours(new_mask, cnt, max_index, 255, cv2.FILLED)

    return new_mask


def color_bbox(img, just_get_bbox=False):
    """Makes a bbox around a white object"""
    # Filter out non-white
    sensitivity = 80
    upper_white = np.array([255, 255, 255])
    lower_white = upper_white - sensitivity
    color_mask = cv2.inRange(img, lower_white, upper_white)

    # Filter out little bits of white
    color_mask = inflate_erode(color_mask)
    color_mask = erode_inflate(color_mask)

    # Only use largest contour
    color_mask = contour_mask(color_mask)

    # Calculate bbox
    x, y, w, h = cv2.boundingRect(color_mask)

    if just_get_bbox:
        return x, y, w, h

    height, width = color_mask.shape
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height

    # Create bbox mask
    bbox_mask = np.zeros([height, width, 1], dtype=np.float32)
    bbox_mask[y : y + h, x : x + w] = 1.0

    # Determine if bbox intersects with central crosshair
    crosshair_in_bbox = x < width // 2 < x + w and y < height // 2 < y + h

    return bbox_mask, cx, cy, crosshair_in_bbox
