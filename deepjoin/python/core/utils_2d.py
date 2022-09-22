import cv2
import numpy as np


def annotation_image(dims, text, hspace=20, fontsize=1.0, color=(0, 0, 0)):
    """
    Returns a white image with a black annotation in the top left corner.
    """
    return annotate_image(
        (np.ones((dims)) * 255).astype(np.uint8),
        text,
        hspace,
        fontsize,
        color=color,
    )


def annotate_image(img, text, hspace=20, fontsize=1.0, color=(0, 0, 0)):
    """
    Adds a black annotation in the top left corner.
    """
    return cv2.putText(
        img, text, (20, hspace), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, 1
    )


def bar_image(dims):
    return (np.zeros((dims)) * 255).astype(np.uint8)


def space_image(dims):
    return (np.ones((dims)) * 255).astype(np.uint8)


def vcutstack(img, size=None, bg_color=1):
    """Cuts an image in half vertically and stacks it back together horizontally"""
    h = img.shape[0]
    if size is None:
        return np.hstack((img[: int(h / 2), :, :], img[int(h / 2) :, :, :]))

    # Compute how much is left over
    single_h = size[0]
    leftover = int(h / 2) % single_h

    # If there's any left over, add a blank image
    if leftover != 0:
        prev_shape = list(img.shape)
        prev_shape[0] = size[0]
        img = np.vstack((img, np.ones((prev_shape)).astype(img.dtype) * bg_color * 255))

    return vcutstack(img)


def hcutstack(img):
    """Cuts an image in half horizontally and stacks it back together vertically"""
    w = img.shape[1]
    return np.hstack((img[:, : int(w / 2), :], img[:, int(w / 2) :, :]))
