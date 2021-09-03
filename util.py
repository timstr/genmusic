import PIL
import sys


def plt_screenshot(plt_figure):
    pil_img = PIL.Image.frombytes(
        "RGB", plt_figure.canvas.get_width_height(), plt_figure.canvas.tostring_rgb()
    )
    return pil_img


def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1) == 0) and (n != 0)


print_width = 50


def horizontal_rule():
    sys.stdout.write("|" + "-" * (print_width - 2) + "|\n")


def line(contents=None):
    if contents is None:
        contents = ""
    n = len(contents)
    l = " " * ((print_width - 2 - n) // 2)
    m = max(0, print_width - 2 - len(l) - len(contents))
    r = " " * m
    sys.stdout.write("|" + l + contents + r + "|\n")


def flush():
    sys.stdout.flush()

def assert_eq(actual, expected):
    if not (actual == expected):
        raise Exception(f"Expected {expected}, but got {actual} instead")
