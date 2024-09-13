from ochra.element import Element


class BoundingBoxIndeterminateException(Exception):

    def __init__(self, e: Element):
        self.element = e

    def __str__(self):
        return "Bounding box of element cannot be determined."
