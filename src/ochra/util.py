class Global:
    """
    Global settings for drawing.
    """
    approx_eps: float = 2 ** -16
    """The epsilon for approximating floating point numbers as equal."""

    boundary_eps: float = 2 ** -16
    """The epsilon for avoiding boundary issues."""

    first_order_step_size: float = 2.0
    """The step size for tracing polylines."""

    second_order_step_size: float = 8.0
    """The step size for tracing quadratic Bézier paths."""

    num_first_order_steps: int = 512
    """The number of steps for tracing polylines."""

    num_second_order_steps: int = 64
    """The number of steps for tracing quadratic Bézier paths."""

    @classmethod
    def set_approx_eps(cls, eps: float):
        cls.approx_eps = eps

    @classmethod
    def set_boundary_eps(cls, zeta: float):
        cls.boundary_eps = zeta

    @classmethod
    def set_first_order_step_size(cls, step: float):
        cls.first_order_step_size = step
        
    @classmethod
    def set_second_order_step_size(cls, step: float):
        cls.second_order_step_size = step

    @classmethod
    def set_num_first_order_steps(cls, n: int):
        cls.num_first_order_steps = n

    @classmethod
    def set_num_second_order_steps(cls, n: int):
        cls.num_second_order_steps = n


class classproperty:
    """
    Decorator that converts a method with a single cls argument into a property
    that can be accessed directly from the class.
    From https://docs.djangoproject.com/en/5.0/_modules/django/utils/functional/#classproperty.
    """

    def __init__(self, method=None):
        self.fget = method

    def __get__(self, instance, cls=None):
        return self.fget(cls)

    def getter(self, method):
        self.fget = method
        return self


def expected(func):
    """
    Decorator that marks a function as expected to be implemented in the future.
    """
    func.__expected__ = True
    return func
