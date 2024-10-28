class Global:
    """
    Global settings for drawing.
    """
    approx_eps: float = 2 ** -12
    boundary_eps: float = 2 ** -16
    step_size: float = 4.0
    num_first_order_steps: int = 256
    num_second_order_steps: int = 64

    @classmethod
    def set_approx_eps(cls, eps: float):
        cls.approx_eps = eps

    @classmethod
    def set_boundary_eps(cls, zeta: float):
        cls.boundary_eps = zeta

    @classmethod
    def set_step_size(cls, step: float):
        cls.step_size = step

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
