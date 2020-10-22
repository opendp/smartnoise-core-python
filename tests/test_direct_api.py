
from opendp.smartnoise.core import core_library


def test_laplace_mechanism():
    print(core_library.laplace_mechanism(100., .5, 1.2, False))


def test_gaussian_mechanism():
    print(core_library.gaussian_mechanism(100., .5, .0001, 1.2, False))


def test_simple_geometric_mechanism():
    print(core_library.simple_geometric_mechanism(100, .5, 1.2, 90, 110, False))


def test_snapping_mechanism():
    print(core_library.snapping_mechanism(100., .5, 1.2, 50., 150., False))
    print(core_library.snapping_mechanism(100., .5, 1.2, 50., 150., False, 0.5))
