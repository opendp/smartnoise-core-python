import opendp.whitenoise.core as wn


def test_theil_sen():
    """
    :return:
    """
    with wn.Analysis() as analysis:
        res = wn.theil_sen([1., 2., 3.], [2.1, 4.4, 6.6], "theil-sen", 0)
        print()
        print("Validate: ", analysis.validate())
        analysis.release()
        print("Releasable: ", res.releasable)
        print("Results: ", res.value)


def test_linear_regression():
    """
    """
    pass