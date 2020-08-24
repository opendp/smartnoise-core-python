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


def test_theil_sen_with_original_median():
    """
    Use median and exponential mechanism for DP linear regression
    """
    with wn.Analysis() as analysis:
        res = wn.theil_sen([1., 2., 3.], [2.1, 4.4, 6.6], "theil-sen", 0)
        print()
        print("Validate: ", analysis.validate())
        analysis.release()
        print("Releasable: ", res.releasable)
        print("Results: ", res.value)
        slope_candidates = wn.median(
            res.value['slopes'],
            data_rows=10)
        slope = wn.exponential_mechanism(slope_candidates, candidates=res.value['slopes'], privacy_usage={"epsilon": 1.})

        intercept_candidates = wn.median(
            res.value['intercepts'],
            data_rows=10)
        intercept = wn.exponential_mechanism(intercept_candidates, candidates=res.value['intercepts'], privacy_usage={"epsilon": 1.})

        print("(slope, intercept) = ({}, {})".format(slope.value, intercept.value))


def test_theil_sen_with_gumbel_median():
    """
    Use median and exponential mechanism for DP linear regression
    """
    with wn.Analysis() as analysis:
        res = wn.theil_sen([1., 2., 3.], [2.1, 4.4, 6.6], "theil-sen", 0)
        print()
        print("Validate: ", analysis.validate())
        analysis.release()
        print("Releasable: ", res.releasable)
        print("Results: ", res.value)
        slope = wn.dp_gumbel_median(
            res.value['slopes'], 0., 3., True, privacy_usage={'epsilon': 1})

        intercept = wn.dp_gumbel_median(
            res.value['intercepts'], 0., 1., True, privacy_usage={'epsilon': 1})

        print("(slope, intercept) = ({}, {})".format(slope.value, intercept.value))