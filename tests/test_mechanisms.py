from pprint import pprint

import pytest

import opendp.whitenoise.core as wn
from tests import TEST_CSV_PATH, test_csv_names


def analytic_gaussian_similarity():
    an_gauss = []
    gauss = []
    for i in range(100):
        with wn.Analysis(strict_parameter_checks=False):
            PUMS = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

            age = wn.impute(
                wn.to_float(PUMS['age']),
                data_lower=0.,
                data_upper=100.,
                data_rows=1000)

            an_gauss_component = wn.dp_mean(
                age, mechanism="AnalyticGaussian",
                privacy_usage={"epsilon": 1.0, "delta": 1E-6})
            gauss_component = wn.dp_mean(
                age, mechanism="Gaussian",
                privacy_usage={"epsilon": 1.0, "delta": 1E-6})

            an_gauss.append(an_gauss_component.value)
            gauss.append(gauss_component.value)

    print(sum(an_gauss) / len(an_gauss))
    print(sum(gauss) / len(gauss))

@pytest.mark.parametrize(
    "mechanism,privacy_usage",
    [
        pytest.param("AnalyticGaussian", {"epsilon": 2.0, "delta": 1E-6}, id="AnalyticGaussian"),
        pytest.param("Gaussian", {"epsilon": 1.0, "delta": 1E-6}, id="Gaussian"),
        pytest.param("Laplace", {"epsilon": 2.0}, id="Laplace"),

        pytest.param(
            "Gaussian", {"epsilon": 2.0, "delta": 1E-6},
            id="Gaussian_large_epsilon",
            marks=pytest.mark.xfail(raises=AssertionError)),
    ],
)
def test_mechanism(mechanism, privacy_usage):
    with wn.Analysis() as analysis:
        PUMS = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)
        categorical = wn.to_bool(PUMS['sex'], "1")

        numeric = wn.impute(
            wn.to_float(PUMS['age']),
            data_lower=0.,
            data_upper=100.,
            data_rows=1000)

        all = dp_all(numeric, categorical, privacy_usage, mechanism)

        analysis.release()
        all_values = {stat: all[stat].value for stat in all}
        print()
        pprint(all_values)

        for value in all_values.values():
            assert value is not None


def dp_all(numeric, categorical, usage, mechanism):
    return {
        "covariance": wn.dp_covariance(
            left=numeric, right=numeric,
            mechanism=mechanism,
            privacy_usage=usage),
        "histogram": wn.dp_histogram(
            categorical,
            mechanism=mechanism,
            privacy_usage=usage),
        "maximum": wn.dp_maximum(
            numeric, mechanism=mechanism,
            privacy_usage=usage),
        "mean": wn.dp_mean(
            numeric, mechanism=mechanism,
            privacy_usage=usage),
        "median": wn.dp_median(
            numeric, mechanism=mechanism,
            privacy_usage=usage),
        "minimum": wn.dp_minimum(
            numeric, mechanism=mechanism,
            privacy_usage=usage),
        "quantile": wn.dp_quantile(
            numeric, .75, mechanism=mechanism,
            privacy_usage=usage),
        "raw_moment": wn.dp_raw_moment(
            numeric, 2, mechanism=mechanism,
            privacy_usage=usage),
        "sum": wn.dp_sum(
            numeric, mechanism=mechanism,
            privacy_usage=usage),
        "variance": wn.dp_variance(
            numeric, mechanism=mechanism,
            privacy_usage=usage)
    }