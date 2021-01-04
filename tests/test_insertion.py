import pytest

import opendp.smartnoise.core as sn
import numpy as np


def test_insertion_simple():
    """
    Conduct a differentially private analysis with values inserted from other systems
    """
    with sn.Analysis(protect_floating_point=False) as analysis:

        # construct a fake dataset that describes your actual data (will never be run)
        data = sn.Dataset(path="", column_names=["A", "B", "C", "D"])

        # pull a column out
        col_a = sn.to_float(data['A'])

        # describe the preprocessing you actually perform on the data
        col_a_clamped = sn.impute(sn.clamp(col_a, lower=0., upper=10.))
        col_a_resized = sn.resize(col_a_clamped, number_rows=1000000)

        # run a fake aggregation
        actual_mean = sn.mean(col_a_resized)

        # insert aggregated data from an external system
        actual_mean.set(10.)

        # describe the differentially private operation
        gaussian_mean = sn.gaussian_mechanism(actual_mean, privacy_usage={"epsilon": .4, "delta": 1e-6})

        # check if the analysis is permissible
        analysis.validate()

        # compute the missing releasable nodes- in this case, only the gaussian mean
        analysis.release()

        # retrieve the noised mean
        print("gaussian mean", gaussian_mean.value)

        # release a couple other statistics using other mechanisms in the same batch
        actual_sum = sn.sum(col_a_clamped)
        actual_sum.set(123456.)
        laplace_sum = sn.laplace_mechanism(actual_sum, privacy_usage={"epsilon": .1})

        actual_count = sn.count(col_a)
        actual_count.set(9876)

        geo_count = sn.simple_geometric_mechanism(actual_count, 0, 10000, privacy_usage={"epsilon": .1})

        analysis.release()
        print("laplace sum", laplace_sum.value)
        print("geometric count", geo_count.value)

        actual_histogram_b = sn.histogram(sn.clamp(data['B'], categories=['X', 'Y', 'Z'], null_value="W"))
        actual_histogram_b.set([12, 1280, 2345, 12])
        geo_histogram_b = sn.simple_geometric_mechanism(actual_histogram_b, 0, 10000, privacy_usage={"epsilon": .1})

        col_c = sn.to_bool(data['C'], true_label="T")
        actual_histogram_c = sn.histogram(col_c)
        actual_histogram_c.set([5000, 5000])
        lap_histogram_c = sn.simple_geometric_mechanism(actual_histogram_c, 0, 10000, privacy_usage={"epsilon": .1})

        analysis.release()
        print("noised histogram b", geo_histogram_b.value)
        print("noised histogram c", lap_histogram_c.value)
        print("C dimensionality", col_c.dimensionality)
        print("C categories", col_c.categories)

        # multicolumnar insertion

        # pull a column out
        col_rest = sn.to_float(data[['C', 'D']])

        # describe the preprocessing you actually perform on the data
        col_rest_resized = sn.resize(sn.impute(sn.clamp(col_rest, lower=[0., 5.], upper=1000.)), number_rows=10000)

        # run a fake aggregation
        actual_mean = sn.mean(col_rest_resized)

        # insert aggregated data from an external system
        actual_mean.set([[10., 12.]])

        # describe the differentially private operation
        gaussian_mean = sn.gaussian_mechanism(actual_mean, privacy_usage={"epsilon": .4, "delta": 1e-6})

        # check if the analysis is permissible
        analysis.validate()

        # compute the missing releasable nodes- in this case, only the gaussian mean
        analysis.release()

        # retrieve the noised mean
        print("rest gaussian mean", gaussian_mean.value)


@pytest.mark.parametrize(
    "query,max_ids",
    [
        pytest.param([
            {
                'type': 'count',
                'value': 23,
                'privacy_usage': {'epsilon': .2}
            },
            {
                'type': 'sum',
                'value': 202,
                'lower': 0,
                'upper': 1000,
                'privacy_usage': {'epsilon': .4}
            },
            {
                'type': 'sum',
                'value': 120.2,
                'lower': 0.,
                'upper': 1000.,
                'privacy_usage': {'epsilon': .4, 'delta': 1.1e-6}
            }
        ], 5, id="PrivatizeQuery"),
    ],
)
def privatize_query(queries, max_ids):
    with sn.Analysis(group_size=max_ids, ) as analysis:

        data = sn.Dataset(value=[])
        for query in queries:
            if query['type'] == 'count':
                counter = sn.count(data)
                counter.set(query['value'])
                query['privatizer'] = sn.simple_geometric_mechanism(counter, lower=0, upper=1000000, privacy_usage=query['privacy_usage'])

            if query['type'] == 'sum':
                # assuming you don't consider the data type private:
                caster = {
                    int: lambda x: sn.to_int(x, lower=query['lower'], upper=query['upper']),
                    float: sn.to_float
                }[type(query['value'])]

                summer = sn.sum(
                    caster(data),
                    data_columns=1,
                    data_lower=query['lower'],
                    data_upper=query['upper'])
                summer.set(query['value'])

                query['privatizer'] = {
                    int: lambda x: sn.simple_geometric_mechanism(x, lower=0, upper=1000000, privacy_usage=query['privacy_usage']),
                    float: lambda x: sn.gaussian_mechanism(x, privacy_usage=query['privacy_usage'])
                }[type(query['value'])](summer)

    analysis.release()
    for query in queries:
        query['privatized'] = query['privatizer'].value

    return analysis.privacy_usage
