import opendp.whitenoise.core as wn


def test_insertion_simple():
    """
    Conduct a differentially private analysis with values inserted from other systems
    :return:
    """
    with wn.Analysis() as analysis:

        # construct a fake dataset that describes your actual data (will never be run)
        data = wn.Dataset(path="", column_names=["A", "B", "C", "D"])

        # pull a column out
        col_a = wn.to_float(data['A'])

        # describe the preprocessing you actually perform on the data
        col_a_clamped = wn.impute(wn.clamp(col_a, lower=0., upper=10.))
        col_a_resized = wn.resize(col_a_clamped, number_rows=1000000)

        # run a fake aggregation
        actual_mean = wn.mean(col_a_resized)

        # insert aggregated data from an external system
        actual_mean.set(10)

        # describe the differentially private operation
        gaussian_mean = wn.gaussian_mechanism(actual_mean, privacy_usage={"epsilon": .4, "delta": 1e-6})

        # check if the analysis is permissible
        analysis.validate()

        # compute the missing releasable nodes- in this case, only the gaussian mean
        analysis.release()

        # retrieve the noised mean
        print("gaussian mean", gaussian_mean.value)

        # release a couple other statistics using other mechanisms in the same batch
        actual_sum = wn.sum(col_a_clamped)
        actual_sum.set(123456)
        laplace_sum = wn.laplace_mechanism(actual_sum, privacy_usage={"epsilon": .1})

        actual_count = wn.count(col_a)
        actual_count.set(9876)

        geo_count = wn.simple_geometric_mechanism(actual_count, 0, 10000, privacy_usage={"epsilon": .1})

        analysis.release()
        print("laplace sum", laplace_sum.value)
        print("geometric count", geo_count.value)

        actual_histogram_b = wn.histogram(wn.clamp(data['B'], categories=['X', 'Y', 'Z'], null_value="W"))
        actual_histogram_b.set([12, 1280, 2345, 12])
        geo_histogram_b = wn.simple_geometric_mechanism(actual_histogram_b, 0, 10000, privacy_usage={"epsilon": .1})

        col_c = wn.to_bool(data['C'], true_label="T")
        actual_histogram_c = wn.histogram(col_c)
        actual_histogram_c.set([5000, 5000])
        lap_histogram_c = wn.laplace_mechanism(actual_histogram_c, privacy_usage={"epsilon": .1})

        analysis.release()
        print("noised histogram b", geo_histogram_b.value)
        print("noised histogram c", lap_histogram_c.value)
        print("C dimensionality", col_c.dimensionality)
        print("C categories", col_c.categories)

        # multicolumnar insertion

        # pull a column out
        col_rest = wn.to_float(data[['C', 'D']])

        # describe the preprocessing you actually perform on the data
        col_rest_resized = wn.resize(wn.impute(wn.clamp(col_rest, lower=[0., 5.], upper=1000.)), number_rows=10000)

        # run a fake aggregation
        actual_mean = wn.mean(col_rest_resized)

        # insert aggregated data from an external system
        actual_mean.set([[10., 12.]])

        # describe the differentially private operation
        gaussian_mean = wn.gaussian_mechanism(actual_mean, privacy_usage={"epsilon": .4, "delta": 1e-6})

        # check if the analysis is permissible
        analysis.validate()

        # compute the missing releasable nodes- in this case, only the gaussian mean
        analysis.release()

        # retrieve the noised mean
        print("rest gaussian mean", gaussian_mean.value)
