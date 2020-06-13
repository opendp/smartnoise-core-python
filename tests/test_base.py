import os
from os.path import abspath, dirname, isfile, join
from distutils.util import strtobool

import pytest
import opendp.whitenoise.core as wn

# Path to the test csv file
#
TEST_CSV_PATH = join(dirname(abspath(__file__)), '..', 'data',
                     'PUMS_california_demographics_1000', 'data.csv')
assert isfile(TEST_CSV_PATH), f'Error: file not found: {TEST_CSV_PATH}'

test_csv_names = ["age", "sex", "educ", "race", "income", "married"]

# Used to skip showing plots, etc.
#
IS_CI_BUILD = strtobool(os.environ.get('IS_CI_BUILD', 'False'))


def test_multilayer_analysis(run=True):
    with wn.Analysis() as analysis:
        PUMS = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        age = wn.to_float(PUMS['age'])
        sex = wn.to_bool(PUMS['sex'], true_label="TRUE")

        age_clamped = wn.clamp(age, lower=0., upper=150.)
        age_resized = wn.resize(age_clamped, number_rows=1000)

        race = wn.to_float(PUMS['race'])
        mean_age = wn.dp_mean(
            data=race,
            privacy_usage={'epsilon': .65},
            data_lower=0.,
            data_upper=100.,
            data_rows=500
        )
        analysis.release()

        sex_plus_22 = wn.add(
            wn.to_float(sex),
            22.,
            left_rows=1000, left_lower=0., left_upper=1.)

        wn.dp_mean(
            age_resized / 2. + sex_plus_22,
            privacy_usage={'epsilon': .1},
            data_lower=mean_age - 5.2,
            data_upper=102.,
            data_rows=500) + 5.

        wn.dp_variance(
            data=wn.to_float(PUMS['educ']),
            privacy_usage={'epsilon': .15},
            data_rows=1000,
            data_lower=0.,
            data_upper=12.
        )

        # wn.dp_raw_moment(
        #     wn.to_float(PUMS['married']),
        #     privacy_usage={'epsilon': .15},
        #     data_rows=1000000,
        #     data_lower=0.,
        #     data_upper=12.,
        #     order=3
        # )
        #
        # wn.dp_covariance(
        #     left=wn.to_float(PUMS['age']),
        #     right=wn.to_float(PUMS['married']),
        #     privacy_usage={'epsilon': .15},
        #     left_rows=1000,
        #     right_rows=1000,
        #     left_lower=0.,
        #     left_upper=1.,
        #     right_lower=0.,
        #     right_upper=1.
        # )

    if run:
        analysis.release()

    return analysis


def test_dp_linear_stats(run=True):
    with wn.Analysis() as analysis:
        dataset_pums = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        age = dataset_pums['age']
        analysis.release()

        num_records = wn.dp_count(
            age,
            privacy_usage={'epsilon': .5},
            lower=0,
            upper=10000
        )
        analysis.release()

        print("number of records:", num_records.value)

        vars = wn.to_float(dataset_pums[["age", "income"]])

        covariance = wn.dp_covariance(
            data=vars,
            privacy_usage={'epsilon': .5},
            data_lower=[0., 0.],
            data_upper=[150., 150000.],
            data_rows=num_records)
        print("covariance released")

        num_means = wn.dp_mean(
            data=vars,
            privacy_usage={'epsilon': .5},
            data_lower=[0., 0.],
            data_upper=[150., 150000.],
            data_rows=num_records)

        analysis.release()
        print("covariance:\n", covariance.value)
        print("means:\n", num_means.value)

        age = wn.to_float(age)

        age_variance = wn.dp_variance(
            age,
            privacy_usage={'epsilon': .5},
            data_lower=0.,
            data_upper=150.,
            data_rows=num_records)

        analysis.release()

        print("age variance:", age_variance.value)

        # If I clamp, impute, resize, then I can reuse their properties for multiple statistics
        clamped_age = wn.clamp(age, lower=0., upper=100.)
        imputed_age = wn.impute(clamped_age)
        preprocessed_age = wn.resize(imputed_age, number_rows=num_records)

        # properties necessary for mean are statically known
        mean = wn.dp_mean(
            preprocessed_age,
            privacy_usage={'epsilon': .5}
        )

        # properties necessary for variance are statically known
        variance = wn.dp_variance(
            preprocessed_age,
            privacy_usage={'epsilon': .5}
        )

        # sum doesn't need n, so I pass the data in before resizing
        age_sum = wn.dp_sum(
            imputed_age,
            privacy_usage={'epsilon': .5}
        )

        # mean with lower, upper properties propagated up from prior bounds
        transformed_mean = wn.dp_mean(
            -(preprocessed_age + 2.),
            privacy_usage={'epsilon': .5}
        )

        analysis.release()
        print("age transformed mean:", transformed_mean.value)

        # releases may be pieced together from combinations of smaller components
        custom_mean = wn.laplace_mechanism(
            wn.mean(preprocessed_age),
            privacy_usage={'epsilon': .5})

        custom_maximum = wn.laplace_mechanism(
            wn.maximum(preprocessed_age),
            privacy_usage={'epsilon': .5})

        custom_maximum = wn.laplace_mechanism(
            wn.maximum(preprocessed_age),
            privacy_usage={'epsilon': .5})

        custom_quantile = wn.laplace_mechanism(
            wn.quantile(preprocessed_age, alpha=.5),
            privacy_usage={'epsilon': 500})

        income = wn.to_float(dataset_pums['income'])
        income_max = wn.laplace_mechanism(
            wn.maximum(income, data_lower=0., data_upper=1000000.),
            privacy_usage={'epsilon': 10})

        # releases may also be postprocessed and reused as arguments to more components
        age_sum + custom_maximum * 23.

        analysis.release()
        print("laplace quantile:", custom_quantile.value)

        age_histogram = wn.dp_histogram(
            wn.to_int(age, lower=0, upper=100),
            edges=list(range(0, 100, 25)),
            null_value=150,
            privacy_usage={'epsilon': 2.}
        )

        sex_histogram = wn.dp_histogram(
            wn.to_bool(dataset_pums['sex'], true_label="1"),
            privacy_usage={'epsilon': 2.}
        )

        education_histogram = wn.dp_histogram(
            dataset_pums['educ'],
            categories=["5", "7", "10"],
            null_value="-1",
            privacy_usage={'epsilon': 2.}
        )

        analysis.release()

        print("age histogram: ", age_histogram.value)
        print("sex histogram: ", sex_histogram.value)
        print("education histogram: ", education_histogram.value)

    if run:
        analysis.release()

        # get the mean computed when release() was called
        print(mean.value)
        print(variance.value)

    return analysis


def test_dp_count(run=True):
    with wn.Analysis() as analysis:
        dataset_pums = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        count = wn.dp_count(
            dataset_pums['sex'] == '1',
            privacy_usage={'epsilon': 0.5})

    if run:
        analysis.release()
        print(count.value)

    return analysis


def test_raw_dataset(run=True):
    with wn.Analysis() as analysis:
        data = wn.to_float(wn.Dataset(value=[1., 2., 3., 4., 5.]))

        wn.dp_mean(
            data=data,
            privacy_usage={'epsilon': 1},
            data_lower=0.,
            data_upper=10.,
            data_rows=10,
            data_columns=1)

    if run:
        analysis.release()

    return analysis


def test_everything(run=True):
    with wn.Analysis(dynamic=True) as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        age_int = wn.to_int(data['age'], 0, 150)
        sex = wn.to_bool(data['sex'], "1")
        educ = wn.to_float(data['educ'])
        race = data['race']
        income = wn.to_float(data['income'])
        married = wn.to_bool(data['married'], "1")

        numerics = wn.to_float(data[['age', 'income']])

        # intentionally busted component
        # print("invalid component id ", (sex + "a").component_id)

        # broadcast scalar over 2d, broadcast scalar over 1d, columnar broadcasting, left and right mul
        numerics * 2. + 2. * educ

        # add different values for each column
        numerics + [[1., 2.]]

        # index into first column
        age = numerics[0]
        income = numerics[[False, True]]

        # boolean ops and broadcasting
        mask = sex & married | (~married ^ False) | (age > 50.) | (age_int == 25)

        # numerical clamping
        wn.clamp(numerics, 0., [150., 150_000.])
        wn.clamp(data['educ'], categories=[str(i) for i in range(8, 10)], null_value="-1")

        wn.count(mask)
        wn.covariance(age, income)
        wn.digitize(educ, edges=[1., 3., 10.], null_value=-1)

        # checks for safety against division by zero
        income / 2.
        income / wn.clamp(educ, 5., 20.)

        wn.dp_count(data, privacy_usage={"epsilon": 0.5})
        wn.dp_count(mask, privacy_usage={"epsilon": 0.5})

        wn.dp_histogram(mask, privacy_usage={"epsilon": 0.5})
        age = wn.impute(wn.clamp(age, 0., 150.))
        wn.dp_maximum(age, privacy_usage={"epsilon": 0.5})
        wn.dp_minimum(age, privacy_usage={"epsilon": 0.5})
        wn.dp_median(age, privacy_usage={"epsilon": 0.5})

        age_n = wn.resize(age, number_rows=800)
        wn.dp_mean(age_n, privacy_usage={"epsilon": 0.5})
        wn.dp_raw_moment(age_n, order=3, privacy_usage={"epsilon": 0.5})

        wn.dp_sum(age, privacy_usage={"epsilon": 0.5})
        wn.dp_variance(age_n, privacy_usage={"epsilon": 0.5})

        wn.filter(income, mask)
        race_histogram = wn.histogram(race, categories=["1", "2", "3"], null_value="3")
        wn.histogram(income, edges=[0., 10000., 50000.], null_value=-1)

        wn.dp_histogram(married, privacy_usage={"epsilon": 0.5})

        wn.gaussian_mechanism(race_histogram, privacy_usage={"epsilon": 0.5, "delta": .000001})
        wn.laplace_mechanism(race_histogram, privacy_usage={"epsilon": 0.5, "delta": .000001})

        wn.raw_moment(educ, order=3)

        wn.log(wn.clamp(educ, 0.001, 50.))
        wn.maximum(educ)
        wn.mean(educ)
        wn.minimum(educ)

        educ % 2.
        educ ** 2.

        wn.quantile(educ, .32)

        wn.resize(educ, number_rows=1200, lower=0., upper=50.)
        wn.resize(race, number_rows=1200, categories=["1", "2"], weights=[1, 2])
        wn.resize(data[["age", "sex"]], 1200, categories=[["1", "2"], ["a", "b"]], weights=[1, 2])
        wn.resize(
            data[["age", "sex"]], 1200,
            categories=[["1", "2"], ["a", "b", "c"]],
            weights=[[1, 2], [3, 7, 2]])

        wn.sum(educ)
        wn.variance(educ)

    if run:
        analysis.release()

    return analysis


def test_histogram():
    import numpy as np

    # establish data information

    data = np.genfromtxt(TEST_CSV_PATH, delimiter=',', names=True)
    education_categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]

    income = list(data[:]['income'])
    income_edges = list(range(0, 100_000, 10_000))

    print('actual', np.histogram(income, bins=income_edges)[0])

    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)
        income = wn.to_int(data['income'], lower=0, upper=0)
        sex = wn.to_bool(data['sex'], true_label="1")

        income_histogram = wn.dp_histogram(
            income,
            edges=income_edges,
            privacy_usage={'epsilon': 1.}
        )

    analysis.release()

    print("Income histogram Geometric DP release:   " + str(income_histogram.value))

def test_covariance():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    data = np.genfromtxt(TEST_CSV_PATH, delimiter=',', names=True)

    with wn.Analysis() as analysis:
        wn_data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)
        # get full covariance matrix
        cov = wn.dp_covariance(data=wn.to_float(wn_data['age', 'sex', 'educ', 'income', 'married']),
                               privacy_usage={'epsilon': 10},
                               data_lower=[0., 0., 1., 0., 0.],
                               data_upper=[100., 1., 16., 500_000., 1.],
                               data_rows=1000)
    analysis.release()

    # store DP covariance and correlation matrix
    dp_cov = cov.value
    print(dp_cov)
    dp_corr = dp_cov / np.outer(np.sqrt(np.diag(dp_cov)), np.sqrt(np.diag(dp_cov)))

    # get non-DP covariance/correlation matrices
    age = list(data[:]['age'])
    sex = list(data[:]['sex'])
    educ = list(data[:]['educ'])
    income = list(data[:]['income'])
    married = list(data[:]['married'])
    non_dp_cov = np.cov([age, sex, educ, income, married])
    non_dp_corr = non_dp_cov / np.outer(np.sqrt(np.diag(non_dp_cov)), np.sqrt(np.diag(non_dp_cov)))

    print('Non-DP Covariance Matrix:\n{0}\n\n'.format(pd.DataFrame(non_dp_cov)))
    print('Non-DP Correlation Matrix:\n{0}\n\n'.format(pd.DataFrame(non_dp_corr)))
    print('DP Correlation Matrix:\n{0}'.format(pd.DataFrame(dp_corr)))

    # skip plot step
    if IS_CI_BUILD:
        return

    plt.imshow(non_dp_corr - dp_corr, interpolation='nearest')
    plt.colorbar()
    plt.show()


def test_properties():
    with wn.Analysis():
        # load data
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        # establish data
        age_dt = wn.cast(data['age'], 'FLOAT')

        # ensure data are non-null
        non_null_age_dt = wn.impute(age_dt, distribution='Uniform', lower=0., upper=100.)
        clamped = wn.clamp(age_dt, lower=0., upper=100.)

        # create potential for null data again
        potentially_null_age_dt = non_null_age_dt / 0.

        # print('original properties:\n{0}\n\n'.format(age_dt.properties))
        print('properties after imputation:\n{0}\n\n'.format(non_null_age_dt.nullity))
        print('properties after nan mult:\n{0}\n\n'.format(potentially_null_age_dt.nullity))

        print("lower", clamped.lower)
        print("upper", clamped.upper)
        print("releasable", clamped.releasable)
        # print("props", clamped.properties)
        print("data_type", clamped.data_type)
        print("categories", clamped.categories)


def test_groupby():

    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        partitioned = wn.partition(data[['educ', 'income']], by=is_male)

        counts = {cat: wn.dp_count(partitioned[cat], privacy_usage={'epsilon': 0.1}) for cat in is_male.categories}

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print({cat: counts[cat].value for cat in counts})

    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        partitioned = wn.partition(wn.to_float(data[['educ', 'income']]), by=is_male)

        counts = {
            True: wn.dp_count(partitioned[True], privacy_usage={'epsilon': 0.1}),
            False: wn.dp_mean(partitioned[False],
                              privacy_usage={'epsilon': 0.1},
                              data_rows=500,
                              data_lower=[0., 0.], data_upper=[15., 200_000.])
        }

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print({cat: counts[cat].value for cat in counts})

    # now union the released output
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        means = {}
        for cat in is_male.categories:
            part = partitioned[cat]
            part = wn.resize(part, number_rows=500)
            part = wn.dp_mean(part, privacy_usage={"epsilon": 1.0})
            # print("mean: ", part.properties)
            means[cat] = part

        union = wn.union(means)

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(union.value)

    # now union private data, and apply mechanism after
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        means = {}
        for cat in is_male.categories:
            part = partitioned[cat]
            part = wn.resize(part, number_rows=500)
            part = wn.mean(part)
            means[cat] = part

        union = wn.union(means)
        noised = wn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(noised.value)


def test_fail_groupby():
    with wn.Analysis() as analysis, pytest.raises(Exception):
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        bounds = {"data_lower": [0., 0.], "data_upper": [15., 200_000.], "data_rows": 500}

        union = wn.union({
            True: wn.dp_mean(partitioned[True], privacy_usage={"epsilon": 0.1}, **bounds),
            False: wn.mean(partitioned[False], **bounds),
        })

        wn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

    print(analysis.privacy_usage)


def test_groupby_c_stab():
    # use the same partition multiple times in union
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        def analyze(data):
            return wn.mean(wn.resize(data, number_rows=500))

        means = {
            True: analyze(partitioned[True]),
            False: analyze(partitioned[False]),
            "duplicate_that_inflates_c_stab": analyze(partitioned[True]),
        }

        union = wn.union(means)
        noised = wn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

        # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(noised.value)


def test_multilayer_partition():
    # multilayer partition with mechanisms applied inside partitions
    with wn.Analysis(eager=False) as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        def analyze(data):
            educ = wn.clamp(wn.to_int(wn.index(data, indices=0), lower=0, upper=15), categories=list(range(15)), null_value=-1)
            income = wn.index(data, indices=1)
            repartitioned = wn.partition(income, by=educ)

            inner_count = {}
            inner_means = {}
            for key in [5, 8, 12]:
                educ_level_part = repartitioned[key]

                inner_count[key] = wn.dp_count(educ_level_part, privacy_usage={"epsilon": 0.4})
                inner_means[key] = wn.dp_mean(
                    educ_level_part,
                    privacy_usage={"epsilon": 0.6},
                    data_rows=wn.row_max(1, inner_count[key]))

            return wn.union(inner_means, flatten=False), wn.union(inner_count, flatten=False)

        means = {}
        counts = {}
        for key in partitioned.partition_keys:
            part_means, part_counts = analyze(partitioned[key])
            means[key] = part_means
            counts[key] = part_counts

        means = wn.union(means, flatten=False)
        counts = wn.union(counts, flatten=False)

        # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print("Counts:")
    print(counts.value)

    print("Means:")
    print(means.value)

    #
    # multilayer partition with mechanisms applied after union
    with wn.Analysis(eager=False) as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        educ_inc = wn.impute(wn.clamp(wn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = wn.partition(educ_inc, by=is_male)

        def analyze(data):
            educ = wn.clamp(wn.to_int(wn.index(data, indices=0), lower=0, upper=15), categories=list(range(15)), null_value=-1)
            income = wn.index(data, indices=1)
            repartitioned = wn.partition(income, by=educ)

            inner_count = {}
            inner_means = {}
            for key in [5, 8, 12]:
                educ_level_part = repartitioned[key]

                inner_count[key] = wn.dp_count(educ_level_part, privacy_usage={"epsilon": 0.4})
                inner_means[key] = wn.mean(wn.resize(
                    educ_level_part,
                    number_rows=wn.row_min(1, inner_count[key] * 4 // 5)))

            return wn.union(inner_means), wn.union(inner_count)

        means = {}
        counts = {}
        for key in partitioned.partition_keys:
            part_means, part_counts = analyze(partitioned[key])
            means[key] = part_means
            counts[key] = part_counts

        means = wn.laplace_mechanism(wn.union(means), privacy_usage={"epsilon": 0.6})
        counts = wn.union(counts)

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print("Counts:")
    print(counts.value)

    print("Means:")
    print(means.value)
    print(means.get_accuracy(.05))


def test_dataframe_partitioning():

    # dataframe partition
    with wn.Analysis(eager=False) as analysis:
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        is_male = wn.to_bool(data['sex'], true_label="1")
        partitioned = wn.partition(data, by=is_male)

        means = {
            key: wn.dp_mean(wn.impute(wn.clamp(wn.to_float(partitioned[key]['income']), 0., 200_000.)), implementation="plug-in", privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }

        print(wn.union(means).value)

    # dataframe partition with multi-index grouping
    with wn.Analysis(eager=False):
        data = wn.Dataset(path=TEST_CSV_PATH, column_names=test_csv_names)

        grouper = wn.clamp(
            data[['sex', 'educ']],
            categories=[['0', '1'],
                        [str(i) for i in range(14)]],
            null_value='-1')
        partitioned = wn.partition(data, by=grouper)

        print(wn.union({
            key: wn.dp_count(partitioned[key],
                             privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }).value)

        print(wn.union({
            key: wn.dp_mean(wn.impute(wn.clamp(wn.to_float(partitioned[key]['income']), 0., 200_000.)),
                            implementation="plug-in",
                            privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }).value)

