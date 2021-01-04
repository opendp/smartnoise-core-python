import opendp.smartnoise.core as sn
import random
import string
import numpy as np

from tests import (TEST_PUMS_PATH, TEST_PUMS_NAMES,
                   TEST_EDUC_PATH, TEST_EDUC_NAMES)


def generate_bools():
    private_data = [[True, True], [True, False], [False, True], [False, False]]

    dataset = sn.literal(value=private_data, value_public=False)
    typed = sn.to_bool(dataset, true_label=True)
    return sn.resize(typed, number_columns=2, categories=[True, False])


def generate_synthetic(var_type, n=10, rand_min=0, rand_max=10, cats_str=None, cats_num=None, variants=None):
    cats_str = ['A', 'B', 'C', 'D'] if cats_str is None else cats_str
    cats_num = [0, 1, 2, 3] if cats_num is None else cats_num
    variants = ['Index', 'Random', 'Constant', 'Categories'] if variants is None else variants

    data = []
    names = []

    for variant in variants:
        if var_type == bool:
            data.append(list({
                                 'Index': (bool(i % 2) for i in range(n)),
                                 'Random': (random.choice([True, False]) for _ in range(n)),
                                 'Constant': (bool(1) for _ in range(n)),
                                 'Categories': (bool(random.choice(cats_num)) for _ in range(n))
                             }[variant]))
            names.append('B_' + variant)
        if var_type == float:
            data.append(list({
                                 'Index': (float(i) for i in range(n)),
                                 'Random': (rand_min + random.random() * (rand_max - rand_min) for _ in range(n)),
                                 'Constant': (float(1) for _ in range(n)),
                                 'Categories': (float(random.choice(cats_num)) for _ in range(n)),
                             }[variant]))
            names.append('F_' + variant)
        if var_type == int:
            data.append(list({
                                 'Index': range(n),
                                 'Random': (random.randrange(rand_min, rand_max) for _ in range(n)),
                                 'Constant': (1 for _ in range(n)),
                                 'Categories': (random.choice(cats_num) for _ in range(n)),
                             }[variant]))
            names.append('I_' + variant)
        if var_type == str:
            data.append(list({
                                 'Index': (str(i) for i in range(n)),
                                 'Random': (''.join([random.choice(string.ascii_letters + string.digits)
                                                     for n in range(2)]) for _ in range(n)),
                                 'Constant': (str(1) for _ in range(n)),
                                 'Categories': (random.choice(cats_str) for _ in range(n)),
                             }[variant]))
            names.append('S_' + variant)

    data = list(zip(*data))

    dataset = sn.literal(value=data, value_public=False)
    typed = sn.cast(dataset, atomic_type={
        bool: 'bool', float: 'float', int: 'int', str: 'str'
    }[var_type], true_label=True, lower=0, upper=10)
    resized = sn.resize(typed, number_columns=len(variants), lower=0., upper=10.)
    return sn.to_dataframe(resized, names=names)

def test_dp_covariance():

    # establish data information
    var_names = ["age", "sex", "educ", "race", "income", "married"]

    with sn.Analysis() as analysis:
        wn_data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        # # get scalar covariance
        age_income_cov_scalar = sn.dp_covariance(
            left=sn.to_float(wn_data['age']),
            right=sn.to_float(wn_data['income']),
            privacy_usage={'epsilon': 5000},
            left_lower=0.,
            left_upper=100.,
            left_rows=1000,
            right_lower=0.,
            right_upper=500_000.,
            right_rows=1000)

        data = sn.to_float(wn_data['age', 'income'])
        # get full covariance matrix
        age_income_cov_matrix = sn.dp_covariance(
            data=data,
            privacy_usage={'epsilon': 5000},
            data_lower=[0., 0.],
            data_upper=[100., 500_000.],
            data_rows=1000)

        # get cross-covariance matrix
        cross_covar = sn.dp_covariance(
            left=data,
            right=data,
            privacy_usage={'epsilon': 5000},
            left_lower=[0., 0.],
            left_upper=[100., 500_000.],
            left_rows=1_000,
            right_lower=[0., 0.],
            right_upper=[100., 500_000.],
            right_rows=1000)

    analysis.release()
    print('scalar covariance:\n{0}\n'.format(age_income_cov_scalar.value))
    print('covariance matrix:\n{0}\n'.format(age_income_cov_matrix.value))
    print('cross-covariance matrix:\n{0}'.format(cross_covar.value))


def test_dp_linear_regression():

    with sn.Analysis():
        wn_data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)
        wn_data = sn.resize(
            sn.to_float(wn_data[["age", "income"]]),
            number_rows=1000,
            lower=[0., 0.],
            upper=[100., 500_000.])

        dp_linear_regression = sn.dp_linear_regression(
            data_x=sn.index(wn_data, indices=0),
            data_y=sn.index(wn_data, indices=1),
            privacy_usage={'epsilon': 10.},
            lower_slope=0., upper_slope=1000.,
            lower_intercept=0., upper_intercept=1000.
        )

        print(dp_linear_regression.value)


def test_divide():
    with sn.Analysis():
        data_A = generate_synthetic(float, variants=['Random'])

        f_random = data_A['F_Random']
        imputed = sn.impute(f_random, lower=0., upper=10.)
        clamped_nonzero = sn.clamp(imputed, lower=1., upper=10.)
        clamped_zero = sn.clamp(imputed, lower=0., upper=10.)

        # test properties
        assert f_random.nullity
        assert not imputed.nullity
        assert (2. / imputed).nullity
        assert (f_random / imputed).nullity
        assert (2. / clamped_zero).nullity

        # TODO: fix these assertions in the validator- we should be able to get these tighter bounds
        # assert not (2. / clamped_nonzero).nullity
        # assert not (imputed / 2.).nullity


def test_dp_mean():
    with sn.Analysis():
        data = generate_synthetic(float, variants=['Random'])
        mean = sn.dp_mean(
            data['F_Random'],
            privacy_usage={'epsilon': 0.1},
            data_lower=0.,
            data_upper=10.,
            data_rows=10)

        print("accuracy", mean.get_accuracy(0.05))
        print(mean.from_accuracy(2.3, .05))

    with sn.Analysis():
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)
        print(sn.dp_mean(sn.to_float(data['income']),
                         implementation="plug-in",
                         data_lower=0., data_upper=200_000.,
                         privacy_usage={"epsilon": 0.5}).value)

    with sn.Analysis(protect_sensitivity=False, protect_floating_point=False):
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)
        mean = sn.mean(sn.to_float(data['income']), data_lower=0., data_upper=200_000., data_rows=1000)
        print(sn.gaussian_mechanism(
            mean,
            sensitivity=[[0.0000001]],
            privacy_usage={"epsilon": 0.5, 'delta': .000001}).value)


def test_dp_median():
    with sn.Analysis():
        data = generate_synthetic(float, variants=['Random'])

        candidates = [-10., -2., 2., 3., 4., 7., 10., 12.]

        median_scores = sn.median(
            data['F_Random'],
            candidates=candidates,
            data_rows=10,
            data_lower=0.,
            data_upper=10.)

        dp_median = sn.exponential_mechanism(median_scores, candidates=candidates, privacy_usage={"epsilon": 1.})

        print(dp_median.value)
        assert sn.dp_median(
            data['F_Random'],
            privacy_usage={"epsilon": 1.},
            candidates=candidates,
            data_lower=0.,
            data_upper=10.).value is not None


def test_dp_median_raw():
    with sn.Analysis() as analysis:
        # create a literal data vector, and tag it as private
        data = sn.Component.of([float(i) for i in range(20)], public=False)

        dp_median = sn.dp_median(
            sn.to_float(data),
            privacy_usage={"epsilon": 1.},
            candidates=[-10., -2., 2., 3., 4., 7., 10., 12.],
            data_lower=0.,
            data_upper=10.,
            data_columns=1).value
        print(dp_median)

        # analysis.plot()
        assert dp_median is not None


def test_median_education():
    # import pandas as pd
    # print(pd.read_csv(data_path)['value'].median())
    with sn.Analysis(filter_level="all") as analysis:
        data = sn.Dataset(path=TEST_EDUC_PATH, column_names=TEST_EDUC_NAMES)
        candidates = list(map(float, range(1, 200, 2)))
        median_scores = sn.median(
            sn.impute(sn.to_float(data['value']), 100., 200.),
            candidates=candidates)

        # print(list(zip(candidates, median_scores.value[0])))

        dp_median = sn.exponential_mechanism(
            median_scores,
            candidates=candidates,
            privacy_usage={"epsilon": 100.})
        print(dp_median.value)
    analysis.release()


def test_equal():
    with sn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        equality = sn.index(data, indices=0) == sn.index(data, indices=1)

        analysis.release()
        assert np.array_equal(equality.value, np.array([True, False, False, True]))


def test_partition():
    with sn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        partitioned = sn.partition(data, num_partitions=3)
        analysis.release()
        # print(partitioned.value)

        assert np.array_equal(partitioned.value[0], np.array([[True, True], [True, False]]))
        assert np.array_equal(partitioned.value[1], np.array([[False, True]]))
        assert np.array_equal(partitioned.value[2], np.array([[False, False]]))


def test_histogram():
    # generate raw data
    import numpy as np
    import pandas as pd
    import tempfile
    import os

    n = 1000
    data = np.random.normal(loc=10, scale=25, size=n)
    mean = np.mean(data)
    sd = np.std(data)
    data = pd.DataFrame([(elem - mean) / sd for elem in data])

    with sn.Analysis(), tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'temp_data.csv')
        data.to_csv(data_path)

        print(sn.dp_histogram(
            sn.to_float(sn.Dataset(path=data_path, column_names=['d'])['d']),
            edges=np.linspace(-3., 3., 1000),
            privacy_usage={'epsilon': 0.1}
        ).value)


def test_index():
    with sn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        index_0 = sn.index(data, indices=0)

        analysis.release()
        assert all(a == b for a, b in zip(index_0.value, [True, True, False, False]))
