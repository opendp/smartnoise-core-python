import opendp.whitenoise.core as wn
import random
import string
import numpy as np

from tests import (TEST_PUMS_PATH, TEST_PUMS_NAMES,
                   TEST_EDUC_PATH, TEST_EDUC_NAMES)


def generate_bools():
    private_data = [[True, True], [True, False], [False, True], [False, False]]

    dataset = wn.literal(value=private_data, value_public=False)
    typed = wn.to_bool(dataset, true_label=True)
    return wn.resize(typed, number_columns=2, categories=[True, False])


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

    dataset = wn.literal(value=data, value_public=False)
    typed = wn.cast(dataset, atomic_type={
        bool: 'bool', float: 'float', int: 'int', str: 'str'
    }[var_type], true_label=True, lower=0, upper=10)
    resized = wn.resize(typed, number_columns=len(variants), lower=0., upper=10.)
    return wn.to_dataframe(resized, names=names)


def test_dp_covariance():

    # establish data information
    var_names = ["age", "sex", "educ", "race", "income", "married"]

    with wn.Analysis() as analysis:
        wn_data = wn.Dataset(path=TEST_PUMS_PATH, column_names=var_names)

        # get scalar covariance
        age_income_cov_scalar = wn.dp_covariance(
            left=wn.to_float(wn_data['age']),
            right=wn.to_float(wn_data['income']),
            privacy_usage={'epsilon': 5000},
            left_lower=0.,
            left_upper=100.,
            left_rows=1000,
            right_lower=0.,
            right_upper=500_000.,
            right_rows=1000)

        data = wn.to_float(wn_data['age', 'income'])
        # get full covariance matrix
        age_income_cov_matrix = wn.dp_covariance(
            data=data,
            privacy_usage={'epsilon': 5000},
            data_lower=[0., 0.],
            data_upper=[100., 500_000.],
            data_rows=1000)

        # get cross-covariance matrix
        cross_covar = wn.dp_covariance(
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

    with wn.Analysis():
        wn_data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)
        wn_data = wn.resize(
            wn.to_float(wn_data[["age", "income"]]),
            number_rows=1000,
            lower=[0., 0.],
            upper=[100., 500_000.])

        dp_linear_regression = wn.dp_linear_regression(
            data_x=wn.index(wn_data, indices=0),
            data_y=wn.index(wn_data, indices=1),
            privacy_usage={'epsilon': 10.},
            lower_slope=0., upper_slope=1000.,
            lower_intercept=0., upper_intercept=1000.
        )

        print(dp_linear_regression.value)


def test_divide():
    with wn.Analysis():
        data_A = generate_synthetic(float, variants=['Random'])

        f_random = data_A['F_Random']
        imputed = wn.impute(f_random, lower=0., upper=10.)
        clamped_nonzero = wn.clamp(imputed, lower=1., upper=10.)
        clamped_zero = wn.clamp(imputed, lower=0., upper=10.)

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
    with wn.Analysis() as analysis:
        data = generate_synthetic(float, variants=['Random'])
        mean = wn.dp_mean(
            data['F_Random'],
            privacy_usage={'epsilon': 0.1},
            data_lower=0.,
            data_upper=10.,
            data_rows=10)

        analysis.release()
        print(mean.value)
        print(analysis.report())


def test_dp_median():
    with wn.Analysis():
        data = generate_synthetic(float, variants=['Random'])

        candidates = wn.Component.of([-10., -2., 2., 3., 4., 7., 10., 12.])

        median_scores = wn.median(
            data['F_Random'],
            candidates=candidates,
            data_rows=10,
            data_lower=0.,
            data_upper=10.)

        dp_median = wn.exponential_mechanism(median_scores, candidates=candidates, privacy_usage={"epsilon": 1.})

        print(dp_median.value)

        assert wn.dp_median(
            data['F_Random'],
            privacy_usage={"epsilon": 1.},
            candidates=candidates,
            data_lower=0.,
            data_upper=10.).value is not None


def test_median_education():
    # import pandas as pd
    # print(pd.read_csv(data_path)['value'].median())
    with wn.Analysis(filter_level="all") as analysis:
        data = wn.Dataset(path=TEST_EDUC_PATH, column_names=TEST_EDUC_NAMES)
        candidates = list(map(float, range(1, 200, 2)))
        median_scores = wn.median(
            wn.impute(wn.to_float(data['value']), 100., 200.),
            candidates=candidates)

        # print(list(zip(candidates, median_scores.value[0])))

        dp_median = wn.exponential_mechanism(
            median_scores,
            candidates=candidates,
            privacy_usage={"epsilon": 100.})
        print(dp_median.value)
    analysis.release()


def test_equal():
    with wn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        equality = wn.index(data, indices=0) == wn.index(data, indices=1)

        analysis.release()
        assert np.array_equal(equality.value, np.array([True, False, False, True]))


def test_partition():
    with wn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        partitioned = wn.partition(data, num_partitions=3)
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

    with wn.Analysis(), tempfile.TemporaryDirectory() as temp_dir:
        data_path = os.path.join(temp_dir, 'temp_data.csv')
        data.to_csv(data_path)

        print(wn.dp_histogram(
            wn.to_float(wn.Dataset(path=data_path, column_names=['d'])['d']),
            edges=np.linspace(-3., 3., 1000),
            privacy_usage={'epsilon': 0.1}
        ).value)


def test_index():
    with wn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        index_0 = wn.index(data, indices=0)

        analysis.release()
        assert all(a == b for a, b in zip(index_0.value, [True, True, False, False]))
