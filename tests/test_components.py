import opendp.whitenoise.core as wn
import random
import string
import numpy as np


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
    return wn.rename(resized, column_names=names)


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
    with wn.Analysis():
        data = generate_synthetic(float, variants=['Random'])
        mean = wn.dp_mean(
            data['F_Random'],
            # privacy_usage={'epsilon': 0.1},
            accuracy={'value': .2, 'alpha': .05},
            data_lower=0.,
            data_upper=10.,
            data_n=10)

        print("accuracy", mean.get_accuracy(0.05))
        print(mean.from_accuracy(2.3, .05))


def test_dp_median():
    with wn.Analysis(eager=True, dynamic=False) as analysis:
        data = generate_synthetic(float, variants=['Random'])

        dp_median = wn.dp_median(
            data['F_Random'],
            privacy_usage={"epsilon": .1},
            candidates=[-10., -2., 2., 3., 4., 7., 10., 12.],
            data_lower=0.,
            data_upper=10.)

        analysis.release()

        print(dp_median.value)


def test_equal():
    with wn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        equality = data[0] == data[1]

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


def test_index():
    with wn.Analysis(filter_level='all') as analysis:
        data = generate_bools()

        index_0 = data[0]

        analysis.release()
        assert all(a == b for a, b in zip(index_0.value, [True, True, False, False]))
