

import pytest
import opendp.whitenoise.core as wn
from tests import TEST_PUMS_PATH, TEST_PUMS_NAMES


def test_groupby_1():

    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = wn.to_bool(data['sex'], true_label="1")
        partitioned = wn.partition(data[['educ', 'income']], by=is_male)

        counts = {cat: wn.dp_count(partitioned[cat], privacy_usage={'epsilon': 0.1}) for cat in is_male.categories}

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print({cat: counts[cat].value for cat in counts})


def test_groupby_2():
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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


def test_groupby_3():
    # now union the released output
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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


def test_groupby_4():
    # now union private data, and apply mechanism after
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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
    with wn.Analysis() as analysis, pytest.raises(RuntimeError):
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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


def test_multilayer_partition_1():
    # multilayer partition with mechanisms applied inside partitions
    with wn.Analysis(eager=False) as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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
    print("releasing")
    print(len(analysis.components.items()))
    analysis.release()
    print(analysis.privacy_usage)
    print("Counts:")
    print(counts.value)

    print("Means:")
    print(means.value)


def test_multilayer_partition_2():
    #
    # multilayer partition with mechanisms applied after union
    with wn.Analysis(eager=False) as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

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


def test_dataframe_partitioning_1():

    # dataframe partition
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = wn.to_bool(data['sex'], true_label="1")
        partitioned = wn.partition(data, by=is_male)

        print(wn.union({
            key: wn.dp_mean(wn.impute(wn.clamp(wn.to_float(partitioned[key]['income']), 0., 200_000.)),
                            implementation="plug-in",
                            privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }).value)
        print(analysis.privacy_usage)


def test_dataframe_partitioning_2():
    # dataframe partition with multi-index grouping
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        grouper = wn.clamp(
            data[['sex', 'educ']],
            categories=[['0', '1'],
                        [str(i) for i in range(14)]],
            null_value='-1')
        partitioned = wn.partition(data, by=grouper)

        wn.union({
            key: wn.dp_count(partitioned[key],
                             privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }, flatten=False)

        print(wn.union({
            key: wn.dp_mean(wn.to_float(partitioned[key]['income']),
                            implementation="plug-in",
                            # data_rows=100,
                            data_lower=0., data_upper=200_000.,
                            privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }))
        print(analysis.privacy_usage)


def test_map_1():
    # map a count over all dataframe partitions
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = wn.partition(
            data,
            by=wn.to_bool(data['sex'], true_label="1"))

        counts = wn.dp_count(
            partitioned,
            privacy_usage={"epsilon": 0.5})

        print(counts.value)
        print(analysis.privacy_usage)


def test_map_2():
    # map a count over a large number of tuple partitions of dataframes
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        grouper = wn.clamp(
            data[['sex', 'educ']],
            categories=[['0', '1'],
                        [str(i) for i in range(14)]],
            null_value='-1')
        partitioned = wn.partition(data, by=grouper)

        counts = wn.dp_count(
            partitioned,
            privacy_usage={"epsilon": 0.5})

        print(counts.value)
        print(analysis.privacy_usage)


def test_map_3():
    # chain multiple maps over an array partition with implicit preprocessing
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = wn.partition(
            wn.to_float(data['age']),
            by=wn.to_bool(data['sex'], true_label="1"))

        means = wn.dp_mean(
            partitioned,
            privacy_usage={'epsilon': 0.1},
            data_rows=500,
            data_lower=0., data_upper=15.)

        print(means.value)
        print(analysis.privacy_usage)


def test_map_4():
    # chain multiple mapped releases over a partition with implicit preprocessing
    with wn.Analysis() as analysis:
        data = wn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = wn.partition(
            wn.to_float(data['age']),
            by=wn.to_bool(data['sex'], true_label="1"))

        counts = wn.row_max(
            1, wn.dp_count(partitioned, privacy_usage={'epsilon': 0.5}))

        means = wn.dp_mean(
            partitioned,
            privacy_usage={'epsilon': 0.7},
            data_rows=counts,
            data_lower=0., data_upper=15.)

        print("counts:", counts.value)
        print("means:", means.value)

        print(analysis.privacy_usage)
