

import pytest
import opendp.smartnoise.core as sn
from tests import TEST_PUMS_PATH, TEST_PUMS_NAMES


def test_groupby_1():

    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        partitioned = sn.partition(data[['educ', 'income']], by=is_male)

        counts = {cat: sn.dp_count(partitioned[cat], privacy_usage={'epsilon': 0.1}) for cat in is_male.categories}

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print({cat: counts[cat].value for cat in counts})


def test_groupby_2():
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        partitioned = sn.partition(sn.to_float(data[['educ', 'income']]), by=is_male)

        counts = {
            True: sn.dp_count(partitioned[True], privacy_usage={'epsilon': 0.1}),
            False: sn.dp_mean(partitioned[False],
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
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)


        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        means = {}
        for cat in is_male.categories:
            part = partitioned[cat]
            part = sn.resize(part, number_rows=500)
            part = sn.dp_mean(part, privacy_usage={"epsilon": 1.0})
            # print("mean: ", part.properties)
            means[cat] = part

        union = sn.union(means)

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(union.value)


def test_groupby_4():
    # now union private data, and apply mechanism after
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)


        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        means = {}
        for cat in is_male.categories:
            part = partitioned[cat]
            part = sn.resize(part, number_rows=500)
            part = sn.mean(part)
            means[cat] = part

        union = sn.union(means)
        noised = sn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

    # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(noised.value)


def test_groupby_5():
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        bounds = {"data_lower": [0., 0.], "data_upper": [15., 200_000.], "data_rows": 500}

        union = sn.union({
            True: sn.mean(partitioned[True], **bounds),
            False: sn.mean(partitioned[False], **bounds),
        })

        sn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

        print(analysis.privacy_usage)


def test_groupby_c_stab():
    # use the same partition multiple times in union
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        def analyze(data):
            return sn.mean(sn.resize(data, number_rows=500))

        means = {
            True: analyze(partitioned[True]),
            False: analyze(partitioned[False]),
            "duplicate_that_inflates_c_stab": analyze(partitioned[True]),
        }

        union = sn.union(means)
        noised = sn.laplace_mechanism(union, privacy_usage={"epsilon": 1.0})

        # analysis.plot()
    analysis.release()
    print(analysis.privacy_usage)
    print(noised.value)


def test_multilayer_partition_1():
    # multilayer partition with mechanisms applied inside partitions
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        def analyze(data):
            educ = sn.clamp(sn.to_int(sn.index(data, indices=0), lower=0, upper=15), categories=list(range(15)), null_value=-1)
            income = sn.index(data, indices=1)
            repartitioned = sn.partition(income, by=educ)

            inner_count = {}
            inner_means = {}
            for key in [5, 8, 12]:
                educ_level_part = repartitioned[key]

                inner_count[key] = sn.dp_count(educ_level_part, privacy_usage={"epsilon": 0.4})
                inner_means[key] = sn.dp_mean(
                    educ_level_part,
                    privacy_usage={"epsilon": 0.6},
                    data_rows=sn.row_max(1, inner_count[key]))

            return sn.union(inner_means, flatten=False), sn.union(inner_count, flatten=False)

        means = {}
        counts = {}
        for key in partitioned.partition_keys:
            part_means, part_counts = analyze(partitioned[key])
            means[key] = part_means
            counts[key] = part_counts

        means = sn.union(means, flatten=False)
        counts = sn.union(counts, flatten=False)

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
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        educ_inc = sn.impute(sn.clamp(sn.to_float(data[['educ', 'income']]), lower=[0., 0.], upper=[15., 200_000.]))

        partitioned = sn.partition(educ_inc, by=is_male)

        def analyze(data):
            educ = sn.clamp(sn.to_int(sn.index(data, indices=0), lower=0, upper=15), categories=list(range(15)), null_value=-1)
            income = sn.index(data, indices=1)
            repartitioned = sn.partition(income, by=educ)

            inner_count = {}
            inner_means = {}
            for key in [5, 8, 12]:
                educ_level_part = repartitioned[key]

                inner_count[key] = sn.dp_count(educ_level_part, privacy_usage={"epsilon": 0.4})
                inner_means[key] = sn.mean(sn.resize(
                    educ_level_part,
                    number_rows=sn.row_min(1, inner_count[key] * 4 // 5)))

            return sn.union(inner_means), sn.union(inner_count)

        means = {}
        counts = {}
        for key in partitioned.partition_keys:
            part_means, part_counts = analyze(partitioned[key])
            means[key] = part_means
            counts[key] = part_counts

        means = sn.laplace_mechanism(sn.union(means), privacy_usage={"epsilon": 0.6})
        counts = sn.union(counts)

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
    with sn.Analysis(protect_floating_point=False) as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        is_male = sn.to_bool(data['sex'], true_label="1")
        partitioned = sn.partition(data, by=is_male)

        print(sn.union({
            key: sn.dp_mean(sn.impute(sn.clamp(sn.to_float(partitioned[key]['income']), 0., 200_000.)),
                            implementation="plug-in",
                            privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }).value)
        print(analysis.privacy_usage)


def test_dataframe_partitioning_2():
    # dataframe partition with multi-index grouping
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        grouper = sn.clamp(
            data[['sex', 'educ']],
            categories=[['0', '1'],
                        [str(i) for i in range(14)]],
            null_value='-1')
        partitioned = sn.partition(data, by=grouper)

        sn.union({
            key: sn.dp_count(partitioned[key],
                             privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }, flatten=False)

        print(sn.union({
            key: sn.dp_mean(sn.to_float(partitioned[key]['income']),
                            implementation="plug-in",
                            # data_rows=100,
                            data_lower=0., data_upper=200_000.,
                            privacy_usage={"epsilon": 0.5})
            for key in partitioned.partition_keys
        }).value)
        print(analysis.privacy_usage)


def test_map_1():
    # map a count over all dataframe partitions
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = sn.partition(
            data,
            by=sn.to_bool(data['sex'], true_label="1"))

        counts = sn.dp_count(
            partitioned,
            privacy_usage={"epsilon": 0.5})

        print(counts.value)
        print(analysis.privacy_usage)


def test_map_2():
    # map a count over a large number of tuple partitions of dataframes
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        grouper = sn.clamp(
            data[['sex', 'educ']],
            categories=[['0', '1'],
                        [str(i) for i in range(14)]],
            null_value='-1')
        partitioned = sn.partition(data, by=grouper)

        counts = sn.dp_count(
            partitioned,
            privacy_usage={"epsilon": 0.5})

        print(counts.value)
        print(analysis.privacy_usage)


def test_map_3():
    # chain multiple maps over an array partition with implicit preprocessing
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = sn.partition(
            sn.to_float(data['age']),
            by=sn.to_bool(data['sex'], true_label="1"))

        means = sn.dp_mean(
            partitioned,
            privacy_usage={'epsilon': 0.1},
            data_rows=500,
            data_lower=0., data_upper=15.)

        print(means.value)
        print(analysis.privacy_usage)


def test_map_4():
    # chain multiple mapped releases over a partition with implicit preprocessing
    with sn.Analysis() as analysis:
        data = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        partitioned = sn.partition(
            sn.to_float(data['age']),
            by=sn.to_bool(data['sex'], true_label="1"))

        counts = sn.row_max(
            1, sn.dp_count(partitioned, privacy_usage={'epsilon': 0.5}))

        means = sn.dp_mean(
            partitioned,
            privacy_usage={'epsilon': 0.7},
            data_rows=counts,
            data_lower=0., data_upper=15.)

        print("counts:", counts.value)
        print("means:", means.value)

        print(analysis.privacy_usage)
