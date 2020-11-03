import opendp.smartnoise.core as sn
# turn on stack traces from panics
import os
os.environ['RUST_BACKTRACE'] = 'full'

from tests import test_partitioning
from tests import test_components
from tests import test_insertion
from tests import test_mechanisms
from tests import test_base

# establish data information
# data_path = os.path.join('.', 'data', 'PUMS_california_demographics_1000', 'data.csv')
# var_names = ["age", "sex", "educ", "race", "income", "married", "pid"]

# ~~~ SCRAP AREA FOR TESTING ~~~
# test_components.test_dp_median_raw()
# test_components.test_dp_linear_regression()
# test_components.test_dp_covariance()
# test_median()

# test_histogram()
# test_partitioning.test_map_4()
# test_mechanisms.test_mechanism({
#     "mechanism": "Snapping",
#     "privacy_usage": {"epsilon": 2.0, "delta": 1E-6}
# }, test_mechanisms.dp_all_snapping)
# test_mechanisms.test_snapping()
# test_components.test_dp_median()
#
#
# # test_validator_properties.test_dp_mean()
# test_insertion.test_insertion_simple()
# test_components.test_partition()

# def histogram_insertion_simple(hist, total, privacy):
#     """
#     Conduct a differentially private analysis with values inserted from other systems
#     :return:
#     """
#     with sn.Analysis() as analysis:
#         # construct a fake dataset that describes your actual data (will never be run)
#         data = sn.Dataset(path="", column_names=["A", "B", "C", "D"])
#         # run a fake aggregation
#         actual_histogram_b = sn.histogram(sn.clamp(data['B'], categories=list(range(999)), null_value=-1))
#         actual_histogram_b.set(hist)
#         geo_histogram_b = sn.simple_geometric_mechanism(actual_histogram_b, 0, total,
#                                                         privacy_usage={"epsilon": privacy})
#         analysis.release()
#         # check if the analysis is permissible
#         analysis.validate()
#     return geo_histogram_b.value
#
#
# import numpy as np
#
#
# def preprocess(value, component):
#     from opendp.smartnoise.core.value import detect_atomic_type
#     value = np.array(value)
#     # add trailing singleton axes
#     value.shape += (1,) * (component.dimensionality - value.ndim)
#     if value.ndim != component.dimensionality:
#         raise ValueError(f"Expected dimensionality is {component.dimensionality}, but passed dimensionality is {value.ndim}.")
#     if component.dimensionality > 0 and value.shape[0] != component.num_records:
#         raise ValueError(f"Expected {component.num_records} records, but passed {value.shape[0]} records.")
#     if component.dimensionality > 1 and value.shape[1] != component.num_columns:
#         raise ValueError(f"Expected {component.num_columns} columns, but passed {value.shape[1]} columns.")
#     # TODO: make cast.rs accept i64, f64, and standardize .data_type property to i64, f64
#     atomic_type = {"bool": "bool", "i64": "int", "f64": "float", "string": "string"}[detect_atomic_type(value)]
#     if atomic_type != component.data_type:
#         raise ValueError(f"Expected {component.data_type} atomic type, but passed {atomic_type} atomic type.")
#
#     return value

# print(histogram_insertion_simple(range(1000), 100, .000001))

