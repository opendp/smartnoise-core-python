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
test_components.test_dp_median_raw()
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