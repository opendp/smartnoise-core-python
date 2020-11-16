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

# print("SimpleGeometric")
# test_base.test_accuracies("SimpleGeometric")
# print("Laplace")
# test_base.test_accuracies("Laplace")
# print("Snapping")
# test_base.test_accuracies("Snapping")
# print("Analytic Gaussian")
# test_base.test_accuracies("AnalyticGaussian")
# print("Gaussian")
# test_base.test_accuracies("Gaussian")
#
# print("Empirical Accuracies")
# print("SimpleGeometric")
# test_base.test_accuracy_empirical("SimpleGeometric")
# print("Laplace")
# test_base.test_accuracy_empirical("Laplace")
# print("Snapping")
# test_base.test_accuracy_empirical("Snapping")
# print("Analytic Gaussian")
# test_base.test_accuracy_empirical("AnalyticGaussian")
# print("Gaussian")
# test_base.test_accuracy_empirical("Gaussian")

