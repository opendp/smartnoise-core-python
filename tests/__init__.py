from os.path import abspath, dirname, isfile, join

# Path to the test csv file
TEST_PUMS_PATH = join(dirname(abspath(__file__)), '..', 'data', 'PUMS_california_demographics_1000', 'data.csv')
assert isfile(TEST_PUMS_PATH), f'Error: file not found: {TEST_PUMS_PATH}'
TEST_PUMS_NAMES = ["age", "sex", "educ", "race", "income", "married"]

TEST_EDUC_PATH = join(dirname(abspath(__file__)), '..', 'data', 'Synthetic_Education', 'Final Simulated Results Table.csv')
assert isfile(TEST_EDUC_PATH), f'Error: file not found: {TEST_EDUC_PATH}'
TEST_EDUC_NAMES = ["metric_id", "locale_id", "race", "AcademicYear", "foster", "term", "value", "denom", "metric_perc"]
