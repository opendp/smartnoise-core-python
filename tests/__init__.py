from os.path import abspath, dirname, isfile, join

# Path to the test csv file

TEST_CSV_PATH = join(dirname(abspath(__file__)), '..', 'data', 'PUMS_california_demographics_1000', 'data.csv')
assert isfile(TEST_CSV_PATH), f'Error: file not found: {TEST_CSV_PATH}'

test_csv_names = ["age", "sex", "educ", "race", "income", "married"]
