# load libraries
import os
import pprint

import opendp.whitenoise.core as wn

# establish data information
data_path = os.path.join('/Users', 'ethancowan', 'IdeaProjects', 'whitenoise-samples', 'analysis', 'data', 'PUMS_california_demographics_1000', 'data.csv')
var_names = ["age", "sex", "educ", "race", "income", "married", "pid"]


def test_reports():
    """
    This is for generating and testing JSON validation
    :return:
    """
    with wn.Analysis() as analysis:
        # load data
        data = wn.Dataset(path = data_path, column_names = var_names)

        # get mean of age
        age_mean = wn.dp_mean(data = wn.to_float(data['age']),
                              privacy_usage = {'epsilon': .65},
                              data_lower = 0.,
                              data_upper = 100.,
                              data_rows = 1000
                              )
        print("Pre-Release\n")
        print("DP mean of age: {0}".format(age_mean.value))
        print("Privacy usage: {0}\n\n".format(analysis.privacy_usage))

        analysis.release()
        print("Report: ")
        r = analysis.report()
        for x in r:
            pprint.pprint(x)
        print("Post-Release\n")
        print("DP mean of age: {0}".format(age_mean.value))
        print("Privacy usage: {0}\n\n".format(analysis.privacy_usage))


if __name__ == '__main__':
    test_reports()