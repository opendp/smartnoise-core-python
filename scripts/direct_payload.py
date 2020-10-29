# from opendp.smartnoise.core.api import LibraryWrapper
#
# smartnoise_library = LibraryWrapper()
# smartnoise_library.validate_analysis(analysis, release)
import os

# import opendp.smartnoise.core as sn
#
# with sn.Analysis(filter_level="all"):
#     raw_data = [0, 1, 2, 3]
#
#     preprocessed = sn.impute(sn.clamp(
#         sn.resize(
#             sn.to_float(sn.Dataset(value=raw_data)),
#             number_rows=4,
#             number_columns=1,
#             lower=0., upper=5.),
#         lower=0., upper=5.))
#
#     mean = sn.mean(preprocessed)
#     dp_mean = sn.laplace_mechanism(mean, privacy_usage={"epsilon": 0.1})
#     print(mean.value)
#     print(dp_mean.value)

import opendp.smartnoise.core as sn
import numpy as np


with sn.Analysis() as analysis:
    # load data
    priv_data = sn.Dataset(value=[1, 3, 7, 3, 2, 3, 1, 7, 7])

    # estimate sample size
    count = sn.dp_count(priv_data, privacy_usage={'epsilon': .05})

    # load data
    priv_data = sn.resize(
        sn.to_float(priv_data),
        number_columns=1,
        number_rows=sn.row_max(1, count),
        lower=0., upper=100_000.)

    priv_data = sn.impute(sn.clamp(priv_data, lower=0., upper=100_000.))

    # get mean
    mean = sn.dp_mean(priv_data, privacy_usage={'epsilon': 0.1})

    # get median
    median = sn.dp_median(priv_data, privacy_usage={'epsilon': 0.1})

    # get min
    _min = sn.dp_minimum(priv_data, privacy_usage={'epsilon': 0.1})

    # get max
    _max = sn.dp_maximum(priv_data, privacy_usage={'epsilon': 0.1})

    analysis.release()
    print(analysis.report())