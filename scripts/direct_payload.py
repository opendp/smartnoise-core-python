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
with sn.Analysis(filter_level="all"):
    raw_data = np.array([1, 3, 7, 3, 2, 3, 1, 7, 7])
    preprocessed = sn.resize(
            sn.to_float(sn.Dataset(value=raw_data)),
            number_columns=1,
            lower=0., upper=5.)

    print(sn.dp_mean(
        preprocessed,
        implementation="plug-in",
        privacy_usage={"epsilon": 0.1},
        data_lower=0.,
        data_upper=5.).value)
