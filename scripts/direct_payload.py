# from opendp.whitenoise.core.api import LibraryWrapper
#
# whitenoise_library = LibraryWrapper()
# whitenoise_library.validate_analysis(analysis, release)
import os

# import opendp.whitenoise.core as wn
#
# with wn.Analysis(filter_level="all"):
#     raw_data = [0, 1, 2, 3]
#
#     preprocessed = wn.impute(wn.clamp(
#         wn.resize(
#             wn.to_float(wn.Dataset(value=raw_data)),
#             number_rows=4,
#             number_columns=1,
#             lower=0., upper=5.),
#         lower=0., upper=5.))
#
#     mean = wn.mean(preprocessed)
#     dp_mean = wn.laplace_mechanism(mean, privacy_usage={"epsilon": 0.1})
#     print(mean.value)
#     print(dp_mean.value)

import opendp.whitenoise.core as wn
import numpy as np
with wn.Analysis(filter_level="all"):
    raw_data = np.array([1, 3, 7, 3, 2, 3, 1, 7, 7])
    preprocessed = wn.resize(
            wn.to_float(wn.Dataset(value=raw_data)),
            number_columns=1,
            lower=0., upper=5.)

    print(wn.dp_mean(
        preprocessed,
        implementation="plug-in",
        privacy_usage={"epsilon": 0.1},
        data_lower=0.,
        data_upper=5.).value)
