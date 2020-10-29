import numpy as np
from tests import TEST_PUMS_PATH, TEST_PUMS_NAMES
import opendp.smartnoise.core as sn


def test_sgd():
    with sn.Analysis():
        PUMS = sn.Dataset(path=TEST_PUMS_PATH, column_names=TEST_PUMS_NAMES)

        sgd_process = sn.dp_sgd(
            data=sn.to_float(PUMS[["age", "sex", "educ", "race", "income", "married"]]),
            theta=np.random.uniform(-10, 10, size=(1, 6)),
            learning_rate=0.1,
            noise_scale=0.1,
            group_size=10,
            gradient_norm_bound=0.5,
            max_iters=100,
            clipping_value=0.5,
            sample_size=100
        )
        print(sgd_process.value)