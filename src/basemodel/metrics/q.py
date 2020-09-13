'''
Full reference to: https://github.com/andrewekhalel/sewar/blob/master/sewar
'''
import numpy as np


def q_metric(y_true,y_pred):
    return uqi(y_true,y_pred,8)

def _initial_check(GT, P):
    import numpy as np
    import warnings
    assert GT.shape == P.shape, "Supplied images have different sizes " + \
                                str(GT.shape) + " and " + str(P.shape)
    if GT.dtype != P.dtype:
        msg = "Supplied images have different dtypes " + \
              str(GT.dtype) + " and " + str(P.dtype)
        warnings.warn(msg)

    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    return GT.astype(np.float64), P.astype(np.float64)


def _uqi_single(GT, P, ws):
    from scipy.ndimage import uniform_filter

    N = ws ** 2
    window = np.ones((ws, ws))

    GT_sq = GT * GT
    P_sq = P * P
    GT_P = GT * P

    GT_sum = uniform_filter(GT, ws)
    P_sum = uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum * P_sum
    GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
    numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
    denominator1 = N * (GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1 * GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0), (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index] / denominator[index]

    s = int(np.round(ws / 2))
    return np.mean(q_map[s:-s, s:-s])


def uqi(GT, P, ws=8):
    """calculates universal image quality index (uqi).
    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :param ws: sliding window size (default = 8).
    :returns:  float -- uqi value.
    """
    GT, P = _initial_check(GT, P)
    return np.mean([_uqi_single(GT[:, :, i], P[:, :, i], ws) for i in range(GT.shape[2])])
