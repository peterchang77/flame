import numpy as np
from skimage import io, exposure
from jarvis.utils.math import zscore 

def create_hst(raw, **kwargs):

    if type(raw) is str:
        return {}

    raw.data = equalize(raw.data, **kwargs)
    raw.data = multi_window(raw.data, **kwargs)
    raw.data = raw.data.astype('float16')

    return {'hst': raw}

def equalize(x, method='hist', kernel_size=50, **kwargs):

    if method == 'adapthist':
        x = exposure.equalize_adapthist(x, kernel_size=kernel_size)
    else:
        x = exposure.equalize_hist(x)

    x = zscore(x)

    return x

def multi_window(x, epsilon=1e-6, steps=10, lower_clip=0, upper_clip=99.9, axis=-1):

    x = np.squeeze(x)

    c = lambda x, lower, upper : (x.clip(lower, upper) - lower) / (upper - lower + epsilon)

    # --- Create narrow dynamic range images 
    p = np.percentile(x, np.linspace(0, 100, steps + 1))
    m = [c(x, lower, upper) for lower, upper in zip(p[:-1], p[1:])]

    for n in range(2, len(m) + 1):
        if not m[-n].any():
            m[-n][:] = m[-n + 1][:]

    # --- Create full dynamic range (clipped) image
    lower, upper = np.percentile(x, [lower_clip, upper_clip])
    m = [c(x, lower, upper)] + m
    
    # --- Finalize array
    m = np.stack(m, axis=axis)
    m = np.expand_dims(m, axis=0 if axis == -1 else -1)

    return m

def create_msk(rpn, fpr, rpn_thresh=0.5, fpr_thresh=0.25, **kwargs):

    if type(rpn) is str:
        return {}

    if type(fpr) is str:
        return {}

    msk = (rpn.data > rpn_thresh)
    msk[fpr.data > fpr_thresh] = 0

    msk = np.squeeze(msk)
    msk = msk.astype('uint8') * 255

    return {'msk': msk} 
