import os, glob
import numpy as np, pandas as pd
from scipy import ndimage
from skimage import io, exposure
from jarvis.utils import arrays as jars

def create_hdr(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', csv='./csvs/meta.csv'):

    sids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(pattern))]
    hard = ['I049', 'I050', 'I057', 'I107', 'I108', 'I110', 'I8']

    df = pd.DataFrame(index=sids)
    df.index.name = 'sids'

    df['cohort-all'] = True
    df['cohort-easy'] = [s not in hard for s in sids]
    df['cohort-hard'] = [s in hard for s in sids]

    df.to_csv(csv)

def create_raw(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', ignore=('I1',), test=False, skip_existing=True, **kwargs):

    sids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(pattern))]
    sids = [s for s in sids if s not in ignore]

    for sid in sids:

        print('Creating: {}'.format(sid))

        dat = '{}/{}.tif'.format(os.path.dirname(pattern), sid)
        lbl = '/data/raw/flame/zips/AccumulatedTrainSet/GroundTruth/{}.png'.format(sid)
        wgt = '/data/raw/flame/zips/DrChang/Train/Background/{}.png'.format(sid)
        prd = '../comp/base/proc/raw/{}/prd.hdf5'.format(sid)

        if not os.path.exists('/data/raw/flame/proc/raw/{}/dat.hdf5'.format(sid)) or not skip_existing:

            dat = load_tif(dat)

            if not test:

                lbl = load_png(lbl)
                wgt = load_png(wgt)
                prd = jars.create(prd).data.squeeze()

                # --- Derive wgt
                wgt[lbl > 0] = 1
                pos = filter_by_hard_cells(prd, lbl)
                wgt[pos > 0] = 2

                # --- Derive dst
                dst = ndimage.distance_transform_edt(1 - lbl)
                dst[lbl == 1] = ndimage.distance_transform_edt(lbl)[lbl == 1] * -1
                dst = dst.astype('float16')

            jars.create(data=dat).to_hdf5('/data/raw/flame/proc/raw/{}/dat.hdf5'.format(sid))

            if not test:
                jars.create(data=lbl).to_hdf5('/data/raw/flame/proc/raw/{}/lbl.hdf5'.format(sid))
                jars.create(data=wgt).to_hdf5('/data/raw/flame/proc/raw/{}/wgt.hdf5'.format(sid))
                jars.create(data=dst).to_hdf5('/data/raw/flame/proc/raw/{}/dst.hdf5'.format(sid))

def load_tif(path, shape=(1200, 1200), kernel_size=50):

    if not os.path.exists(path):
        return np.zeros(shape, dtype='float16')

    x = io.imread(path)
    x = exposure.equalize_adapthist(x, kernel_size=kernel_size)
    x = (x - x.mean()) / (x.std())

    return x.astype('float16')

def load_png(path, shape=(1200, 1200)):

    if not os.path.exists(path):
        return np.ones(shape, dtype='uint8')

    x = io.imread(path)

    return (x > 0).astype('uint8')

def filter_by_hard_cells(prd, lbl, min_size=300, max_pred=0.6, **kwargs):

    labeled, _ = ndimage.label(lbl > 0)

    cnts = np.bincount(labeled.ravel())[1:]
    vals = np.nonzero(cnts > min_size)[0] + 1

    pos = np.zeros_like(lbl)

    for v in vals:
        if prd[labeled == v].mean() < max_pred:
            pos[labeled == v] = 1

    return pos 

if __name__ == '__main__':

    # ============================================================================
    # create_hdr()
    # ============================================================================
    # create_raw(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', skip_existing=False)
    # create_raw(pattern='/data/raw/flame/zips/DrChang/Test/TestRaw/*.tif', test=True)
    # ============================================================================
    pass
