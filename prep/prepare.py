import os, glob, shutil
import numpy as np, pandas as pd
from scipy import ndimage
from skimage import io, exposure, morphology
from jarvis.utils import arrays as jars

def create_hdr(pattern='/data/raw/flame/zips/TrainSet/Raw/*.tif', csv='./csvs/meta.csv'):

    sids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(pattern))]

    hard = []
    hard += [sid for sid in sids if '431_' in sid]
    hard += [sid for sid in sids if 'I8' == sid]

    cls1 = [sid for sid in sids if '431_24' in sid]
    cls2 = [sid for sid in sids if 'Pt431_' in sid]
    cls3 = ['I8']
    cls0 = [sid for sid in sids if sid not in cls1 + cls2 + cls3]

    df = pd.DataFrame(index=sids)
    df.index.name = 'sid'

    df['cohort-all'] = True

    # --- 2-class
    df['cohort-easy'] = [s not in hard for s in sids]
    df['cohort-hard'] = [s in hard for s in sids]

    # --- 4-class
    df['cohort-cls0'] = [s in cls0 for s in sids]
    df['cohort-cls1'] = [s in cls1 for s in sids]
    df['cohort-cls2'] = [s in cls2 for s in sids]
    df['cohort-cls3'] = [s in cls3 for s in sids]

    df.to_csv(csv)

def create_raw(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', ignore=('I1',), test=False, skip_existing=True, **kwargs):
    """
    Method to create training data

      lbl : 1   ==> nuclei
      wgt : 1   ==> valid regions (ignoring areas that are not labeled)
            2   ==> hard cells
            3   ==> edge between cells
      dst : <0  ==> distance transform (nuclei)
            >0  ==> distance transform (edges)

    """
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
                pos = create_msk_hard_cells(prd, lbl)
                wgt[pos > 0] = 2
                per = create_msk_edge(lbl)
                wgt[per > 0] = 3

                # --- Derive dst
                dst = ndimage.distance_transform_edt(1 - lbl)
                dst[lbl == 1] = ndimage.distance_transform_edt(lbl)[lbl == 1] * -1
                dst = dst.astype('float16')

            jars.create(data=dat).to_hdf5('/data/raw/flame/proc/raw/{}/dat.hdf5'.format(sid))

            if not test:
                jars.create(data=lbl).to_hdf5('/data/raw/flame/proc/raw/{}/lbl.hdf5'.format(sid))
                jars.create(data=wgt).to_hdf5('/data/raw/flame/proc/raw/{}/wgt.hdf5'.format(sid))
                jars.create(data=dst).to_hdf5('/data/raw/flame/proc/raw/{}/dst.hdf5'.format(sid))

def create_v00(pattern='/data/raw/flame/zips/TrainSet/Raw/*.tif', ignore=('I1',), test=False, skip_existing=True, **kwargs):
    """
    Method to create training data

      lbl : 1   ==> nuclei
      dst : <0  ==> distance transform (nuclei)
            >0  ==> distance transform (edges)

    """
    sids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(pattern))]
    sids = [s for s in sids if s not in ignore]

    for sid in sids:

        print('Creating: {}'.format(sid))

        dat = '{}/{}.tif'.format(os.path.dirname(pattern), sid)
        lbl = '/data/raw/flame/zips/TrainSet/GroundTruth/{}.png'.format(sid)
        wgt = '/data/raw/flame/zips/DrChang/Train/Background/{}.png'.format(sid)

        if not os.path.exists('/data/raw/flame/proc/v00/{}/dat.hdf5'.format(sid)) or not skip_existing:

            dat = load_tif(dat)

            if not test:

                lbl = load_png(lbl, shape=dat.shape)
                wgt = load_png(wgt, shape=dat.shape)

                # --- Derive wgt
                wgt[lbl > 0] = 1

                # --- Derive dst
                dst = ndimage.distance_transform_edt(1 - lbl)
                dst[lbl == 1] = ndimage.distance_transform_edt(lbl)[lbl == 1] * -1
                dst = dst.astype('float16')

            jars.create(data=dat).to_hdf5('/data/raw/flame/proc/v00/{}/dat.hdf5'.format(sid))

            if not test:
                jars.create(data=lbl).to_hdf5('/data/raw/flame/proc/v00/{}/lbl.hdf5'.format(sid))
                jars.create(data=wgt).to_hdf5('/data/raw/flame/proc/v00/{}/wgt.hdf5'.format(sid))
                jars.create(data=dst).to_hdf5('/data/raw/flame/proc/v00/{}/dst.hdf5'.format(sid))

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

def create_msk_hard_cells(prd, lbl, min_size=300, max_pred=0.6, **kwargs):
    """
    Method to identify hard cells

      (1) min_size cutoff: overall larger cell size
      (2) max_pred cutoff: overall inaccurate prediction via /base/ model

    """
    labeled, _ = ndimage.label(lbl > 0)

    cnts = np.bincount(labeled.ravel())[1:]
    vals = np.nonzero(cnts > min_size)[0] + 1

    pos = np.zeros_like(lbl)

    for v in vals:
        if prd[labeled == v].mean() < max_pred:
            pos[labeled == v] = 1

    return pos 

def create_msk_edge(lbl, r=5, min_size=10, **kwargs):

    structure = morphology.disk(radius=r)

    # --- Erode large cells
    lrg = jars.blobs.areaopen(lbl, n=300)
    lbl[lrg > 0] = 0
    lrg = ndimage.binary_erosion(lrg, iterations=1)
    lbl[lrg > 0] = 1

    # --- Binary close
    msk = ndimage.binary_closing(lbl, iterations=3, structure=structure)
    msk[lbl == 1] = 0

    # --- Filter by blobs greater than min size
    labeled, _ = ndimage.label(msk > 0)
    cnts = np.bincount(labeled.ravel())[1:]
    vals = np.nonzero(cnts > min_size)[0] + 1

    # --- Filter by blobs that are between two other blobs
    uniques, _ = ndimage.label(lbl > 0)

    msk_ = np.zeros_like(msk)

    for v in vals:
        m = labeled == v
        u = np.unique(uniques[ndimage.binary_dilation(m)])
        if u[u > 0].size > 1:
            msk_[labeled == v] = 1

    return msk_

def create_prd(pattern='../comp/sens/proc/raw/*/prd.hdf5', **kwargs):

    for src in sorted(glob.glob(pattern)):

        sid = src.split('/')[-2]
        dst = '/data/raw/flame/proc/raw/{}/prd.hdf5'.format(sid)

        if not os.path.exists(dst):
            shutil.copy(src=src, dst=dst)

if __name__ == '__main__':

    # ============================================================================
    # create_hdr()
    # ============================================================================
    # create_raw(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', skip_existing=False)
    # create_raw(pattern='/data/raw/flame/zips/DrChang/Test/TestRaw/*.tif', test=True)
    # ============================================================================
    # create_v00(skip_existing=False)
    # ============================================================================
    # create_prd()
    # ============================================================================
    pass
