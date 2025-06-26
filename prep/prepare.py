import os, glob, shutil
import numpy as np, pandas as pd
from scipy import ndimage
from skimage import io, exposure, morphology
from jarvis.utils import arrays as jars

def create_hdr(v='v02', csv='./csvs/meta.csv'):

    sids = [p.split('/')[-2] for p in sorted(glob.glob('/data/raw/flame/proc/{}/*/dat.hdf5'.format(v)))]
    v01_ = [p.split('/')[-2] for p in sorted(glob.glob('/data/raw/flame/proc/v01/*/dat.hdf5'))]

    hard = []
    hard += [sid for sid in sids if '431_' in sid]
    hard += [sid for sid in sids if 'I8' == sid]

    cls1 = [sid for sid in sids if '431_24' in sid]
    cls2 = [sid for sid in sids if 'Pt431_' in sid]
    cls3 = ['I8']
    cls4 = [sid for sid in sids if sid not in v01_]
    cls0 = [sid for sid in sids if sid not in cls1 + cls2 + cls3 + cls4]

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
    df['cohort-cls4'] = [s in cls4 for s in sids]

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

            hst = load_tif(dat, method='hist')
            dat = load_tif(dat)
            win = np.expand_dims(multi_window(dat), axis=0)
            hst = np.expand_dims(multi_window(hst), axis=0)

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
            jars.create(data=win).to_hdf5('/data/raw/flame/proc/v00/{}/win.hdf5'.format(sid))
            jars.create(data=hst).to_hdf5('/data/raw/flame/proc/v00/{}/hst.hdf5'.format(sid))

            if not test:
                jars.create(data=lbl).to_hdf5('/data/raw/flame/proc/v00/{}/lbl.hdf5'.format(sid))
                jars.create(data=wgt).to_hdf5('/data/raw/flame/proc/v00/{}/wgt.hdf5'.format(sid))
                jars.create(data=dst).to_hdf5('/data/raw/flame/proc/v00/{}/dst.hdf5'.format(sid))

def create_v01(pattern='/data/raw/flame/zips/TrainSet/Raw/*.tif', ignore=('I1', 'I2', 'I3'), test=False, skip_existing=True, suffix='v01', **kwargs):
    """
    Method to create training data

      lbl : 1  ==> nuclei
      lbl : 2  ==> inner border (1 pixel) 
      lbl : 3  ==> outer border (4 pixel) 
      dst : <0 ==> distance transform (nuclei)
            >0 ==> distance transform (edges)

    """
    sids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(pattern))]
    sids = [s for s in sids if s not in ignore]

    for sid in sids:

        print('Creating: {}'.format(sid))

        dat = '{}/{}.tif'.format(os.path.dirname(pattern), sid)
        lbl = dat.replace('Raw', 'GroundTruth').replace('.tif', '.png')

        if not os.path.exists(lbl):
            print('Error missing file: {}'.format(lbl))

        else:
            if not os.path.exists('/data/raw/flame/proc/{}/{}/dat.hdf5'.format(suffix, sid)) or not skip_existing:

                dat = load_tif(dat, method='hist')
                hst = np.expand_dims(multi_window(dat), axis=0)

                if not test:

                    lbl = load_png(lbl, shape=dat.shape)

                    # --- Derive dst
                    dst = ndimage.distance_transform_edt(1 - lbl)
                    dst[lbl == 1] = ndimage.distance_transform_edt(lbl)[lbl == 1] * -1
                    dst = dst.astype('float16')

                    # --- Derive lbl
                    lbl[(dst <= 0) & (dst >= -1)] = 2
                    lbl[(dst <= 4) & (dst >   0)] = 3

                jars.create(data=dat).to_hdf5('/data/raw/flame/proc/{}/{}/dat.hdf5'.format(suffix, sid))
                jars.create(data=hst).to_hdf5('/data/raw/flame/proc/{}/{}/hst.hdf5'.format(suffix, sid))

                if not test:
                    jars.create(data=lbl).to_hdf5('/data/raw/flame/proc/{}/{}/lbl.hdf5'.format(suffix, sid))
                    jars.create(data=dst).to_hdf5('/data/raw/flame/proc/{}/{}/dst.hdf5'.format(suffix, sid))

def load_tif(path, shape=(1200, 1200), kernel_size=50, method='adapthist'):

    if not os.path.exists(path):
        return np.zeros(shape, dtype='float16')

    x = io.imread(path)

    if method == 'adapthist':
        x = exposure.equalize_adapthist(x, kernel_size=kernel_size)
    else:
        x = exposure.equalize_hist(x)

    x = (x - x.mean()) / (x.std())

    return x.astype('float16')

def multi_window(x, axis=-1):

    c = lambda x, lower, upper : (x.clip(lower, upper) - lower) / (upper - lower + 1e-6)

    p = np.percentile(x, np.linspace(0, 100, 11))
    m = [c(x, lower, upper) for lower, upper in zip(p[:-1], p[1:])]

    for n in range(2, len(m) + 1):
        if not m[-n].any():
            m[-n][:] = m[-n + 1][:]

    lower, upper = np.percentile(x, [0, 99.9])
    m = [c(x, lower, upper)] + m

    return np.stack(m, axis=axis)

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
    # create_hdr(v='v02')
    # ============================================================================
    # create_raw(pattern='/data/raw/flame/zips/AccumulatedTrainSet/Raw/*.tif', skip_existing=False)
    # create_raw(pattern='/data/raw/flame/zips/DrChang/Test/TestRaw/*.tif', test=True)
    # ============================================================================
    # create_v00(skip_existing=False)
    # create_v01(skip_existing=False)
    # create_v01(pattern='/data/raw/flame/zips/06-25/Raw/*.tif', suffix='v02', skip_existing=False)
    # ============================================================================
    # create_prd()
    # ============================================================================
    pass
