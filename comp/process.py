import os, numpy as np
from scipy import ndimage
from matplotlib import pyplot 

def create_cmb(base='./sens', edge='./edge', comb='./comb', thresh_base=0.25, thresh_edge=0.50, min_size=20, **kwargs):

    sids = [p.split('/')[-1] for p in sorted(glob.glob('{}/proc/raw/*'.format(base)))]

    for n, sid in enumerate(sids):
        print('Creating: {:04d} / {:04d}'.format(n + 1, len(sids)), end='\r')
        
        b = '{}/proc/raw/{}/prd.hdf5'.format(base, sid)
        e = '{}/proc/raw/{}/prd.hdf5'.format(edge, sid)

        if os.path.exists(b) and os.path.exists(e):

            b = jars.create(b)
            e = jars.create(e)

            m = (b.data > thresh_base) * (e.data < thresh_edge)
            m = m.squeeze()
            m = jars.blobs.areaopen(m, min_size)

            os.makedirs('{}/{}'.format(comb, sid), exist_ok=True)
            pyplot.imsave('{}/{}/msk.png'.format(comb, sid), m, cmap='gray')

def create_lbl(arr, thresh_start=0.90, thresh_end=0.5, thresh_step=0.01, repeats=1, **kwargs):

    prd = np.squeeze(arr.data)
    assert prd.ndim == 2

    # --- Provide initial label
    lbl, _ = ndimage.label(prd > thresh_start)

    for thresh in np.arange(thresh_start, thresh_end, -thresh_step):
        for repeat in range(repeats):

            print('Running: thresh = {:0.3f} | repeat = {:02d}'.format(thresh, repeat), end='\r')

            lbl, cont = step(lbl=lbl, prd=prd, thresh=thresh, repeat=repeat)

            if not cont:
                break

    return lbl

def step(lbl, prd, thresh=0.5, repeat=0, **kwargs):

    cont = False

    # =======================================================
    # (1) CHECK DILATION
    # =======================================================

    nxt = (prd >= thresh) & perim(lbl) 
    nxt = np.stack(np.nonzero(nxt), axis=-1)

    if nxt.size > 0:

        # --- Check all neighbors
        for n in nxt:
            u = neighbor(coord=n, lbl=lbl)
            if u.size == 1:
                cont = True
                lbl[n[0], n[1]] = u

    # =======================================================
    # (2) CHECK FOR NEW SEEDS 
    # =======================================================

    if repeat == 0:

        nxt = (prd >= thresh)
        nxt, l = ndimage.label(nxt)

        for val in range(1, l + 1):

            # --- Check for lbl overlap 
            if not lbl[(nxt == val)].any():
                cont = True
                lbl[(nxt == val)] = np.max(lbl) + 1

    return lbl, cont 

def perim(lbl):

    p = ndimage.binary_dilation(lbl > 0)
    p = p & (lbl == 0)

    return p

def neighbor(coord, lbl):

    lo = (coord - 1).clip(min=0)
    hi = (coord + 2).clip(max=max(lbl.shape))

    u = np.unique(lbl[lo[0]:hi[0], lo[1]:hi[1]])

    return u[u > 0]

def load(sids):

    if type(sids) is str:
        sids = [sids]

    dats = []
    lbls = []
    prds = []

    for sid in sids:

        dat = '/data/raw/flame/proc/raw/{}/dat.hdf5'.format(sid)
        lbl = '/data/raw/flame/proc/raw/{}/lbl.hdf5'.format(sid)
        prd = './pred/proc/raw/{}/prd.hdf5'.format(sid)

        if os.path.exists(dat):

            dats.append(jars.create(dat).data)
            lbls.append(jars.create(lbl).data)
            prds.append(jars.create(prd).data)

    return np.concatenate(dats), np.concatenate(lbls), np.concatenate(prds) 

if __name__ == '__main__':

    create_cmb()

    pass
