import os, glob
from jarvis.utils import io 
from jarvis.utils.general import gpus
from jarvis.utils import arrays as jars
from jarvis.auto.predict import JarvisPipeline

gpus.autoselect()

# ==================================================================
# INTERACTIVE
# ==================================================================
# sid = '431_241106_LeftThigh1_Image01_FOV600_z65_32A0'
# sid = 'Pt431_LT1_Image01_FOV600_z65_32A1'
# sid = '431_241209_Image02_FOV370_z-90_32A1'
# sid = 'I8'
# sid = 'I050'
# sid = 'Mosaic14_4x4_FOV600_z25_3Ch_t15_z01'
# sid = 'Image02_FOV600_z80_32A1_Im_00001'
# arr = jars.create('/data/raw/flame/proc/v03/{}/dat.hdf5'.format(sid))
# hst = jars.create('/data/raw/flame/proc/v03/{}/hst.hdf5'.format(sid))
# lbl = jars.create('/data/raw/flame/proc/v02/{}/lbl.hdf5'.format(sid))
# raw = jars.create('/data/raw/flame/zips/06-25/Raw/{}.tif'.format(sid))
# ==================================================================
# pipeline = JarvisPipeline(yml='./ymls/db-v04.yml')
# outs = pipeline.run(arrs=arr, cols=['rpn-raw', 'fpr-raw'])
# mask = outs['rpn-raw'].data > 0.5
# mask[outs['fpr-raw'].data > 0.25] = 0
# ==================================================================
# FULL 
# ==================================================================
# arrs = sorted(glob.glob('/data/raw/flame/zips/DrChang/Test/TestRaw/*.tif')) 
# sids = [a.split('/')[-1].replace('.tif', '') for a in arrs]
# ==================================================================
# arrs = sorted(glob.glob('/data/raw/flame/proc/v01/*/hst.hdf5')) 
# sids = [a.split('/')[-2] for a in arrs]
# pipeline = JarvisPipeline(yml='./ymls/db-v01.yml', save_funcs=io.save_funcs)
# db = pipeline.run(
#     arrs={'hst-raw': arrs}, 
#     sids=sids,
#     # cols=['rpn-raw'], 
#     # cols=['fpr-raw'], 
#     cols=['msk-png'], 
#     output_dir='./pred',
#     skip_existing=False,
#     align_with=False)
# ==================================================================

# ==================================================================
# LEGACY
# ==================================================================
# pipeline = JarvisPipeline(yml='./ymls/db-base.yml')
# base = pipeline.run(arrs=arr)
# pipeline = JarvisPipeline(yml='./ymls/db-edge.yml')
# edge = pipeline.run(arrs=arr)
# prd = (base['prd-raw'].data > 0.25) * (edge['prd-raw'].data < 0.5)
# ==================================================================
# arrs = sorted(glob.glob('/data/raw/flame/proc/raw/*/dat.hdf5'))
# arrs = [a for a in arrs if os.path.exists(a.replace('dat.hdf5', 'lbl.hdf5'))]
# sids = [a.split('/')[-2] for a in arrs]
# pipeline = JarvisPipeline(yml='./ymls/db-edge.yml')
# outs = pipeline.run(arrs=arrs, sids=sids, output_dir='./edge')
# ==================================================================
