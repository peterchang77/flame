import os, glob
from skimage import io
from jarvis.utils.general import gpus
from jarvis.utils import arrays as jars
from jarvis.auto.predict import JarvisPipeline

gpus.autoselect()

# ==================================================================
# arr = jars.create('/data/raw/flame/proc/v00/431_241209_Image02_FOV370_z-90_32A1/dat.hdf5')
# arr = jars.create('/data/raw/flame/proc/v00/431_241209_Image02_FOV370_z-90_32A1/win.hdf5')
# ==================================================================
# sid = '431_241106_LeftThigh1_Image01_FOV600_z65_32A0'
# sid = 'Pt431_LT1_Image01_FOV600_z65_32A1'
# sid = '431_241209_Image02_FOV370_z-90_32A1'
# sid = 'I8'
# arr = jars.create('/data/raw/flame/proc/v00/{}/hst.hdf5'.format(sid))
# lbl = jars.create('/data/raw/flame/proc/v00/{}/lbl.hdf5'.format(sid))
# raw = jars.create(data=io.imread('/data/raw/flame/zips/TrainSet/Raw/{}.tif'.format(sid)))
# ==================================================================
# pipeline = JarvisPipeline(yml='./ymls/db-v00.yml')
# prd = pipeline.run(arrs=arr)
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
