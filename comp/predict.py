import os, glob
from jarvis.utils.general import gpus
from jarvis.utils import arrays as jars
from jarvis.auto.predict import JarvisPipeline

gpus.autoselect()

# ==================================================================
# arr = jars.create('/data/raw/flame/proc/v00/431_241209_Image02_FOV370_z-90_32A1/dat.hdf5')
# arr = jars.create('/data/raw/flame/proc/v00/431_241209_Image02_FOV370_z-90_32A1/win.hdf5')
arr = jars.create('/data/raw/flame/proc/v00/431_241209_Image02_FOV370_z-90_32A1/hst.hdf5')
# ==================================================================
# lbl = jars.create('/data/raw/flame/proc/v00/Pt431_LT1_Image05_FOV600_z125_32A1/lbl.hdf5')
pipeline = JarvisPipeline(yml='./ymls/db-v00.yml')
prd = pipeline.run(arrs=arr)
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
