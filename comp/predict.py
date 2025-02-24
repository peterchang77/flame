import os, glob
from jarvis.utils.general import gpus
from jarvis.utils import arrays as jars
from jarvis.auto.predict import JarvisPipeline

gpus.autoselect()

# ==================================================================
# arr = jars.create('/data/raw/flame/proc/raw/I8/dat.hdf5')
# lbl = jars.create('/data/raw/flame/proc/raw/I8/lbl.hdf5')
# pipeline = JarvisPipeline(yml='./ymls/db-prd.yml')
# outs = pipeline.run(arrs=arr)
# ==================================================================
# arrs = sorted(glob.glob('/data/raw/flame/proc/raw/*/dat.hdf5'))
# arrs = [a for a in arrs if os.path.exists(a.replace('dat.hdf5', 'lbl.hdf5'))]
# sids = [a.split('/')[-2] for a in arrs]
# pipeline = JarvisPipeline(yml='./ymls/db-prd.yml')
# outs = pipeline.run(arrs=arrs, sids=sids, output_dir='./pred')
# ==================================================================
