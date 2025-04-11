import os, glob
from jarvis.utils import io
from jarvis.utils.general import tools as jtools
from jarvis.auto.predict import JarvisPipeline

def run(data='/data', main='./main.py', **kwargs):

    # --- Set Jarvis paths
    code = os.path.abspath(os.path.dirname(main))
    jtools.set_paths(code, project_id='flame')

    # --- Set arrs / sids 
    arrs = sorted(glob.glob('{}/*/*.tif'.format(data)))
    sids = [a.split('/')[-2].replace('.tif', '') for a in arrs]

    # --- Create pipeline 
    pipeline = JarvisPipeline(
        yml='{}/comp/ymls/db-v01.yml'.format(code), 
        ignore_proc=True,
        save_funcs=io.save_funcs)

    # --- Run pipeline
    db = pipeline.run(
        arrs=arrs, 
        sids=sids,
        cols=['hst-raw', 'rpn-raw', 'fpr-raw', 'msk-png'], 
        output_dir=data,
        stepwise=True,
        skip_existing=False,
        align_with=False)
