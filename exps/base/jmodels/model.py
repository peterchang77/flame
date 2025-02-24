import os
from tensorflow.keras import Model
from jarvis.train import models
from jarvis.utils.general import gpus, printd, printq
from jarvis.auto.network import create_models, create_data
from jarvis.auto.general import load_configs_decorator, show_arrs
from jarvis.auto.predict import create_predictions 

@load_configs_decorator
def test_data(configs, n=1, visualize=True, yk=None, training=True, **kwargs):

    # --- Create inputs and other data sources
    data, inputs, xforms, client = create_data(configs=configs, **kwargs)

    # --- Create model
    model = Model(inputs=inputs, outputs=xforms)

    for i in range(n):

        printd('Loading batch: {:03d}'.format(i))
        batch = next(data['x'])[0]
        xform = model(batch, training=training)

        if visualize:
            show_arrs(arrs=xform, yk=yk, **kwargs)

        if printq('Continue (y/n)? ') != 'y':
            return

@load_configs_decorator
def train(configs, path_configs, **kwargs):

    # --- Create hyperparameters
    params = configs.get('params', {})

    # --- Create inputs and other data sources
    data, inputs, xforms, client = create_data(configs=configs, **kwargs)

    # --- Create models
    backbone, training, blocks, losses, callbacks = create_models(
        names=('backbone', 'training', 'blocks', 'losses', 'callbacks'), 
        configs={k: v for k, v in configs.items() if k in ['layers', 'blocks', 'models']}, 
        inputs=inputs, 
        xforms=xforms)

    # --- Train
    models.train(
        model=training,
        graphs={'backbone': backbone},
        client=client,
        output_dir=os.path.dirname(path_configs) if 'JARVIS_AUTO_CONFIGS' in os.environ else None,
        callbacks=callbacks,
        logdirs='logs',
        **data,
        **params) 

    return client

if __name__ == '__main__':

    # --- Autoselect GPU
    gpus.autoselect()

    # =========================================================
    # TEST DATA
    # =========================================================
    # test_data(path_hyper='./csvs/hyper.csv', row=4, yk='w', n=8)
    # =========================================================

    # --- Train model
    client = train(path_hyper='./csvs/hyper.csv', verbose=True)

    # --- Write stats
    create_predictions(client=client, path_hyper='./csvs/hyper.csv')

    pass
