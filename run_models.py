import model
import time
from parameters import *

def try_model(updates):

    print('Updating parameters...')
    update_parameters(updates)

    t0 = time.time()
    try:
        model.main()
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))


def cross_rate_sweep():

    updates = {
        'iterations'        : 1001,
        'task'              : 'dms',
        'n_hidden'          : 150,
    }

    for i, rate in enumerate([0.0, 0.01, 0.05, 0.1, 0.2]):
        updates['cross_rate'] = rate
        updates['save_fn'] = 'crossrate{}_dms_v0'.format(i)
        try_model(updates)


def mutation_strength_sweep():

    updates = {
        'iterations'        : 1001,
        'task'              : 'dms',
        'n_hidden'          : 150,
    }

    for i, rate in enumerate([0.1, 0.25, 0.4, 0.6]):
        updates['mutation_strength'] = rate
        updates['save_fn'] = 'mutstrength{}_dms_v0'.format(i)
        try_model(updates)


#cross_rate_sweep()
#mutation_strength_sweep()







double_neurons = {
    'iterations'        : 10001,
    'task'              : 'dms',
    'save_fn'           : 'double_neurons_dms_v1',
    'n_hidden'          : 200,
    'batch_size'        : 256,
    'mutation_strength' : 0.40
}

quad_neurons = {
    'iterations'        : 10001,
    'task'              : 'dms',
    'save_fn'           : 'quad_neurons_dms_v1',
    'n_hidden'          : 400,
    'batch_size'        : 64,
    'mutation_strength' : 0.40
}

output_constant = {
    'iterations'      : 10001,
    'task'            : 'dms',
    'save_fn'         : 'output_constant80_dms_v0',
    'output_constant' : 80,
}

base_model = {
    'iterations'          : 10001,
    'task'                : 'dms',
    'save_fn'             : 'standard_model_dms_v2',
    'use_weight_momentum' : True,
    'n_hidden'            : 400,
    'batch_size'          : 64,
    'freq_cost'           : 1e-3,
}

evo_model = {
    'iterations'          : 200001,
    'cell_type'           : 'rate',
    'membrane_constant'   : 100,
    'task'                : 'dms',
    'learning_method'     : 'ES',
    'ES_learning_rate'    : 0.002,
    'ES_sigma'            : 0.01,
    'save_fn'             : 'evo_model_dms_out_der_NN10_lr002_si01_v0',
    'n_hidden'            : 100,
    'dt'                  : 20,
    'use_latency'         : False,
    'output_constant'     : 20,
    'batch_size'          : 256,
    'freq_cost'           : 1e-4,
    'freq_cost'           : 0.,
    'reciprocal_cost'     : 0.,
    'tuning_height'       : 4.,
    'response_multiplier' : 1,
    'n_networks'          : 2001,
    'delay_time'          : 300,
}

evo_model_adex = {
    'iterations'          : 51,
    'task'                : 'dms',
    'learning_method'     : 'ES',
    'ES_learning_rate'    : 0.0001,
    'ES_sigma'            : 0.01,
    'save_fn'             : 'evo_model_dms_v0',
    'n_hidden'            : 100,
    'batch_size'          : 128,
    'freq_cost'           : 1e-3,
    'n_networks'          : 1001,
}


def evo_model_sweep():
    updates = evo_model

    lr = [1e-4, 1e-3, 1e-2]
    es = [1e-3, 1e-2, 1e-1]

    from itertools import product

    for j in range(5):
        for (il, l), (ie, e) in product(enumerate(lr), enumerate(es)):
            updates['ES_learning_rate'] = l
            updates['ES_sigma']         = e

            updates['save_fn'] = 'evo_dms_adex_lr{}_es{}_v{}'.format(il, ie, j)

            try_model(updates)





#evo_model_sweep()
#quit()
try_model(evo_model)
