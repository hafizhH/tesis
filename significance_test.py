
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError as MeanSquaredErrorLoss
from tensorflow.keras.metrics import R2Score, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error
from keras_tuner.engine import hyperparameters as hyperparameters_module

from pso_tuner import PsoParallelTuner, LstmHyperModel, LstmMvmdHyperModel, ArimaHyperModel, get_stock_market_data, set_all_seeds, create_dataset
import multiprocessing
import os, sys, json

json_db_lock = multiprocessing.Lock()

def init_globals(l):
    global json_db_lock
    json_db_lock = l

def save_to_json_db(key, value, overwrite_key=False, filename="./database.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if json_db_lock is not None:
        json_db_lock.acquire()
    try:
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump({}, f)
        with open(filename, "r") as f:
            db = json.load(f)
        if overwrite_key or key not in db:
            db[key] = value
        with open(filename, "w") as f:
            json.dump(db, f, indent=2)
    finally:
        if json_db_lock is not None:
            json_db_lock.release()

def load_from_json_db(key=None, filename="./database.json"):
    if json_db_lock is not None:
        json_db_lock.acquire()
    try:
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            db = json.load(f)
        return db.get(key, None) if key is not None else db
    finally:
        if json_db_lock is not None:
            json_db_lock.release()

def get_stock_historical(ticker_symbol):
    stock_data = get_stock_market_data(ticker_symbol)
    stock_historical = stock_data['historical_data'].reset_index().drop(['Date', 'Dividends', 'Stock Splits'], axis=1)
    return stock_historical

def compile_pso_report(tuner):
    trials = tuner.oracle.get_all_trials()
    params = tuner.oracle.get_state()['pso']['params']
    iterations = []
    global_best = {
        'position': {},
        'score': np.inf
    }

    for iter in range(params['max_iter']):
        particles = []
        for particle_index in range(params['N']):
            trial_id = "{0:03d}".format(iter * params['N'] + particle_index)
            trial = trials.get(trial_id, None)
            if trial is None:
                print(f"Trial not found for Iteration {iter}, Particle {particle_index}")
                continue

            particles.append({
                'position': trial.hyperparameters.values,
                'score': trial.score,
            })
            if (tuner.oracle.objective.direction == 'min' and trial.score < global_best['score']) or (tuner.oracle.objective.direction == 'max' and trial.score > global_best['score']):
                global_best = {
                    'position': trial.hyperparameters.values,
                    'score': trial.score,
                }

        iterations.append({
            'iter': iter,
            'global_best': global_best,
            'particles': particles,
        })

    return {
        'params': params,
        'iterations': iterations,
    }

def save_tuner_report(key, tuner, report):
    report_data = {
        'best_hp': tuner.get_best_hyperparameters(1)[0].values,
        'pso': compile_pso_report(tuner),
        'retraining': {
            'train_history': report['train_history'],
            'test_forecast': report['test_forecast'],
            'test_metrics': report['test_metrics'],
        },
    }
    if (report.get('mvmd_params', False)):
        report_data.get('training', report_data.get('retraining', {}))['mvmd_params'] = report['mvmd_params'],
        report_data.get('training', report_data.get('retraining', {}))['mvmd_decomposition'] = report['mvmd_decomposition']

    save_to_json_db(key, report_data, overwrite_key=True)
    return report_data

def save_model_report(model_name, hp, report):
    report_data = {
        'hp': hp.values,
        'training': {
            'train_history': report['train_history'],
            'test_forecast': report['test_forecast'],
            'test_metrics': report['test_metrics'],
        },
    }
    if (report.get('mvmd_params', False)):
        report_data['training']['mvmd_params'] = report['mvmd_params'],
        report_data['training']['mvmd_decomposition'] = report['mvmd_decomposition']

    save_to_json_db(model_name, report_data, overwrite_key=True)
    return report_data

def arima_retrain(model_id, dataset, hp_values=None, tuner=None):
    hp_values = {
        'p': 1,
        'd': 1,
        'q': 1,
    }
    hp = hyperparameters_module.HyperParameters()
    for k, v in hp_values.items():
        hp.Fixed(k, v)

    print(hp.values, '\n')

    arima = ArimaHyperModel(dataset=dataset)
    model = arima.build(hp=hp)

    report = {}
    result = arima.fit(hp=hp, model=model, report=report, report_name_prefix=model_id)

    if tuner is None:
        report_data = save_model_report(model_id, hp, report)
    else:
        report_data = save_tuner_report(model_id, tuner, report)
    return report_data


def lstm_retrain(model_id, dataset, hp_values=None, tuner=None):
    if hp_values is None:
        hp_values = {
            'validation_split': 0.15,
            'lag': 30,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32,
            'units': 32
        }
    hp = hyperparameters_module.HyperParameters()
    for k, v in hp_values.items():
        hp.Fixed(k, v)

    print(hp.values, '\n')

    lstm = LstmHyperModel(dataset=dataset)
    model = lstm.build(hp=hp)
    model.summary()

    report = {}
    result = lstm.fit(hp=hp, model=model, report=report, report_name_prefix=model_id)

    if tuner is None:
        report_data = save_model_report(model_id, hp, report)
    else:
        report_data = save_tuner_report(model_id, tuner, report)
    return report_data


def lstm_mvmd_retrain(model_id, dataset, hp_values=None, tuner=None):
    if hp_values is None:
        hp_values = {
            'K': 3,
            'alpha': 3000,
            'validation_split': 0.15,
            'lag': 30,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32,
            'units': 32
        }   
    hp = hyperparameters_module.HyperParameters()
    for k, v in hp_values.items():
        hp.Fixed(k, v)

    print(hp.values, '\n')

    lstm_mvmd = LstmMvmdHyperModel(dataset=dataset)
    model = lstm_mvmd.build(hp=hp)
    model.summary()

    report = {}
    result = lstm_mvmd.fit(hp=hp, model=model, report=report, report_name_prefix=model_id)

    if tuner is None:
        report_data = save_model_report(model_id, hp, report)
    else:
        report_data = save_tuner_report(model_id, tuner, report)
    return report_data


def lstm_pso_retrain(model_id, project_name, dataset):
    pso_tuner_lstm = PsoParallelTuner(
        hypermodel=LstmHyperModel(dataset=dataset),
        objective=keras_tuner.Objective('mse', direction='min'),
        overwrite=False,
        project_name=project_name,
    )
    best_hp = pso_tuner_lstm.get_best_hyperparameters(1)[0]

    return lstm_retrain(model_id, dataset, hp_values=best_hp.values, tuner=pso_tuner_lstm)

def lstm_mvmd_pso_retrain(model_id, project_name, dataset):
    pso_tuner_lstm_mvmd = PsoParallelTuner(
        hypermodel=LstmMvmdHyperModel(dataset=dataset),
        objective=keras_tuner.Objective('mse', direction='min'),
        overwrite=False,
        project_name=project_name,
    )
    best_hp = pso_tuner_lstm_mvmd.get_best_hyperparameters(1)[0]

    return lstm_mvmd_retrain(model_id, dataset, hp_values=best_hp.values, tuner=pso_tuner_lstm_mvmd)

def retrain_worker(seed):
    try:
        set_all_seeds(seed)
        print(f'\n[{seed}] Started retrain worker with seed', seed)
        initial_stdout = sys.stdout
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull

        models = ['arima', 'base_lstm', 'lstm_mvmd', 'pso_tuner_lstm', 'pso_tuner_lstm_mvmd']
        datasets = ['gspc', 'n225', 'jkse']

        for dataset_code in datasets:
            print(f'\n[{seed}] Processing dataset:', dataset_code)
            dataset = get_stock_historical('^' + dataset_code.upper())

            for model_code in models:
                model_id = f'{model_code}_{dataset_code}_{seed}'
                print(f'\n[{seed}] Retraining model:', model_id)
                if model_code == 'arima':
                    report_data = arima_retrain(model_id, dataset)
                elif model_code == 'base_lstm':
                    report_data = lstm_retrain(model_id, dataset)
                elif model_code == 'lstm_mvmd':
                    report_data = lstm_mvmd_retrain(model_id, dataset)
                elif model_code == 'pso_tuner_lstm':
                    project_name = f'pso_tuner_lstm_{dataset_code}'
                    report_data = lstm_pso_retrain(model_id, project_name, dataset)
                elif model_code == 'pso_tuner_lstm_mvmd':
                    project_name = f'pso_tuner_lstm_mvmd_{dataset_code}'
                    report_data = lstm_mvmd_pso_retrain(model_id, project_name, dataset)
                else:
                    print(f'[{seed}] Unknown model code: {model_code}')
                    continue
                print(f'[{seed}] Retraining completed for model:', model_id)

        sys.stdout = initial_stdout
        devnull.close()
        print(f'\n[{seed}] Worker completed')
        return True

    except Exception as e:
        sys.stdout = initial_stdout
        devnull.close()
        print(f'\n[{seed}] Error occured during execution:', str(e))
        return False
    finally:
        print(f'\n[{seed}] Ended worker with seed', seed)

def main():
    N = 30
    if len(sys.argv) == 2:
        N = int(sys.argv[1])

    seeds = np.arange(N, dtype=np.int64).tolist()

    print('Starting parallel significance testing with N =', N)

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_globals,
            initargs=(lock,)
        ) as pool:
            results = pool.map(retrain_worker, seeds)

    print('Significance testing results:', results)
    print('All worker processes completed')

if __name__ == '__main__':
    main()
