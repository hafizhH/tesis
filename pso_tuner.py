
from IPython.display import display
import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError as MeanSquaredErrorLoss
from tensorflow.keras.metrics import R2Score, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error
import keras_tuner
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module
from keras_tuner.engine import hyperparameters as hyperparameters_module
from keras_tuner import synchronized
import multiprocessing
from multiprocessing import shared_memory
import sys
import random
import time
import pprint

def set_all_seeds(seed_value=None):
    if seed_value is None:
        np.random.seed()
        random.seed()
        tf.random.set_seed(None)
        torch.manual_seed(torch.seed())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch.seed())
    else:
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)    

import yfinance as yf

def get_stock_market_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    historical_data = stock.history(start=start_date, end=end_date)

    print("Dataset Summary")
    print(f"Name: {info.get('longName', 'N/A')}")
    print(f"Ticker: {ticker}")
    print(f"Date Range: {historical_data.index.min().date()} to {historical_data.index.max().date()}")
    print(f"Number of Data Points: {len(historical_data)}")
    print(f"Columns: {list(historical_data.columns)}")

    return {
        "info": info,
        "historical_data": historical_data,
    }

test_split = 0.15
validation_split = 0.15

class PsoOracle(oracle_module.Oracle):
    def __init__(
        self,
        hyperparameters,
        objective=None,
        population_size=10,
        inertia=0.9,
        cognitive=3.0,
        social=1.5,
        max_iter=50,
        v_clamp_factor=0.4,
        **kwargs
    ):
        super().__init__(
            objective=objective,
            hyperparameters=hyperparameters,
            **kwargs
        )

        # PSO parameters
        self.N = population_size
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        self.iter = -1  # -1 for uninitialized swarm
        self.max_iter = max_iter
        self.k = v_clamp_factor

        # PSO swarm state
        self.particles = []
        self.global_best = {
            'position': {},
            'score': np.inf
        }
        self.col_width = 32

    @synchronized
    def initialize_swarm(self):
        hps = self.get_space()
        for particle in range(self.N):
            position = {}
            velocity = {}

            for hp in hps.space:
                position[hp.name] = hp.random_sample()

                if isinstance(hp, keras_tuner.engine.hyperparameters.Float):
                    vmax = self.k * (hp.max_value - hp.min_value) / 2
                    velocity[hp.name] = random.uniform(-vmax, vmax)
                elif isinstance(hp, keras_tuner.engine.hyperparameters.Int):
                    vmax = int(self.k * (hp.max_value - hp.min_value) / 2)
                    velocity[hp.name] = random.randint(-vmax, vmax)
                elif isinstance(hp, keras_tuner.engine.hyperparameters.Choice):
                    velocity[hp.name] = random.choice(hp.values)
                elif isinstance(hp, keras_tuner.engine.hyperparameters.Fixed):
                    velocity[hp.name] = 0
                else:
                    raise ValueError(f"Unknown hyperparameter type: {type(hp)}")

            self.particles.append({
                'iter': 0,
                'position': position,
                'velocity': velocity,
                'best_position': position,
                'best_score': np.inf
            })

        self.global_best = {
            'position': {},
            'score': np.inf
        }

        print('PSO swarm initialized', self.particles[0])

    @synchronized
    def _update_swarm(self):
        def _sign(x):
            return 1 if x > 0 else -1 if x < 0 else 0

        def _clamp(x, min_val, max_val):
            return max(min_val, min(x, max_val))

        # Update global best
        for particle in self.particles:
            if particle['best_score'] < self.global_best['score']:
                self.global_best = {
                    'position': particle['best_position'].copy(),
                    'score': particle['best_score']
                }

        # Update particle positions & velocities
        hps = self.get_space()
        for particle in self.particles:
            velocity = particle['velocity']
            for hp in hps.space:
                dimension_name = hp.name
                new_velocity_dimension = self.w * velocity[dimension_name]
                + self.c1 * random.random() * (
                    particle['best_position'][dimension_name] - particle['position'][dimension_name]
                )
                + self.c2 * random.random() * (
                    self.global_best['position'][dimension_name] - particle['position'][dimension_name]
                )

                velocity[dimension_name] = _sign(new_velocity_dimension) * \
                max(abs(new_velocity_dimension), 0.01)

                new_position_dimension = particle['position'][dimension_name] + velocity[dimension_name]

                if isinstance(hp, keras_tuner.engine.hyperparameters.Float):
                    particle['position'][dimension_name] = _clamp(
                        new_position_dimension, hp.min_value, hp.max_value
                    )
                elif isinstance(hp, keras_tuner.engine.hyperparameters.Int):
                    particle['position'][dimension_name] = int(round(_clamp(
                        new_position_dimension, hp.min_value, hp.max_value
                    )))
                elif isinstance(hp, keras_tuner.engine.hyperparameters.Fixed):
                    pass
                else:
                    raise ValueError(f"Unhandled hyperparameter type: {type(hp)}") 
        print('PSO swarm updated with global best:', self.global_best)

    def populate_space(self, trial_id):
        trial_index = int(trial_id)

        # Check for iteration should be incremented
        increment = True
        for particle in self.particles:
            if particle['iter'] <= self.iter:
                increment = False
                break
        if increment:
            self.iter += 1
            print('PSO iter incremented:', self.iter)

            if self.iter == 0:
                self.initialize_swarm()
            elif self.iter >= self.max_iter:
                return {
                    'status': trial_module.TrialStatus.STOPPED,
                    'values': None
                }
            else:
                self._update_swarm()

        trial_iter = int(trial_index / self.N)    
        particle_index = trial_index % self.N
        particle = self.particles[particle_index]

        # Check if trial is beyond current iteration
        if (trial_iter > self.iter):
            return {
                'status': trial_module.TrialStatus.IDLE,
                'values': None
            }

        hp = hyperparameters_module.HyperParameters()
        for name, value in particle['position'].items():
            hp.Fixed(name, value)

        return {
            'status': trial_module.TrialStatus.RUNNING,
            'values': hp.values
        }

    def update_trial(self, trial_id, metrics, step=0):
        # Called after each trial finishes
        score = metrics[self.objective.name]
        particle = self.particles[int(trial_id) % self.N]
        particle["iter"] += 1

        # Update personal best
        if (self.objective.direction == "min" and score < particle["best_score"]) or (
            self.objective.direction == "max" and score > particle["best_score"]
        ):
            particle["best_score"] = score
            particle["best_position"] = particle["position"].copy()

        trial = super().update_trial(trial_id, metrics, step)
        return trial

    def get_all_trials(self):
        return self.trials

    def get_state(self):
        state = super().get_state()
        state['pso'] = {
            'params': {
                'N': self.N,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'max_iter': self.max_iter,
            },
            'utils': {
                'col_width': self.col_width,
            },
            'state': {
                'iter': self.iter,
                'global_best': self.global_best,
                'particles': self.particles,
            },
        }
        return state

    def set_state(self, state):
        super().set_state(state)
        pso_state = state.get('pso', {})
        pso_params = pso_state.get('params', {})
        self.N = pso_params.get('N', self.N)
        self.w = pso_params.get('w', self.w)
        self.c1 = pso_params.get('c1', self.c1)
        self.c2 = pso_params.get('c2', self.c2)
        self.max_iter = pso_params.get('max_iter', self.max_iter)

        pso_utils = pso_state.get('utils', {})
        self.col_width = pso_utils.get('col_width', self.col_width)

        pso_swarm_state = pso_state.get('state', {})
        self.iter = pso_swarm_state.get('iter', self.iter)
        self.global_best = pso_swarm_state.get('global_best', self.global_best)
        self.particles = pso_swarm_state.get('particles', self.particles)

    def score_trial(self, trial):
        super().score_trial(trial)
        self.show_metrics_table(trial)

    def show_metrics_table(self, trial):
        template = "{{0:{0}}}|{{1:{0}}}|{{2}}".format(self.col_width)
        best_trials = self.get_best_trials()
        best_trial = best_trials[0] if len(best_trials) > 0 else None
        if trial.metrics:
            print(f"\nResult metrics for Trial #{str(int(trial.trial_id) + 1)}\n")
            print(
                template.format(
                    "Value",
                    f"Best Value So Far (Trial #{str(int(best_trial.trial_id) + 1) if best_trial else '?'})",
                    "Metrics",
                )
            )
            for metric_name, metric in trial.metrics.metrics.items():
                best_value = (
                    best_trial.metrics.metrics[metric_name].get_best_value()
                    if best_trial
                    else "?"
                )
                value = metric.get_best_value()
                print(
                    template.format(
                        self.format_value(value),
                        self.format_value(best_value),
                        metric_name,
                    )
                )
        else:
            print("Trial metrics not available")


    def format_value(self, val):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return f"{val:.5g}"
        val_str = str(val)
        if len(val_str) > self.col_width:
            val_str = f"{val_str[:self.col_width - 3]}..."
        return val_str

    def format_duration(self, d):
        s = round(d.total_seconds())
        d = s // 86400
        s %= 86400
        h = s // 3600
        s %= 3600
        m = s // 60
        s %= 60

        if d > 0:
            return f"{d:d}d {h:02d}h {m:02d}m {s:02d}s"
        return f"{h:02d}h {m:02d}m {s:02d}s"

class PsoParallelTuner(tuner_module.Tuner):

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        self.seed = seed
        oracle = PsoOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle, hypermodel, **kwargs)

def create_dataset(
    input_dataset,
    lag=1,
    test_split=0.15,
    target_index=3,
    feat_include_target=True,
    normalize=True,
):
    def normalize_features(X_train, X_test):
        # One scaler per feature column
        scalers = []
        for col_index in range(X_train.shape[2]):
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Only fit on train to avoid data leakage
            X_train[:, :, col_index] = scaler.fit_transform(X_train[:, :, col_index])
            X_test[:, :, col_index] = scaler.transform(X_test[:, :, col_index])
            scalers.append(scaler)
        return X_train, X_test, scalers

    def normalize_target(y_train, y_test):
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Only fit on train to avoid data leakage
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = scaler.transform(y_test.reshape(-1, 1))
        return y_train, y_test, scaler

    selected_feat_cols = [
        j for j in range(len(input_dataset.columns))
        if (feat_include_target or j != target_index)
    ]
    dataset = input_dataset.iloc[:, selected_feat_cols].copy()
    target = input_dataset.iloc[:, target_index].copy()

    X, y = [], []
    for i in range(len(dataset) - lag):
        X.append(dataset.iloc[i : (i + lag)])
        y.append(target.iloc[i + lag])

    # Split into train and test
    num_train_samples = int((1 - test_split) * (len(dataset) - lag))
    X_train = X[:num_train_samples]
    y_train = y[:num_train_samples]
    X_test = X[num_train_samples:]
    y_test = y[num_train_samples:]

    if normalize:
        # Normalize features and target separately
        X_train, X_test, feature_scalers = normalize_features(
            np.array(X_train), np.array(X_test)
        )
        y_train, y_test, target_scaler = normalize_target(
            np.array(y_train), np.array(y_test)
        )
        return (
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test),
            feature_scalers,
            target_scaler,
        )
    else:
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def evaluate_model(
    model=None,
    X_test=None,
    y_pred=None,
    y_test=None,
    scaler=None,
    metrics=[],
    report=None,
):
    if model is not None and X_test is not None and y_pred is None:
        y_pred = model.predict(X_test)

    if scaler is not None:
        rescaled_y_pred = scaler.inverse_transform(y_pred)
        rescaled_y_test = scaler.inverse_transform(y_test)
    else:
        rescaled_y_pred = np.array(y_pred)
        rescaled_y_test = np.array(y_test)

    result = {}
    if len(metrics) == 0 or "r2" in metrics:
        r2 = r2_score(rescaled_y_test, rescaled_y_pred)
        result["r2"] = r2
    if len(metrics) == 0 or "mse" in metrics:
        mse = mean_squared_error(rescaled_y_test, rescaled_y_pred)
        result["mse"] = mse
    if len(metrics) == 0 or "rmse" in metrics:
        rmse = root_mean_squared_error(rescaled_y_test, rescaled_y_pred)
        result["rmse"] = rmse
    if len(metrics) == 0 or "mape" in metrics:
        mape = mean_absolute_percentage_error(rescaled_y_test, rescaled_y_pred)
        result["mape"] = mape
    if len(metrics) == 0 or "mae" in metrics:
        mae = mean_absolute_error(rescaled_y_test, rescaled_y_pred)
        result["mae"] = mae
    if len(metrics) == 0 or "da" in metrics:
        correct_directional_change = np.sign(
            rescaled_y_pred[1:] - rescaled_y_test[:-1]
        ) == np.sign(rescaled_y_test[1:] - rescaled_y_test[:-1])
        da = float(np.mean(correct_directional_change))
        result["da"] = da

    if report is not None:
        report["test_metrics"] = result
        report["test_forecast"] = {
            "y_test": rescaled_y_test.tolist(),
            "y_pred": rescaled_y_pred.tolist(),
        }
        # Plotting
        plt.figure(figsize=(25, 8))
        plt.plot(rescaled_y_test, label="True Data")
        plt.plot(rescaled_y_pred, label="Predictions")
        plt.legend()
        plt.show()

        for key, value in result.items():
            print(f"{key.upper()}: {value}")

    return result

def mvmd(signal, alpha, tau, K, DC, init, tol, max_N):
    # ---------------------

    # Period and sampling frequency of input signal
    C, T = signal.shape # T:length of signal C:  channel number
    fs = 1 / float(T)
    # extend the signal by mirroring
    f_mirror = torch.zeros(C, 2*T)
    f_mirror[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1])
    f_mirror[:,T//2:3*T//2] = signal
    f_mirror[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
    f = f_mirror
    # Time Domain 0 to T (of mirrored signal)
    T = float(f.shape[1])
    t = torch.linspace(1/float(T), 1, int(T))
    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = max_N
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * torch.ones(K, dtype=torch.cfloat)
    # Construct and center f_hat
    f_hat = torch.fft.fftshift(torch.fft.fft(f))
    f_hat_plus = f_hat
    f_hat_plus[:, 0:int(int(T)/2)] = 0
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = torch.zeros((N, len(freqs), K, C), dtype=torch.cfloat)

    # Initialization of omega_k
    omega_plus = torch.zeros((N, K), dtype=torch.cfloat)
    if (init == 1):
        for i in range(1, K+1):
            omega_plus[0,i-1] = (0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:] = torch.sort(torch.exp(torch.log(fs)) +
        (torch.log(0.5) - torch.log(fs)) * torch.random.rand(1, K))
    else:
        omega_plus[0,:] = 0
    if (DC):
        omega_plus[0,0] = 0

    # start with empty dual variables
    lamda_hat = torch.zeros((N, len(freqs), C), dtype=torch.cfloat)
    # other inits
    uDiff = tol+2.2204e-16 #updata step
    n = 1 #loop counter
    sum_uk = torch.zeros((len(freqs), C)) #accumulator
    T = int(T)

    # ----------- Main loop for iterative updates

    while uDiff > tol and n < N:
        # update first mode accumulator
        k = 1
        sum_uk = u_hat_plus[n-1,:,K-1,:] + sum_uk - u_hat_plus[n-1,:,0,:]

        #update spectrum of first mode through Wiener filter of residuals
        for c in range(C):
            u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] -
            lamda_hat[n-1,:,c]/2) \
        / (1 + Alpha[k-1] * torch.square(freqs - omega_plus[n-1,k-1]))

        #update first omega if not held at 0
        if DC == False:
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0),
                            torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
            / torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))

        for k in range(2, K+1):

            #accumulator
            sum_uk = u_hat_plus[n,:,k-2,:] + sum_uk - u_hat_plus[n-1,:,k-1,:]

            #mode spectrum
            for c in range(C):
                u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] -
            lamda_hat[n-1,:,c]/2) \
            / (1 + Alpha[k-1] * torch.square(freqs-omega_plus[n-1,k-1]))

            #center frequencies
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0),
                torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
                /  torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T:,k-1,:])))

        #Dual ascent
        lamda_hat[n,:,:] = lamda_hat[n-1,:,:] # + tau * (torch.sum(u_hat_plus[n,:,:,:], dim=1)
                       #  - f_hat_plus)

        #loop counter
        n = n + 1

        #converged yet? (optimized version proposed)
        uDiff = 2.2204e-16
        for i in range(1, K+1):
            diff = u_hat_plus[n-1, :, i-1, :] - u_hat_plus[n-2, :, i-1, :]
            norm_sq = torch.sum(torch.abs(diff)**2) / T
            uDiff += norm_sq
        uDiff = torch.real(uDiff)

    # ------ Postprocessing and cleanup

    # discard empty space if converged early
    N = min(N, n)
    omega = omega_plus[0:N,:]

    # Signal reconstruction
    u_hat = torch.zeros((T,K,C), dtype=torch.cfloat)
    for c in range(C):
        u_hat[T//2:T,:,c] = torch.squeeze(u_hat_plus[N-1,T//2:T,:,c])
        second_index = list(range(1,T//2+1))
        second_index.reverse()
        u_hat[second_index,:,c] = torch.squeeze(torch.conj(u_hat_plus[N-1,T//2:T,:,c]))
        u_hat[0,:,c] = torch.conj(u_hat[-1,:,c])
    u = torch.zeros((K,len(t),C), dtype=torch.cfloat)

    for k in range(1, K+1):
        for c in range(C):
            u[k-1,:,c]  = (torch.fft.ifft(torch.fft.ifftshift(u_hat[:,k-1,c]))).real

    # remove mirror part
    u = u[:,T//4:3*T//4,:]

    #recompute spectrum
    u_hat = torch.zeros((T//2,K,C), dtype=torch.cfloat)

    for k in range(1, K+1):
        for c in range(C):
            u_hat[:,k-1,c] = torch.fft.fftshift(torch.fft.fft(u[k-1,:,c])).conj()

    # ifftshift
    u = torch.fft.ifftshift(u, dim=-1)

    return (u.real, u_hat, omega)

def mvmd_decompose(input_dataset, K, alpha, report=None):
    dataset = input_dataset.copy()
    col_names = dataset.columns

    feature_torches = []
    for col in col_names:
        feature_torches.append(torch.from_numpy(dataset[col].to_numpy()))
    feature_stack = torch.stack(feature_torches,dim=0)

    params = {
        'K': K if K is not None else 5,
        'alpha': alpha if alpha is not None else 3000,
        'tau': 0,
        'DC': False,
        'init': 1,
        'tol': 1e-7,
        'n_iter': 500
    }

    [u, u_hat, omega] = mvmd(
        feature_stack,
        params['alpha'],
        params['tau'],
        params['K'],
        params['DC'],
        params['init'],
        params['tol'],
        params['n_iter']
    )

    if report is not None:
        report['mvmd_params'] = params
        report['mvmd_decomposition'] = {
            'original': feature_stack.numpy().tolist(),
            'decomposed': u.numpy().tolist(),
        }

    imf_dataset = pd.DataFrame()
    for col_index in range(len(col_names)):
        for mode_index in range(u.shape[0]):
            imf_dataset[
                col_names[col_index] + '_IMF' + str(mode_index)
            ] = u[mode_index, :, col_index].numpy()

    return imf_dataset

class LstmHyperModel(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True, dataset=None):
        self.dataset = dataset
        super().__init__(name, tunable)

    def build(self, hp):
        hp.Int("epochs", min_value=20, max_value=50)
        hp.Int("batch_size", min_value=8, max_value=64, step=4)
        hp.Fixed("validation_split", 0.15)
        hp.Fixed("test_split", 0.15)
        lag = hp.Fixed("lag", 30)
        num_features = self.dataset.shape[1]
        units = hp.Int("units", min_value=16, max_value=96)
        learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.01)

        model = Sequential(
            [InputLayer(shape=(int(lag), num_features)), LSTM(units=units), Dense(1)]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=MeanSquaredErrorLoss(),
            metrics=[
                R2Score(), MeanSquaredError(), RootMeanSquaredError(),
                MeanAbsolutePercentageError(), MeanAbsoluteError()
            ]
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        report = kwargs.pop("report", None)
        epochs = hp.get("epochs")
        batch_size = hp.get("batch_size")
        validation_split = hp.get("validation_split")
        test_split = hp.get("test_split")
        lag = hp.get("lag")

        ohlcv_df = self.dataset
        target_index = ohlcv_df.columns.get_loc("Close")
        X_train, y_train, X_test, y_test, feature_scalers, target_scaler = create_dataset(
            ohlcv_df, lag, test_split, target_index=target_index, feat_include_target=True
        )

        history = model.fit(
            *args,
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            validation_split=validation_split,
            **kwargs,
        )
        if report is not None:
            report["train_history"] = {
                "params": history.params, "history": history.history, "epoch": history.epoch
            }
        result = evaluate_model(
            model=model, X_test=X_test, y_test=y_test, scaler=target_scaler, report=report
        )
        return result

class LstmMvmdHyperModel(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True, dataset=None):
        self.dataset = dataset
        super().__init__(name, tunable)

    def build(self, hp):
        K = hp.Int("K", min_value=3, max_value=6)
        hp.Int("alpha", min_value=1000, max_value=5000)
        hp.Int("epochs", min_value=20, max_value=50)
        hp.Int("batch_size", min_value=8, max_value=64, step=4)
        hp.Fixed("validation_split", 0.15)
        hp.Fixed("test_split", 0.15)
        lag = hp.Fixed("lag", 30)
        units = hp.Int("units", min_value=16, max_value=96)
        learning_rate = hp.Float("learning_rate", min_value=0.001, max_value=0.01)
        for i in range(self.dataset.shape[1] * 6):
            hp.Float(f"w{i}", min_value=0.0, max_value=1.0)

        num_features = self.dataset.shape[1] * K + 1
        model = Sequential(
            [InputLayer(shape=(int(lag), int(num_features))), LSTM(units=units), Dense(1)]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=MeanSquaredErrorLoss(),
            metrics=[
                R2Score(), MeanSquaredError(), RootMeanSquaredError(),
                MeanAbsolutePercentageError(), MeanAbsoluteError(),
            ],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        report = kwargs.pop("report", None)
        K = hp.get("K")
        alpha = hp.get("alpha")
        epochs = hp.get("epochs")
        batch_size = hp.get("batch_size")
        validation_split = hp.get("validation_split")
        test_split = hp.get("test_split")
        lag = hp.get("lag")

        ohlcv_df = self.dataset
        close_df = ohlcv_df[["Close"]]
        imf_df = mvmd_decompose(ohlcv_df, K, alpha, report=report)

        num_modes = int(self.dataset.shape[1] * K)
        weight_matrix = []
        for i in range(num_modes):
            weight_matrix.append(hp.get(f"w{i}"))
        weight_matrix = np.array(weight_matrix)
        imf_df = imf_df * weight_matrix

        combined_df = pd.concat([imf_df, close_df], axis=1)
        target_index = combined_df.columns.get_loc("Close")
        X_train, y_train, X_test, y_test, feature_scalers, target_scaler = create_dataset(
            combined_df, lag, test_split, target_index=target_index, feat_include_target=True
        )

        history = model.fit(
            *args, x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
            shuffle=False, validation_split=validation_split, **kwargs
        )
        if report is not None:
            report["train_history"] = {
                "params": history.params, "history": history.history, "epoch": history.epoch
            }
        result = evaluate_model(
            model=model, X_test=X_test, y_test=y_test, scaler=target_scaler, report=report
        )
        return result

class ArimaHyperModel(keras_tuner.HyperModel):
    def __init__(self, name=None, tunable=True, dataset=None):
        self.dataset = dataset
        super().__init__(name, tunable)

    def build(self, hp):
        p = hp.Fixed("p", 1)
        d = hp.Fixed("d", 1)
        q = hp.Fixed("q", 1)
        hp.Fixed("test_split", 0.15)

        model = ARIMA([], order=(p, d, q))
        return model

    def fit(self, hp, model, *args, **kwargs):
        report = kwargs.pop('report', None)
        p = hp.get("p")
        d = hp.get("d")
        q = hp.get("q")
        test_split = hp.get("test_split")

        close_df = self.dataset[['Close']]
        target_index = close_df.columns.get_loc('Close')

        X_train, y_train, X_test, y_test = create_dataset(
            close_df,
            1,
            test_split,
            target_index=target_index,
            feat_include_target=True,
            normalize=False,
        )
        X_train = X_train.flatten()
        X_test = X_test.flatten()

        X_history = [x for x in X_train]
        y_pred = []
        for t in range(len(X_test)):
            model = ARIMA(X_history, order=(p, d, q))

            model_fit = model.fit()

            yhat = model_fit.forecast()[0]
            y_pred.append(yhat)

            obs = X_test[t]
            X_history.append(obs)
            X_history.pop(0) 

        if report is not None:
            report['train_history'] = {
                'params': model.model_orders,
            }

        result = evaluate_model(y_pred=y_pred, y_test=y_test, report=report)
        return result

def tuner_worker(tuner_id, np_shared_name, np_shared_shape):
    os.environ['KERASTUNER_TUNER_ID'] = tuner_id
    os.environ['KERASTUNER_ORACLE_IP'] = '127.0.0.1'
    os.environ['KERASTUNER_ORACLE_PORT'] = '8000'

    with open(f"./outputs/{tuner_id}_output.txt", "w") as f:
        try:
            initial_stdout = sys.stdout
            sys.stdout = f
            print(f'\n[{tuner_id}]', 'Started tuner worker with pid', os.getpid())
            project_name = os.getenv('PROJECT_NAME')
            print(f'\n[{tuner_id}]', 'Using project name:', project_name)

            shm = shared_memory.SharedMemory(name=np_shared_name)
            restored_np = np.ndarray(shape=np_shared_shape, dtype=np.float64, buffer=shm.buf)
            dataset = pd.DataFrame(restored_np, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

            set_all_seeds(42)
            if (project_name.startswith('pso_tuner_lstm_mvmd')):
                print(f'\n[{tuner_id}]', 'Initializing LSTM-MVMD-PSO Tuner:', project_name)
                tuner = PsoParallelTuner(
                    hypermodel=LstmMvmdHyperModel(dataset=dataset),
                    objective=keras_tuner.Objective('mse', direction='min'),
                    max_trials=500,
                    overwrite=False,
                    project_name=project_name,
                )
            elif (project_name.startswith('pso_tuner_lstm')):
                print(f'\n[{tuner_id}]', 'Initializing LSTM-PSO Tuner:', project_name)
                tuner = PsoParallelTuner(
                    hypermodel=LstmHyperModel(dataset=dataset),
                    objective=keras_tuner.Objective('mse', direction='min'),
                    max_trials=500,
                    overwrite=False,
                    project_name=project_name,
                )

            if tuner_id != 'chief':
                devnull = open(os.devnull, 'w')
                sys.stdout = devnull

            tuner.search_space_summary(extended=True)
            tuner.search()
            tuner.results_summary()

            set_all_seeds(None)

            sys.stdout = f
            print(f'\n[{tuner_id}]', 'Tuning completed')

        except Exception as e:
            sys.stdout = f
            print(f'\n[{tuner_id}]', 'Error occured during execution:', str(e))

        finally:
            print(f'\n[{tuner_id}]', 'Ended tuner worker with pid', os.getpid())
            if devnull:
                devnull.close()
            sys.stdout = initial_stdout
            shm.close()

def main():
    default_ticker_symbol = "^JKSE"
    default_project_name = "pso_tuner_lstm_jkse"
    if len(sys.argv) == 2:
        os.environ["PROJECT_NAME"] = sys.argv[1]
        ticker_symbol = default_ticker_symbol
    elif len(sys.argv) == 3:
        os.environ["PROJECT_NAME"] = sys.argv[1]
        ticker_symbol = sys.argv[2]
    else:
        os.environ["PROJECT_NAME"] = default_project_name
        ticker_symbol = default_ticker_symbol

    print("Preparing dataset", ticker_symbol)
    stock_data = get_stock_market_data(ticker_symbol)
    stock_historical = (
        stock_data["historical_data"]
        .reset_index()
        .drop(["Date", "Dividends", "Stock Splits"], axis=1)
        .to_numpy()
    )

    shm = shared_memory.SharedMemory(
        create=True,
        size=int(np.dtype(np.float64).itemsize * int(np.prod(stock_historical.shape))),
        name="npshared",
    )
    dst = np.ndarray(shape=stock_historical.shape, dtype=np.float64, buffer=shm.buf)
    dst[:] = stock_historical[:]

    print(
        "Starting parallel hyperparameter tuning for project",
        os.getenv("PROJECT_NAME", default_project_name),
    )
    processes = []
    for i in range(multiprocessing.cpu_count() + 1):
        print("Starting process", str(i))
        tuner_id = "chief" if i == 0 else "tuner" + str(i - 1)

        p = multiprocessing.Process(
            target=tuner_worker,
            args=(tuner_id, "npshared", dst.shape),
            daemon=tuner_id == "chief" and False,
        )
        processes.append(p)
        p.start()

        if tuner_id == "chief":
            time.sleep(5)
        else:
            time.sleep(1)

    for i in range(len(processes)):
        processes[i].join()
        print("Process", str(i), "joined")
    print("All tuner processes completed")

if __name__ == "__main__":
    main()
