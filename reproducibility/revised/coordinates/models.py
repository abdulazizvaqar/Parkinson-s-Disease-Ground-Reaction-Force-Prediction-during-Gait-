"""TensorFlow model definitions and Bayesian selection over complete inner LOSO.

TensorFlow/KerasTuner are imported only during training. Audit and data tests
therefore work in a lightweight environment without a GPU or TensorFlow.
"""
from __future__ import annotations

import gc
from pathlib import Path
import numpy as np

from protocol import evaluate_candidate, read_json, stable_seed, write_json


def training_dependencies():
    try:
        import tensorflow as tf
        import keras_tuner as kt
    except ImportError as exc:
        raise RuntimeError("Training requires TensorFlow and keras-tuner. Install requirements.txt "
                           "in a Python 3.11 environment; audit mode needs only requirements-audit.txt.") from exc
    return tf, kt


def configure_tensorflow(seed, threads=0):
    tf, _ = training_dependencies()
    if threads:
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return {"tensorflow": tf.__version__, "devices": [d.name for d in tf.config.list_physical_devices()]}


def search_space(architecture, profile):
    _, kt = training_dependencies()
    hp = kt.HyperParameters()
    if profile == "smoke":
        hp.Fixed("num_layers", 1)
        hp.Fixed("activation", "tanh")
        hp.Fixed("optimizer", "adam")
        hp.Fixed("learning_rate", 0.001)
        hp.Fixed("loss", "mae")
        hp.Fixed("encoder_units", 4)
        hp.Fixed("decoder_units", 4)
        hp.Fixed("units_0", 4)
        return hp
    max_layers = 10 if profile == "original" else 3
    hp.Int("num_layers", 1, max_layers)
    # The original GRU code registers tanh first under the shared HP name.
    hp.Choice("activation", ["relu", "tanh"] if architecture == "conv1d_lstm" and profile == "original" else ["tanh"])
    hp.Choice("optimizer", ["adam", "sgd", "rmsprop"] if profile == "original" else ["adam"])
    hp.Choice("learning_rate", [0.0001, 0.001, 0.01] if profile == "original" else [0.0001, 0.001])
    hp.Choice("loss", ["mse", "mae", "huber"] if architecture == "gru" else ["mse", "mae"])
    hp.Fixed("encoder_units", (32 if architecture == "gru" else 100) if profile == "original" else 32)
    if architecture != "gru":
        hp.Choice("decoder_units", list(range(32, 513, 32)) if profile == "original" else [32, 64, 128])
    unit_options = (list(range(32, 513, 32)) if architecture == "gru" else list(range(100, 513, 32)))
    if profile != "original":
        unit_options = [32, 64, 128]
    for layer in range(max_layers):
        with hp.conditional_scope("num_layers", list(range(layer + 1, max_layers + 1))):
            hp.Choice(f"units_{layer}", unit_options)
    return hp


def build_model(architecture, timesteps, parameters, profile="original"):
    tf, _ = training_dependencies()
    layers = tf.keras.layers
    model = tf.keras.Sequential(name=architecture)
    model.add(layers.Input(shape=(timesteps, 1)))
    activation = parameters["activation"]
    if architecture == "conv1d_lstm":
        model.add(layers.Conv1D(4 if profile == "smoke" else 64, 3, activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2))
    if architecture in {"lstm", "conv1d_lstm"}:
        model.add(layers.LSTM(parameters["encoder_units"], activation=activation))
        model.add(layers.RepeatVector(timesteps))
        model.add(layers.LSTM(parameters["decoder_units"], activation=activation, return_sequences=True))
        for layer in range(parameters["num_layers"]):
            model.add(layers.LSTM(parameters[f"units_{layer}"], activation=activation, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(1)))
    elif architecture == "gru":
        model.add(layers.GRU(parameters["encoder_units"], activation=activation, return_sequences=True))
        for layer in range(parameters["num_layers"]):
            model.add(layers.GRU(parameters[f"units_{layer}"], activation=activation, return_sequences=True))
        model.add(layers.Dense(1))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    optimizer = {"adam": tf.keras.optimizers.Adam, "sgd": tf.keras.optimizers.SGD,
                 "rmsprop": tf.keras.optimizers.RMSprop}[parameters["optimizer"]](
                     learning_rate=parameters["learning_rate"])
    loss = tf.keras.losses.Huber() if parameters["loss"] == "huber" else parameters["loss"]
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model


class TensorFlowBackend:
    def __init__(self, architecture, timesteps, profile, epochs, patience, batch_size, verbose):
        self.tf, _ = training_dependencies()
        self.architecture, self.timesteps, self.profile = architecture, timesteps, profile
        self.epochs, self.patience = epochs, patience
        self.batch_size, self.verbose = batch_size, verbose

    def clear(self):
        self.tf.keras.backend.clear_session()
        gc.collect()

    def fresh(self, parameters, seed):
        self.clear()
        self.tf.keras.utils.set_random_seed(seed)
        return build_model(self.architecture, self.timesteps, parameters, self.profile)

    def fit_inner(self, x_train, y_train, x_validation, y_validation, parameters, seed, history_path):
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.fresh(parameters, seed)
        tf = self.tf
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                               patience=self.patience, min_delta=0,
                                               restore_best_weights=True)
        try:
            history = model.fit(x_train, y_train,
                                validation_data=(x_validation, y_validation),
                                epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                                callbacks=[stop, tf.keras.callbacks.TerminateOnNaN(),
                                           tf.keras.callbacks.CSVLogger(str(history_path))],
                                verbose=self.verbose)
            values = np.asarray(history.history["val_mae"], dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("Nonfinite inner training/validation history; trial must fail")
            best_epoch = int(np.argmin(values)) + 1
            prediction = self.predict(model, x_validation)
            return {"prediction": prediction, "best_epoch": best_epoch, "epochs_run": len(values)}
        finally:
            del model
            self.clear()

    def fit_final(self, x_train, y_train, parameters, seed, epochs, history_path):
        """No validation split or validation-driven callback in the outer refit."""
        model = self.fresh(parameters, seed)
        tf = self.tf
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size,
                            shuffle=True, callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                                      tf.keras.callbacks.CSVLogger(str(history_path))],
                            verbose=self.verbose)
        if len(history.history["loss"]) != epochs or not np.isfinite(history.history["loss"]).all():
            raise ValueError("Outer refit did not complete with finite losses")
        return model

    def predict(self, model, inputs):
        return np.asarray(model.predict(inputs, batch_size=self.batch_size, verbose=0))

    def save(self, model, path):
        model.save(path)
        lines = []
        model.summary(print_fn=lambda line, **kwargs: lines.append(line))
        Path(path).with_suffix(".summary.txt").write_text("\n".join(lines), encoding="utf-8")


def select_hyperparameters(ds, outer_subject, architecture, backend, seed, directory,
                           profile, max_trials, scale_input, resume=False):
    """A Bayesian trial is a COMPLETE inner LOSO, never just one validation subject."""
    _, kt = training_dependencies()
    directory = Path(directory)
    trial_root = directory / "candidate_fits"
    trial_root.mkdir(parents=True, exist_ok=True)

    class InnerLOSOTuner(kt.Tuner):
        def run_trial(self, trial, *args, **kwargs):
            parameters = dict(trial.hyperparameters.values)
            trial_dir = trial_root / trial.trial_id
            print(f"  {ds.task}/{architecture}/{outer_subject}: trial {trial.trial_id}", flush=True)
            try:
                summary = evaluate_candidate(ds, outer_subject, parameters, backend, seed,
                                             trial_dir, scale_input)
            except (ValueError, backend.tf.errors.ResourceExhaustedError) as exc:
                write_json(trial_dir / "failure.json", {"error": str(exc), "parameters": parameters})
                # Do not average only the successful validation participants.
                raise kt.errors.FailedTrialError(str(exc)) from exc
            return {"inner_subject_mae": summary["inner_subject_mae"]}

    hp = search_space(architecture, profile)
    oracle = kt.oracles.BayesianOptimizationOracle(
        objective=kt.Objective("inner_subject_mae", direction="min"),
        max_trials=max_trials, num_initial_points=min(3, max_trials),
        hyperparameters=hp, allow_new_entries=False, tune_new_entries=False,
        seed=stable_seed(seed, ds.task, architecture, outer_subject, "oracle"),
        max_retries_per_trial=0, max_consecutive_failed_trials=3)
    tuner = InnerLOSOTuner(oracle=oracle, hypermodel=None, directory=str(directory),
                          project_name="tuner", overwrite=not resume)
    tuner.search()
    best = tuner.oracle.get_best_trials(num_trials=1)
    if not best or best[0].score is None or not np.isfinite(best[0].score):
        raise RuntimeError("No successful full inner LOSO trial; inspect failure.json files")
    trial = best[0]
    summary = read_json(trial_root / trial.trial_id / "inner_summary.json")
    write_json(directory / "selection.json", {
        "trial_id": trial.trial_id, "inner_subject_mae": summary["inner_subject_mae"],
        "parameters": summary["parameters"], "search_profile": profile,
        "n_inner_folds": len(summary["folds"])})
    return summary["parameters"], summary
