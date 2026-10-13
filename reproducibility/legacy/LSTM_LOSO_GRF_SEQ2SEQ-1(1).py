#!/usr/bin/env python
# coding: utf-8

#%% Imports
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

import keras
import keras_tuner
from keras_tuner.tuners import BayesianOptimization

from math import sqrt

#%% GPU / Strategy
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
strategy = tf.distribute.MirroredStrategy()

# ========== CONFIG ==========
BASE_INPUT  = Path('OFF/INPUT/INPUT-ANKLE')   # has many subfolders like "AP LEFT ANK OFF"
BASE_OUTPUT = Path('OFF/OUTPUT')              # has subfolders like "AP LEFT OFF"
VALID_SPLITS = ['train', 'test']              # required splits
UB_PATTERN = re.compile(r'UB(\d{2})', re.IGNORECASE)  # detects UBxx tokens
# ============================================

def extract_ub_token(filename: str):
    """Return UBxx (uppercase) if found, else None."""
    m = UB_PATTERN.search(filename)
    return m.group(0).upper() if m else None

def list_excel_files(folder: Path):
    """Return list of Excel files in folder (xlsx, xls)."""
    if not folder.exists():
        return []
    return sorted([*folder.glob('*.xlsx'), *folder.glob('*.xls')])

def build_save_dirs(input_subdir: Path):
    """Create per-subfolder output dirs inside the input subfolder."""
    out = {
        'Models':         input_subdir / 'Models',
        'Figures':        input_subdir / 'Figures',
        'Stats':          input_subdir / 'Stats',
        'Hyperparameters':input_subdir / 'Hyperparameters',
        'Summary':        input_subdir / 'Summary',
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out

def read_timesheet(p: Path):
    """Uniform Excel read: no header, skip first row as in original code."""
    return pd.read_excel(p, header=None, skiprows=1)

def clean_align(input_train, input_test, output_train, output_test):
    """Mimic the original cleaning: drop columns with >=100 NaNs (by output), reindex to input_train length, fillna(0), then transpose."""
    threshold = 100

    # drop by train, mirror on input
    drop_cols = [col for col in output_train.columns if output_train[col].isnull().sum() >= threshold]
    output_train = output_train.drop(columns=drop_cols)
    input_train = input_train.drop(columns=drop_cols)

    # drop by test, mirror on input
    drop_cols_test = [col for col in output_test.columns if output_test[col].isnull().sum() >= threshold]
    output_test = output_test.drop(columns=drop_cols_test)
    input_test = input_test.drop(columns=drop_cols_test)

    # enforce same length = len(input_train)
    length = len(input_train)
    input_train  = input_train.reindex(range(length)).fillna(0)
    input_test   = input_test.reindex(range(length)).fillna(0)
    output_train = output_train.reindex(range(length)).fillna(0)
    output_test  = output_test.reindex(range(length)).fillna(0)

    # transpose: samples = columns, timesteps = rows
    return (input_train.T, input_test.T, output_train.T, output_test.T, length)

def to_seq3d(df):
    """(samples, timesteps, 1)"""
    samples, timesteps = df.shape[0], df.shape[1]
    return df.values.reshape((samples, timesteps, 1)), samples, timesteps

def build_model_builder(timesteps_in):
    """Return a keras-tuner build function that constructs the seq2seq model under strategy.scope()."""
    def build_model(hp):
        with strategy.scope():
            model = Sequential()
            # Encoder
            model.add(LSTM(100, activation=hp.Choice('activation', ['tanh']),
                           input_shape=(timesteps_in, 1)))
            model.add(RepeatVector(timesteps_in))
            # Decoder stack
            model.add(layers.LSTM(units=hp.Int('input_units', 32, 512, step=32),
                                  activation=hp.Choice('activation', ['tanh']),
                                  return_sequences=True))
            for l in range(hp.Int('num_layers', 1, 10)):
                model.add(layers.LSTM(units=hp.Int(f'units_{l}', 100, 512, step=32),
                                      activation=hp.Choice('activation', ['tanh']),
                                      return_sequences=True))
            model.add(TimeDistributed(Dense(1)))

            opt_name = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
            lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            if opt_name == 'adam':
                opt = keras.optimizers.Adam(learning_rate=lr)
            elif opt_name == 'sgd':
                opt = keras.optimizers.SGD(learning_rate=lr)
            else:
                opt = keras.optimizers.RMSprop(learning_rate=lr)

            loss = hp.Choice('loss', ['mse', 'mae'])
            model.compile(optimizer=opt, loss=loss, metrics=['mae'])
        return model
    return build_model

# ===== Main pairing & training loop =====
for input_subdir in sorted([p for p in BASE_INPUT.iterdir() if p.is_dir()]):
    sub_name = input_subdir.name
    prefix_in = sub_name[:7]  # first 7 chars of input folder

    # try to find matching output subfolder by first 7 chars
    output_match = None
    for cand in sorted([c for c in BASE_OUTPUT.iterdir() if c.is_dir()]):
        if cand.name[:7] == prefix_in:
            output_match = cand
            break

    if output_match is None:
        print(f"[SKIP] No output folder matching prefix '{prefix_in}' for input '{sub_name}'.")
        continue

    output_subdir = output_match
    print(f"\n=== Processing pair ===\nInput:  {input_subdir}\nOutput: {output_subdir}\n")

    # Create per-subfolder save dirs (inside the input subfolder)
    SAVE = build_save_dirs(input_subdir)

    # Validate splits and build UB maps per split
    split_files = {}
    for split in VALID_SPLITS:
        in_split = input_subdir / split
        out_split = output_subdir / split
        if not in_split.exists() or not out_split.exists():
            print(f"[WARN] Missing split folder(s): {in_split} or {out_split}. Skipping split.")
            continue

        in_files  = list_excel_files(in_split)
        out_files = list_excel_files(out_split)

        # Map UBxx ➜ file path (if present)
        in_map = {}
        for f in in_files:
            tok = extract_ub_token(f.name)
            if tok:
                in_map[tok] = f

        out_map = {}
        for f in out_files:
            tok = extract_ub_token(f.name)
            if tok:
                out_map[tok] = f

        # Intersection UB tokens present in both input and output for this split
        common = sorted(set(in_map.keys()) & set(out_map.keys()))
        if not common:
            print(f"[WARN] No matching UBxx tokens in split '{split}' for {input_subdir.name}.")
            continue

        split_files[split] = {'in_map': in_map, 'out_map': out_map, 'common': common}

    # Need both train and test available
    if not all(s in split_files for s in VALID_SPLITS):
        print(f"[WARN] Missing train/test matching for '{input_subdir.name}'. Skipping this subfolder.")
        continue

    # Iterate through UB tokens present in BOTH train and test
    ub_common_both = sorted(set(split_files['train']['common']) & set(split_files['test']['common']))
    if not ub_common_both:
        print(f"[WARN] No UBxx files found in BOTH train and test for '{input_subdir.name}'.")
        continue

    for ub_tok in ub_common_both:
        print(f"\n--- UB pair: {ub_tok} ---")
        # Resolve file paths
        in_train_fp  = split_files['train']['in_map'][ub_tok]
        out_train_fp = split_files['train']['out_map'][ub_tok]
        in_test_fp   = split_files['test']['in_map'][ub_tok]
        out_test_fp  = split_files['test']['out_map'][ub_tok]

        # ==== LOAD ====
        input_train  = read_timesheet(in_train_fp)
        output_train = read_timesheet(out_train_fp)
        input_test   = read_timesheet(in_test_fp)
        output_test  = read_timesheet(out_test_fp)

        # ==== CLEAN & ALIGN ====
        input_train_T, input_test_T, output_train_T, output_test_T, L = clean_align(
            input_train, input_test, output_train, output_test
        )

        # ==== TO SEQUENCES ====
        seq_in,  n_samp_in,  T_in  = to_seq3d(input_train_T)
        seq_out, n_samp_out, T_out = to_seq3d(output_train_T)
        print("Train X:", seq_in.shape, "Train Y:", seq_out.shape)

        # ==== TUNER ====
        project_tag = f"{input_subdir.name}__{ub_tok}".replace(' ', '_')
        tuner = BayesianOptimization(
            build_model_builder(T_in),
            objective='val_mae',
            max_trials=10,
            directory=str(SAVE['Hyperparameters']),  # each subfolder has its own tuner dir
            project_name=project_tag,
            overwrite=True
        )
        callbacks = [EarlyStopping(monitor='val_mae', mode="min", patience=5, restore_best_weights=True)]
        tuner.search(seq_in, seq_out, epochs=1000, validation_split=0.1, shuffle=True,
                     callbacks=callbacks, verbose=1)

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        # Save best HPs
        pd.DataFrame([best_hp.values]).to_excel(
            SAVE['Hyperparameters'] / f"{project_tag}__best_hyperparameters.xlsx",
            index_label='Hyperparameter'
        )

        # Rebuild from best HP and fit again
        model = tuner.hypermodel.build(best_hp)
        model.summary()
        with open(SAVE['Summary'] / f"{project_tag}__Best_model_summary.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        model.fit(seq_in, seq_out, epochs=1000, validation_split=0.1, shuffle=True,
                  callbacks=callbacks, verbose=1)

        # Save model
        model.save(SAVE['Models'] / f"{project_tag}__best.h5")

        # ==== TEST ====
        seq_test, n_samp_test, T_test = to_seq3d(input_test_T)
        seq_base, _, _ = to_seq3d(output_test_T)
        pred = model.predict(seq_test, verbose=0)

        # Per-sample plots + metrics
        for k in range(n_samp_test):
            a = L
            x = np.arange(a)
            y_pred = pred[k, :, 0]
            y_true = seq_base[k, :, 0]

            # Plot
            plt.figure()
            plt.plot(x, y_pred, label="Predicted GRF (pred)")
            plt.plot(x, y_true, label="Measured GRF (expt)")
            plt.legend()
            fig_name = SAVE['Figures'] / f"{project_tag}__sample-{k}.png"
            plt.savefig(fig_name)
            plt.close()

            # Metrics
            mse_pred = np.square(np.subtract(y_true, y_pred)).mean()
            rmse_pred = sqrt(mse_pred)
            denom = (y_true.max() - y_true.min())
            nrmse_pred = rmse_pred / (denom if denom != 0 else 1.0)

            rmse_df  = pd.DataFrame([rmse_pred], columns=["ForcePlate*Seq2Seq_LSTM"])
            nrmse_df = pd.DataFrame([nrmse_pred], columns=["ForcePlate*Seq2Seq_LSTM"])
            out_df = pd.DataFrame({"ForcePlate": y_true, "Seq2Seq_LSTM": y_pred})

            # Save Excel
            stat_path = SAVE['Stats'] / f"{project_tag}__model_output__sample-{k}.xlsx"
            with pd.ExcelWriter(stat_path, engine='xlsxwriter') as writer:
                out_df.to_excel(writer, sheet_name='Raw_data', index=False)
                rmse_df.to_excel(writer, sheet_name='RMSE', index=False)
                nrmse_df.to_excel(writer, sheet_name='NRMSE', index=False)
                pd.DataFrame(["ForcePlate", "Seq2Seq_LSTM"]).to_excel(writer, sheet_name='Header', index=False, header=False)
                pd.DataFrame(["ForcePlate*Seq2Seq_LSTM"]).to_excel(writer, sheet_name='Header2', index=False, header=False)

        # Clear session AFTER finishing this UB pair
        K.clear_session()

print("\nDone.")
