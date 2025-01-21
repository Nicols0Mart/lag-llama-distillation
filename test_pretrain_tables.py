# %% [markdown]
# ## Imports

# %%
import itertools
from gluonts.dataset.common import MetaData
from typing import List, Optional, Callable, Iterable
from itertools import islice
import json, sys
import pickle
import os, random
import scipy.sparse as sp
import torch_metrics
import metrics
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator

import lightning.pytorch as pl
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    InstanceSampler
)
from gluonts.evaluation import make_evaluation_predictions, Evaluator
# from gluonts.torch.model.tft.module import TemporalFusionTransformerModel
# from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import (
    FileDataset, ListDataset, TrainDatasets,
    CategoricalFeatureInfo, BasicFeatureInfo,
)
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from datasets import Dataset, Features, Value, Sequence
from pandas.tseries.frequencies import to_offset


# %%
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# %%
train_dataset_names = ["PEMS03", "PEMS07", "PEMS08"]
# test_datasets = ["PEMS07M"]
test_datasets = ["PEMS04"]

data_id_to_name_map = {}
name_to_data_id_map  = {}
dataset_paths = {
     "PEMS03": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS03/PEMS03.npz",
     "PEMS04": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS04/PEMS04.npz",
     "PEMS07": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS07/PEMS07.npz",
     "PEMS08": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS08/PEMS08.npz",
     "PEMS07M": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS07M/PEMS07M.npz",
}
starts = {
     "PEMS03": "2018-09-01",
     "PEMS04": "2018-01-01",
     "PEMS07": "2017-05-01",
     "PEMS08": "2016-07-01",
     "PEMS07M": "2016-07-01"
}
for data_id, name in enumerate(train_dataset_names):
     data_id_to_name_map[data_id] = name
     name_to_data_id_map[name] = data_id
test_data_id = -1
for name in test_datasets:
     data_id_to_name_map[test_data_id] = name
     name_to_data_id_map[name] = test_data_id
     test_data_id -= 1

# %% [markdown]
# ### Pretraining load dataset

# %%

def create_train_dataset_without_last_k_timesteps(
    raw_train_dataset,
    freq,
    k=0
):
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        s_train["target"] = s_train["target"][:len(s_train["target"])-k]
        train_data.append(s_train)
    train_data = ListDataset(train_data, freq=freq)
    return train_data

class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        
    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])
    

class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


def _count_timesteps(
    left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset
) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """
    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if type(left) == pd.Period:
        left = left.to_timestamp()
    if type(right) == pd.Period:
        right = right.to_timestamp()
    assert (
        right >= left
    ), f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )


def create_train_dataset_last_k_percentage(
    raw_train_dataset,
    freq,
    k=100
):
    # Get training data
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        number_of_values = int(len(s_train["target"]) * k / 100)
        train_start_index = len(s_train["target"]) - number_of_values
        s_train["target"] = s_train["target"][train_start_index:]
        train_data.append(s_train)

    train_data = ListDataset(train_data, freq=freq)

    return train_data


def create_train_and_val_datasets_with_dates(
    name,
    dataset_path,
    data_id,
    history_length,
    prediction_length=None,
    num_val_windows=None,
    val_start_date=None,
    train_start_date=None,
    freq=None,
    last_k_percentage=None
):
    """
    Train Start date is assumed to be the start of the series if not provided
    Freq is not given is inferred from the data
    We can use ListDataset to just group multiple time series - https://github.com/awslabs/gluonts/issues/695
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        raw_dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                        "function_delay_minute", "instances_minute", \
                        "memory_limit_minute", "memory_usage_minute", \
                        "platform_delay_minute", "requests_minute"):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f: data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        raw_dataset = get_dataset(name, path=Path(dataset_path))

    if prediction_length is None:
        prediction_length = raw_dataset.metadata.prediction_length
    if freq is None:
        freq = raw_dataset.metadata.freq
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    raw_train_dataset = raw_dataset.train

    if not num_val_windows and not val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")
    if num_val_windows and val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")

    max_train_end_date = None

    # Get training data
    total_train_points = 0
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = train_end_index - number_of_values
        else:
            train_start_index = 0
        s_train["target"] = series["target"][train_start_index:train_end_index]
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + to_offset(freq) * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        s_val["start"] = series["start"] + val_start_index * timestep_delta
        s_val["target"] = series["target"][val_start_index:]
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
        total_train_points
        + total_val_points
        - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
    )


def create_test_dataset(
    name, dataset_path, history_length, freq=None, data_id=None
):
    """
    For now, only window per series is used.
    make_evaluation_predictions automatically only predicts for the last "prediction_length" timesteps
    NOTE / TODO: For datasets where the test set has more series (possibly due to more timestamps), \
    we should check if we only use the last N series where N = series per single timestamp, or if we should do something else.
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = os.path.join(dataset_path, "ett_datasets")
        dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                        "function_delay_minute", "instances_minute", \
                        "memory_limit_minute", "memory_usage_minute", \
                        "platform_delay_minute", "requests_minute"):
        path = os.path.join(dataset_path, "huawei/" + name + ".json")
        with open(path, "r") as f: data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = os.path.join(dataset_path, "air_quality/" + name + ".json")
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    else:
        dataset = get_dataset(name, path=Path(dataset_path))

    if freq is None:
        freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    data = []
    total_points = 0
    for i, series in enumerate(dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": series["start"] + offset,
                    "item_id": i,
                    "data_id": data_id,
                }
            )
        else:
            series_copy = copy.deepcopy(series)
            series_copy["item_id"] = i
            series_copy["data_id"] = data_id
            data.append(series_copy)
        total_points += len(data[-1]["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points

# %%
node_dict = {}
node_dict['PEMS08'], node_dict['PEMS07'], node_dict['PEMS04'], node_dict['PEMS03'], node_dict['PEMS07M'] = 170, 883, 307, 358, 228
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

# %%
KNN = 3

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    print(f"Number of isolated points: {isolated_point_num}")
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe


def pems_loader(path):
    df_pems4 = np.load(path)["data"]
    try:
        df_pems4_new = pd.DataFrame(df_pems4[:,:,0])
    except:
        df_pems4_new = pd.DataFrame(df_pems4)
    return df_pems4_new

def dataset_factory_pems(
        loaded_df,
        date, 
        data_id, 
        windows: List[tuple[int, ...]]=None, 
        future_seq_len=12, 
        num_val_windows=14, 
        last_k_percentage=None, 
        val_start_date=None, 
        train_start_date=None, 
        freq="5min", 
        history_length=2016, 
        prediction_length=12, name="PEMS04",
    ):
    aggregated_04 = []
    import datetime
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=KNN)
    neigh.fit(loaded_df.T.values)
    k_nn = neigh.kneighbors_graph().toarray()
    for column in loaded_df.columns:
        first_nn = [c for c, val in enumerate(k_nn[column]) if val == 1]
        aggregated_04.append(loaded_df[first_nn].values.T)
    train_num_short = int(loaded_df.shape[0]* 0.7)
    train_data_04 = loaded_df[:train_num_short]
    feat_time_04 = [ddd[:, :train_num_short] for ddd in aggregated_04]
    feat_time_04[0].shape
    
    def to_deepar_format(dataframe, time_feature, index=None, dataset_name="PEMS04"):
        freq = "5min"
        start_index = datetime.datetime.strptime(date, "%Y-%m-%d")
        # day_data, week_data, _ = time_add(dataframe, **dataset_params[dataset_name])
        # if len(dataframe.shape) == 2:
        #     data = np.expand_dims(data, axis=-1)
        #     day_data = np.expand_dims(day_data, axis=-1).astype(int)
        #     week_data = np.expand_dims(week_data, axis=-1).astype(int)
        #     # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        #     data_time_related = np.concatenate([day_data, week_data], axis=-1)
        # elif len(dataframe.shape) > 2:
        #     day_data = np.expand_dims(day_data, axis=-1).astype(int)
        #     week_data = np.expand_dims(week_data, axis=-1).astype(int)
        #     data_time_related = np.concatenate([day_data, week_data], axis=-1)
        A, _ = get_adjacency_matrix(dataset_params[dataset_name]["file"], num_of_vertices=node_dict[dataset_name], **dataset_params[dataset_name]["args"])
        lpls = cal_lape(A.copy())
        if windows:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5) if windows is None else start_data,
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.FEAT_STATIC_REAL: lpls[i],
                        FieldName.ITEM_ID: i,
                        "data_id": data_id
                    } 
                    for i, c in enumerate(dataframe.columns) for start_data, _ in windows]
        else:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5),
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.FEAT_STATIC_REAL: lpls[i],
                        FieldName.ITEM_ID: i,
                        "data_id": data_id
                    } 
                    for i, c in enumerate(dataframe.columns)]
        return ListDataset(data, freq=freq)
    train_data_lds_04 = to_deepar_format(train_data_04, feat_time_04, dataset_name=name)
    test_data_lds_04 = to_deepar_format(loaded_df, aggregated_04, dataset_name=name)
    meta_data = MetaData(freq="5T", prediction_length=future_seq_len)
    raw_train_dataset = TrainDatasets(train=train_data_lds_04, test=test_data_lds_04, metadata=meta_data)
    max_train_end_date = None
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    # Get training data
    total_train_points = 0
    train_data = []
    for i, series in enumerate(raw_train_dataset.train):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = train_end_index - number_of_values
        else:
            train_start_index = 0
        s_train["target"] = series["target"][train_start_index:train_end_index]
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + to_offset(freq) * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    for i, series in enumerate(raw_train_dataset.train):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        s_val["start"] = series["start"] + val_start_index * timestep_delta
        s_val["target"] = series["target"][val_start_index:]
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
        total_train_points
        + total_val_points
        - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
    )

# %%
dataset_params = {
            "PEMS07M":{
                "file":"/home/seyed/PycharmProjects/step/FlashST/data/PEMS07M/PEMS07M.csv",
                "args":{}
            },
            "PEMS07":{
                "file": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS07/PEMS07.csv",
                "args":{}
            },
            "PEMS08":{
                "file":"/home/seyed/PycharmProjects/step/FlashST/data/PEMS08/PEMS08.csv",
                "args":{}
            },
            "PEMS03":{
                "file": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS03/PEMS03.csv",
                "args": {"id_filename":"/home/seyed/PycharmProjects/step/FlashST/data/PEMS03/PEMS03.txt"}
            },
            "PEMS04":{
                "file": "/home/seyed/PycharmProjects/step/FlashST/data/PEMS04/PEMS04.csv",
                "args": {}
            },
        }

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    From STGCN-IJCAI2018
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, WMASK = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        A = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * WMASK
        return A
    else:
        return W


def test_dataset_factory_pems(
        loaded_df,
        date, 
        data_id, 
        windows: List[tuple[int, ...]]=None, 
        future_seq_len=12, 
        num_val_windows=14, 
        last_k_percentage=None, 
        val_start_date=None, 
        train_start_date=None, 
        freq="5min", 
        history_length=2016, 
        prediction_length=12,
        name=None
    ):
    aggregated_04 = []
    import datetime
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=KNN)
    neigh.fit(loaded_df.T.values)
    k_nn = neigh.kneighbors_graph().toarray()
    for column in loaded_df.columns:
        first_nn = [c for c, val in enumerate(k_nn[column]) if val == 1]
        aggregated_04.append(loaded_df[first_nn].values.T)
    train_num_short = int(loaded_df.shape[0]* 0.7)
    train_data_04 = loaded_df[:train_num_short]
    feat_time_04 = [ddd[:, :train_num_short] for ddd in aggregated_04]
    feat_time_04[0].shape
    def to_deepar_format(dataframe, time_feature, index=None, dataset_name="PEMS04"):
        freq = "5min"
        start_index = datetime.datetime.strptime(date, "%Y-%m-%d")
        
        # day_data, week_data, _ = time_add(dataframe, **dataset_params[dataset_name])
        # if len(dataframe.shape) == 2:
        #     data = np.expand_dims(data, axis=-1)
        #     day_data = np.expand_dims(day_data, axis=-1).astype(int)
        #     week_data = np.expand_dims(week_data, axis=-1).astype(int)
        #     # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        #     data_time_related = np.concatenate([day_data, week_data], axis=-1)
        # elif len(dataframe.shape) > 2:
        #     day_data = np.expand_dims(day_data, axis=-1).astype(int)
        #     week_data = np.expand_dims(week_data, axis=-1).astype(int)
        #     data_time_related = np.concatenate([day_data, week_data], axis=-1)
        if name != "PEMS07M":
            A, _ = get_adjacency_matrix(dataset_params[dataset_name]["file"], num_of_vertices=node_dict[dataset_name], **dataset_params[dataset_name]["args"])
        else:
            A = weight_matrix(dataset_params[dataset_name]["file"]).astype(np.float32)
            A = A + np.eye(A.shape[0])
        lpls = cal_lape(A.copy())
        if windows:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5) if windows is None else start_data,
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.FEAT_STATIC_REAL: lpls[i],
                        FieldName.ITEM_ID: i,
                        "data_id": data_id
                    } 
                    for i, c in enumerate(dataframe.columns) for start_data, _ in windows]
        else:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5),
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.FEAT_STATIC_REAL: lpls[i],
                        FieldName.ITEM_ID: i,
                        "data_id": data_id
                    } 
                    for i, c in enumerate(dataframe.columns)]
        return ListDataset(data, freq=freq)
    test_data = to_deepar_format(train_data_04, feat_time_04, dataset_name=name)
    train_data = create_train_dataset_without_last_k_timesteps(test_data, freq=freq, k=24)
    meta_data = MetaData(freq="5T", prediction_length=future_seq_len)
    raw_train_dataset = TrainDatasets(train=train_data, test=test_data, metadata=meta_data)
    data = []
    for i, series in enumerate(raw_train_dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": series["start"] + offset,
                    "item_id": i,
                    "data_id": data_id,
                    "feat_static_real": series["feat_static_real"]
                }
            )
        else:
            series_copy = copy.deepcopy(series)
            series_copy["item_id"] = i
            series_copy["data_id"] = data_id
            data.append(series_copy)
    return ListDataset(data, freq=freq)

# %%
all_datasets, val_datasets, dataset_num_series = [], [], []
dataset_train_num_points, dataset_val_num_points = [], []
for data_id, name in enumerate(train_dataset_names):
     # data_id = name_to_data_id_map[name]
     # (
     # train_dataset,
     # val_dataset,
     # total_train_points,
     # total_val_points,
     # total_val_windows,
     # max_train_end_date,
     # total_points,
     # ) = create_train_and_val_datasets_with_dates(
     # name,
     # args.dataset_path,
     # data_id,
     # history_length,
     # prediction_length,
     # num_val_windows=args.num_validation_windows,
     # last_k_percentage=args.single_dataset_last_k_percentage
     # )
     data_id = name_to_data_id_map[name]
     (
     train_dataset,
     val_dataset,
     total_train_points,
     total_val_points,
     total_val_windows,
     max_train_end_date,
     total_points,
     ) = dataset_factory_pems(loaded_df=pems_loader(dataset_paths[name]), date=starts[name], data_id=data_id, name=name)
     print(
     "Dataset:",
     name,
     "Total train points:", total_train_points,
     "Total val points:", total_val_points,
     )
     all_datasets.append(train_dataset)
     val_datasets.append(val_dataset)
     dataset_num_series.append(len(train_dataset))
     dataset_train_num_points.append(total_train_points)
     dataset_val_num_points.append(total_val_points)


train_data = CombinedDataset(all_datasets, weights=None)
val_data = CombinedDataset(val_datasets, weights=None)

# %%
fine_tune_datasets = test_datasets
fine_all_datasets, fine_val_datasets, fine_dataset_num_series = [], [], []
fine_dataset_train_num_points, fine_dataset_val_num_points = [], []
for data_id, name in enumerate(fine_tune_datasets):
     # data_id = name_to_data_id_map[name]
     # (
     # train_dataset,
     # val_dataset,
     # total_train_points,
     # total_val_points,
     # total_val_windows,
     # max_train_end_date,
     # total_points,
     # ) = create_train_and_val_datasets_with_dates(
     # name,
     # args.dataset_path,
     # data_id,
     # history_length,
     # prediction_length,
     # num_val_windows=args.num_validation_windows,
     # last_k_percentage=args.single_dataset_last_k_percentage
     # )
     data_id = name_to_data_id_map[name]
     (
     train_dataset,
     val_dataset,
     total_train_points,
     total_val_points,
     total_val_windows,
     max_train_end_date,
     total_points,
     ) = dataset_factory_pems(loaded_df=pems_loader(dataset_paths[name]), date=starts[name], data_id=data_id)
     print(
     "Dataset:",
     name,
     "Total train points:", total_train_points,
     "Total val points:", total_val_points,
     )
     fine_all_datasets.append(train_dataset)
     fine_val_datasets.append(val_dataset)
     fine_dataset_num_series.append(len(train_dataset))
     fine_dataset_train_num_points.append(total_train_points)
     fine_dataset_val_num_points.append(total_val_points)


fine_train_data = CombinedDataset(fine_all_datasets, weights=None)
fine_val_data = CombinedDataset(fine_val_datasets+[], weights=None)

# %%
test_dataset = test_dataset_factory_pems(loaded_df=pems_loader(dataset_paths["PEMS07M"]), date=starts["PEMS07M"], data_id=-1, name="PEMS07M")

# %%
test_dataset[0]

# %%
assert len(test_dataset) == 228


# %% [markdown]
# ###### Debug

# %%
from lag_llama.gluon.estimator import LagLlamaEstimator

ckpt = torch.load("checkpoints/lag-llama.ckpt", map_location=torch.device('cuda:0'))
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
# input_size = estimator_args["input_size"]
input_size = estimator_args["input_size"]
estimator = LagLlamaEstimator(
    ckpt_path=None,
    prediction_length=12,
    context_length=48*3,
    # estimator args
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"]*4,
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"]*2,
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],
    num_batches_per_epoch=350,
    trainer_kwargs={
        "max_epochs": 2
    },
    mistral=False,
)

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), mae_loss

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def calc_metrics(ground_truth, prediction, horizon=5):
    reals = torch.from_numpy(ground_truth[0].loc[prediction[0].start_date:prediction[0].start_date + 11].values)
    preds =  torch.from_numpy(prediction[0].mean).unsqueeze(0)

    for i in range(1, len(prediction)):
        real = ground_truth[i//101].loc[prediction[i].start_date:prediction[i].start_date + 11]
        pred = prediction[i].mean
        # for j in range(3):
        #      print(MAE_torch(torch.from_numpy(real.values[times[j]]), torch.from_numpy(pred[times[j],...]), 0.0), RMSE_torch(torch.from_numpy(real.values[times[j]]), torch.from_numpy(pred[times[j],...]), 0.0)) 
        preds = torch.cat([preds, torch.from_numpy(pred).unsqueeze(0)])
        reals = torch.cat([reals, torch.from_numpy(real.values)], dim=1)
    return MAE_torch(preds.T[horizon, ...], reals[horizon, ...], 0.0)[0], RMSE_torch(preds.T[horizon, ...], reals[horizon, ...], 0.0), metrics.masked_mape(reals[horizon, ...], preds.T[horizon, ...])
# %%
train_output = estimator.train_model(train_data, shuffle_buffer_length=None, ckpt_path=None, use_lora=False)

predictor = train_output.predictor
# lightning_module = train_output.trained_net
# BASED_CHECKPOINT_CL_96 = train_output.trainer.checkpoint_callback.best_model_path
# BASED_CHECKPOINT_CL_96 = "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_449/checkpoints/epoch=481-step=168700.ckpt"
# BASED_CHECKPOINT_CL_96 = "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_450/checkpoints/epoch=633-step=221900.ckpt"

# %%
mae_cl_96, rmse_cl_96, mape_cl_96 = [[], [], []], [[], [], []], [[], [], []]
estimator.trainer_kwargs["max_epochs"] = 670
# best_model_path = BEST_FINE_TUNE
# estimator.ckpt_path = BASED_CHECKPOINT_CL_96
# predictor = estimator.train(fine_train_data, shuffle_buffer_length=None, ckpt_path=BASED_CHECKPOINT_CL_96, use_lora=False)
# Make evaluations
forecast_it, ts_it = make_evaluation_predictions(
dataset=test_dataset, predictor=predictor, num_samples=100
)

forecasts = list(itertools.islice(forecast_it, 5000))
tss = list(itertools.islice(ts_it, 5000))

times = [2, 5, 11]
for i, t in enumerate(times):
     mae_val, rmse_val, mape_val = calc_metrics(tss, forecasts, t)
     mae_cl_96[i].append(mae_val)
     rmse_cl_96[i].append(rmse_val)
     mape_cl_96[i].append(mape_val)

print("15 Min_______________\n")
print("MAE", mae_cl_96[0][-1].item())
print("RMSE", rmse_cl_96[0][-1].item())
print("MAPE", mape_cl_96[0][-1].item())
print("30 Min_______________\n")
print("MAE", mae_cl_96[1][-1].item())
print("RMSE", rmse_cl_96[1][-1].item())
print("MAPE", mape_cl_96[1][-1].item())
print("60 Min_______________\n")
print("MAE", mae_cl_96[2][-1].item())
print("RMSE", rmse_cl_96[2][-1].item())
print("MAPE", mape_cl_96[2][-1].item())


# lightning_module.log("MAE", mae_cl_96[1][-1].item(), on_step=False, on_epoch=False, prog_bar=True, logger=True)
# lightning_module.log("RMSE", rmse_cl_96[1][-1].item(), on_step=False, on_epoch=False, prog_bar=True, logger=True)
# lightning_module.log("MAPE", mape_cl_96[1][-1].item(), on_step=False, on_epoch=False, prog_bar=True, logger=True)
