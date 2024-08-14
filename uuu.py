# %% [markdown]
# ## Imports

# %%
from gluonts.dataset.common import MetaData

# %%
from typing import List, Optional, Callable, Iterable
from itertools import islice
import json
import torch_metrics

import numpy as np
import pandas as pd
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
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.repository.datasets import get_dataset
from datasets import Dataset, Features, Value, Sequence

# %%
from omegaconf import OmegaConf

data_percentages = OmegaConf.load('/home/seyed/PycharmProjects/step/lag-llama/confs/transfer_learning.yaml')
percentages = data_percentages.percentages

# %%
id_sensor = pd.read_csv('/home/seyed/PycharmProjects/step/STEP/datasets/METR-LA/graph_sensor_locations.csv', index_col=0)
adj_path = "/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA/adj_METR-LA_nn.json"
rev_map = json.loads(id_sensor["sensor_id"].to_json())
main_map = {str(v):k for k,v in rev_map.items()}
nn_data = None
with open(adj_path, "r") as f:
    nn_data = json.load(f)
q = []
for i in nn_data:
    if i != "26":
        if len(nn_data[i]["1_hop"]["nodes"]) == 1:
            print(i, nn_data[i], rev_map[str(i)])
        q.append(len(nn_data[i]["1_hop"]["nodes"]))
DROP_IND = "26"
DROP_SENSOR = rev_map[DROP_IND]
df = pd.read_hdf("/home/seyed/Downloads/metr-la(1).h5")
df_new = df.copy()
aggregated = []
# ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
KNN = 3
for column in df_new.columns:
    if column != str(DROP_SENSOR):
        first_nn = nn_data[main_map[column]]["1_hop"]["nodes"]
        cols = [str(rev_map[str(i)]) for i in first_nn[:KNN]]
        aggregated.append(df_new[cols].values.T)
df.drop(columns=[str(DROP_SENSOR)], inplace=True)
df_new.drop(columns=[str(DROP_SENSOR)], inplace=True)
df.index = pd.to_datetime(df.index)
df.index.freq='5min'
history_seq_len = 2016
future_seq_len = 12
len_of_df = len(df)
num_samples = len_of_df - (history_seq_len + future_seq_len) + 1
    # keep same number of validation and test samples with Graph WaveNet (input 12, output 12)
test_num_short = 200
valid_num_short = 200
# train_num_short = num_samples - valid_num_short - test_num_short
train_num_short = 30000
train_data = df[:train_num_short]
feat_time = [ddd[:, :train_num_short] for ddd in aggregated]
test_data = df
feat_time[0].shape
def to_deepar_format(dataframe, time_feature):
    freq = pd.infer_freq(dataframe.index) 
    start_index = dataframe.index.min()
    data = [{
                FieldName.START:  start_index,
                FieldName.TARGET:  dataframe[c].values,
                FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                FieldName.ITEM_ID: i,
                "data_id": i
            } 
            for i, c in enumerate(dataframe.columns)]
    # print(data[0]["feat_dynamic_real"].shape)
    return ListDataset(data, freq=freq)
train_data_lds = to_deepar_format(train_data, feat_time)
test_data_lds = to_deepar_format(test_data, aggregated)

# %%
df.shape[0]

# %%
meta_data = MetaData(freq="5T", prediction_length=future_seq_len)
dataset_metrla = TrainDatasets(train=train_data_lds, test=test_data_lds, metadata=meta_data)

# %%
assert len(dataset_metrla.test[0]["target"]) == df.shape[0]

# %%
KNN = 3
df_pems4 = np.load("/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/PEMS04/PEMS04.npz")["data"]
df_pems4_new = pd.DataFrame(df_pems4[:,:,0])


def dataset_factory(df_pems4_new, windows: List[tuple[int, ...]]=None):
    aggregated_04 = []
    import datetime
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=KNN)
    neigh.fit(df_pems4_new.T.values)
    k_nn = neigh.kneighbors_graph().toarray()
    for column in df_pems4_new.columns:
        first_nn = [c for c, val in enumerate(k_nn[column]) if val == 1]
        aggregated_04.append(df_pems4_new[first_nn].values.T)
    train_num_short = int(df_pems4_new.shape[0]* 0.7)
    train_data_04 = df_pems4_new[:train_num_short]
    feat_time_04 = [ddd[:, :train_num_short] for ddd in aggregated_04]
    feat_time_04[0].shape
    def to_deepar_format(dataframe, time_feature, index=None):
        freq = "5min"
        start_index = datetime.datetime(2018, 1, 1, 0, 0)
        if windows:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5) if windows is None else start_data,
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.ITEM_ID: i,
                        "data_id": i
                    } 
                    for i, c in enumerate(dataframe.columns) for start_data, _ in windows]
        else:
            data = [{
                        FieldName.START:  start_index if (index is None and windows is None) else start_index + datetime.timedelta(minutes=index * 5),
                        FieldName.TARGET:  dataframe[c].values,
                        FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
                        FieldName.ITEM_ID: i,
                        "data_id": i
                    } 
                    for i, c in enumerate(dataframe.columns)]
        return ListDataset(data, freq=freq)
    train_data_lds_04 = to_deepar_format(train_data_04, feat_time_04)
    test_data_lds_04 = to_deepar_format(df_pems4_new, aggregated_04)

    return train_data_lds_04, test_data_lds_04

# %%
df_pems4_new.set_index(pd.date_range(start="2018-01-01", periods=df_pems4_new.shape[0], freq="5min"), inplace=True)

import datetime

INDEX = 7

def df_sampler(input_df, index=0):
    current_percent = percentages[index]
    possible_indices = list(range(0, len(df_pems4_new)- round(16992 * (current_percent/100))))
    sampled_indices = np.random.choice(possible_indices, current_percent, replace=False)
    context_window = 192
    start_offset = datetime.datetime(2018, 1, 1, 0, 0)
    dates = [(start_offset + datetime.timedelta(minutes=int(i) * 5), 0) for i in sampled_indices]
    df_merged = pd.concat([input_df.iloc[index: index + context_window + 1] for index in sampled_indices])
    return df_merged, dates

# sampled_df, dates = df_sampler(df_pems4_new, index=INDEX)


meta_data = MetaData(freq="5T", prediction_length=future_seq_len)
train_data_lds_04, test_data_lds_04 = dataset_factory(df_pems4_new, windows=None)
dataset_04 = TrainDatasets(train=train_data_lds_04, test=test_data_lds_04, metadata=meta_data)

# %%
import metrics

def evaluate_model(estimator, dataset, module, path):
    estimator.ckpt_path = path or "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_225/checkpoints/epoch=49-step=15000.ckpt"
    module = module or estimator.create_lightning_module(use_lora=False)
    test_loader = estimator.create_test_dataloader(
        module, freq="5min", data=dataset or dataset_04.test, batch_size=4
    )
    mae_list = []
    rmse_list = []
    mape_list = []
    for batch in test_loader:
        print(batch)
        for tns in batch:
            batch[tns] = batch[tns].to(module.device)
        outputs = module(**batch)
        ground_truth = batch["future_target"].cpu().numpy()
        meds = np.median(outputs.cpu().numpy(), axis=1)
        rmse = metrics.rmse(meds,ground_truth)
        mae = metrics.mae(meds,ground_truth)
        mape = metrics.mape(meds,ground_truth)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
    print(f"MAE: {np.mean(mae_list)}")

    # %%
    print(f"MAPE: {np.mean(mape_list)}")

    # %%
    print(f"RMSE: {np.mean(rmse_list)}")
    plt.scatter(mape_list, rmse_list, alpha=0.2, c="r")
    plt.xlabel("MASE")
    plt.ylabel("MAPE")
    plt.show()
    return mae_list, mape_list, rmse_list


# %%
# def to_deepar_format_all(dataframe, time_feature, d2, t2):
#     freq = pd.infer_freq(dataframe.index) 
#     start_index = dataframe.index.min()
#     data = [{
#                 FieldName.START:  start_index,
#                 FieldName.TARGET:  dataframe[c].values,
#                 FieldName.FEAT_DYNAMIC_REAL: time_feature[i],
#                 FieldName.ITEM_ID: i,
#                 "data_id": i
#             } 
#             for i, c in enumerate(dataframe.columns)]
#     data2 = [{
#                 FieldName.START:  datetime.datetime(2018, 1, 1, 0, 0),
#                 FieldName.TARGET:  d2[c].values,
#                 FieldName.FEAT_DYNAMIC_REAL: t2[i],
#                 FieldName.ITEM_ID: i,
#                 "data_id": i
#             } 
#             for i, c in enumerate(d2.columns)]
#     data.extend(data2)
#     # print(data[0]["feat_dynamic_real"].shape)
#     return ListDataset(data, freq=freq)
# train_data_all_lds = to_deepar_format_all(train_data, feat_time, train_data_04, feat_time_04)
# test_data_all_lds = to_deepar_format_all(test_data, aggregated, test_data_04, aggregated_04)

# %%
train_data_metrla = to_deepar_format(train_data, feat_time)
test_data_lds_metrla = to_deepar_format(test_data, aggregated)

# %%

# dataset_metr_la = TrainDatasets(train=train_data_metrla, test=test_data_lds_metrla, metadata=meta_data)
# dataset_04 = TrainDatasets(train=train_data_lds_04, test=test_data_lds_04, metadata=meta_data)
# meta_data

# %%

# train_data_lds_04
# test_data_lds_04

# %%
# from gluonts.dataset.multivariate_grouper import MultivariateGrouper

# num_of_variates = len(dataset_04.train)

# test_grouper = MultivariateGrouper(
#     max_target_dim=num_of_variates,
#     num_test_dates=1, # number of rolling test windows
# )
# multi_variate_test_dataset = test_grouper(dataset_04.test)

# %%
dataset_metr_la = TrainDatasets(train=train_data_metrla, test=test_data_lds_metrla, metadata=meta_data)


# %%
import os

file_path = "/home/seyed/PycharmProjects/step/lag-llama/checkpoints/lag-llama.ckpt"
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")


# %%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()

# %%
CHECKPOINTS = {
    "exp_2": "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_131/checkpoints/epoch=37-step=38000.ckpt"
}


# %%
from lag_llama.gluon.estimator import LagLlamaDomainAdaptiationEstimator, LagLlamaEstimator

# %%
BASED_CHECKPOINTES = {
    "metrla": "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_225/checkpoints/epoch=49-step=15000.ckpt",
    "pems04": "/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_309/checkpoints/epoch=149-step=750000.ckpt",
    "pemsbay": "###########",
}

# %% [markdown]
# ### A1- New Evaluator

# %%
### A-1 with new evaluator
from lag_llama.gluon.estimator import LagLlamaEstimator
ckpt = torch.load("/home/seyed/PycharmProjects/step/lag-llama/checkpoints/lag-llama.ckpt", map_location=torch.device('cuda:0'))
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
estimator = LagLlamaEstimator(
    ckpt_path=None,
    batch_size=4,
    prediction_length=12,
    context_length=48,
    # estimator args
    input_size=1,
    n_layer=32,
    n_embd_per_head=128 // 2,
    n_head=32// 2,
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],
    num_batches_per_epoch=5000,
    trainer_kwargs={
        "max_epochs": 150
    }
)
# predictor = estimator.train(dataset_04.train, ckpt_path="/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_192/checkpoints/epoch=98-step=99000.ckpt")
# predictor = estimator.train(dataset_04.train, ckpt_path="/home/seyed/PycharmProjects/step/lag-llama/lightning_logs/version_276/checkpoints/epoch=99-step=500000.ckpt")
predictor = estimator.train(dataset_metrla.train)
