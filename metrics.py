import numpy as np


def mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, np.broadcast_to(np.array(null_val), labels.shape), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds-labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def mape(preds, labels, null_val=0.0):
    labels = np.where(np.abs(labels) < 1e-4, np.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, np.broadcast_to(np.array(null_val), labels.shape), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(np.abs(preds-labels)/labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, np.broadcast_to(np.array(null_val), labels.shape), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def rmse(preds, labels, null_val=0.0):
    return np.sqrt(mse(preds, labels, null_val))
    