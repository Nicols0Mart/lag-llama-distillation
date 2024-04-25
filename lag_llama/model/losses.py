import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import grad
import pytorch_lightning as pl

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    correct = y_hat.eq(label.view(label.size(0)).type_as(y_hat))
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(
            class_output, label.type_as(y_hat).view(label.size(0))
        )
        loss = torch.sum(weights * losses)  / torch.sum(weights)
    return loss, correct


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def entropy_logits_loss(linear_output):
    return torch.mean(entropy_logits(linear_output))


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116

    alpha = torch.rand(h_s.size(0), 1)
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    try:
        differences = h_t - h_s

        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

        preds = critic(interpolates)
        gradients = grad(
            preds,
            interpolates,
            grad_outputs=torch.ones_like(preds),
            retain_graph=True,
            create_graph=True,
        )[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    except:
        gradient_penalty = 0

    return gradient_penalty


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Code from XLearn: computes the full kernel matrix,
    which is less than optimal since we don't use all of it
    with the linear MMD estimate.
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)  # /len(kernel_val)


def compute_mmd_loss(kernel_values, batch_size):
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernel_values[s1, s2] + kernel_values[t1, t2]
        loss -= kernel_values[s1, t2] + kernel_values[s2, t1]
    return loss / float(batch_size)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_aggregated_metrics(metric_name_list, metric_outputs):
    metric_dict = {}
    for metric_name in metric_name_list:
        metric_dim = len(metric_outputs[0][metric_name].shape)
        if metric_dim == 0:
            metric_value = torch.stack([x[metric_name] for x in metric_outputs]).mean()
        else:
            metric_value = (
                torch.cat([x[metric_name] for x in metric_outputs]).double().mean()
            )
        metric_dict[metric_name] = metric_value.item()
    return metric_dict


def get_aggregated_metrics_from_dict(input_metric_dict):
    metric_dict = {}
    for metric_name, metric_value in input_metric_dict.items():
        metric_dim = len(metric_value.shape)
        if metric_dim == 0:
            metric_dict[metric_name] = metric_value
        else:
            metric_dict[metric_name] = metric_value.double().mean()
    return metric_dict



# multi-GPUs: mandatory to convert float values into tensors
def get_metrics_from_parameter_dict(parameter_dict, device):
    return {k: torch.tensor(v, device=device) for k, v in parameter_dict.items()}


class BaseAdaptTrainer(pl.LightningModule):
    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        method=None,
        lambda_init=1.0,
        adapt_lambda=True,
        adapt_lr=True,
        nb_init_epochs=10,
        nb_adapt_epochs=50,
        batch_size=32,
        init_lr=1e-3,
        optimizer=None,
    ):
        """Base class for all domain adaptation architectures.

        This class implements the classic building blocks used in all the derived architectures
        for domain adaptation.
        If you inherit from this class, you will have to implement only:
         - a forward pass
         - a `compute_loss` function that returns the task loss :math:`\mathcal{L}_c` and adaptation loss :math:`\mathcal{L}_a`, as well as
           a dictionary for summary statistics and other metrics you may want to have access to.

        The default training step uses only the task loss :math:`\mathcal{L}_c` during warmup,
        the uses the loss defined as:

        :math:`\mathcal{L} = \mathcal{L}_c + \lambda \mathcal{L}_a`,

        where :math:`\lambda` will follow the schedule defined by the DANN paper:

        :math:`\lambda_p = \frac{2}{1 + \exp{(-\gamma \cdot p)}} - 1` where $p$ the learning progress
        changes linearly from 0 to 1.

        Args:
            dataset (ada.datasets.MultiDomainDatasets): the multi-domain datasets to be used
                for train, validation, and tests.
            feature_extractor (torch.nn.Module): the feature extractor network (mapping inputs :math:`x\in\mathcal{X}` to
                a latent space :math:`\mathcal{Z}`,)
            task_classifier (torch.nn.Module): the task classifier network that learns to predict labels
                :math:`y \in \mathcal{Y}` from latent vectors,
            method (Method, optional): the method implemented by the class. Defaults to None.
                Mostly useful when several methods may be implemented using the same class.
            lambda_init (float, optional): Weight attributed to the adaptation part of the loss. Defaults to 1.0.
            adapt_lambda (bool, optional): Whether to make lambda grow from 0 to 1 following the schedule from
                the DANN paper. Defaults to True.
            adapt_lr (bool, optional): Whether to use the schedule for the learning rate as defined
                in the DANN paper. Defaults to True.
            nb_init_epochs (int, optional): Number of warmup epochs (during which lambda=0, training only on the source). Defaults to 10.
            nb_adapt_epochs (int, optional): Number of training epochs. Defaults to 50.
            batch_size (int, optional): Defaults to 32.
            init_lr (float, optional): Initial learning rate. Defaults to 1e-3.
            optimizer (dict, optional): Optimizer parameters, a dictionary with 2 keys:
                "type": a string in ("SGD", "Adam", "AdamW")
                "optim_params": kwargs for the above PyTorch optimizer.
                Defaults to None.
        """
        super().__init__()
        self._method = method

        self._init_lambda = lambda_init
        self.lamb_da = lambda_init
        self._adapt_lambda = adapt_lambda
        self._adapt_lr = adapt_lr

        self._init_epochs = nb_init_epochs
        self._non_init_epochs = nb_adapt_epochs - self._init_epochs
        assert self._non_init_epochs > 0
        self._batch_size = batch_size
        self._init_lr = init_lr
        self._lr_fact = 1.0
        self._grow_fact = 0.0
        self._dataset = dataset
        self.feat = feature_extractor
        self.classifier = task_classifier
        self._dataset.prepare_data_loaders()
        self._nb_training_batches = None  # to be set by method train_dataloader
        self._optimizer_params = optimizer

    @property
    def method(self):
        return self._method

    def _update_batch_epoch_factors(self, batch_id):
        if self.current_epoch >= self._init_epochs:
            delta_epoch = self.current_epoch - self._init_epochs
            p = (batch_id + delta_epoch * self._nb_training_batches) / (
                self._non_init_epochs * self._nb_training_batches
            )
            self._grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            if self._adapt_lr:
                self._lr_fact = 1.0 / ((1.0 + 10 * p) ** 0.75)

        if self._adapt_lambda:
            self.lamb_da = self._init_lambda * self._grow_fact

    def get_parameters_watch_list(self):
        """
        Update this list for parameters to watch while training (ie log with MLFlow)
        """
        return {
            "lambda": self.lamb_da,
            "last_epoch": self.current_epoch,
        }

    def forward(self, x):
        raise NotImplementedError("Forward pass needs to be defined.")

    def compute_loss(self, batch, split_name="V"):
        """Define the loss of the model

        Args:
            batch (tuple): batches returned by the MultiDomainLoader.
            split_name (str, optional): learning stage (one of ["T", "V", "Te"]).
                Defaults to "V" for validation. "T" is for training and "Te" for test.
                This is currently used only for naming the metrics used for logging.

        Returns:
            a 3-element tuple with task_loss, adv_loss, log_metrics.
            log_metrics should be a dictionary.

        Raises:
            NotImplementedError: children of this classes should implement this method.
        """
        raise NotImplementedError("Loss needs to be defined.")

    def training_step(self, batch, batch_nb):
        """The most generic of training steps

        Args:
            batch (tuple): the batch as returned by the MultiDomainLoader dataloader iterator:
                2 tuples: (x_source, y_source), (x_target, y_target) in the unsupervised setting
                3 tuples: (x_source, y_source), (x_target_labeled, y_target_labeled), (x_target_unlabeled, y_target_unlabeled) in the semi-supervised setting
            batch_nb (int): id of the current batch.

        Returns:
            dict: must contain a "loss" key with the loss to be used for back-propagation.
                see pytorch-lightning for more details.
        """
        self._update_batch_epoch_factors(batch_nb)

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="T")
        if self.current_epoch < self._init_epochs:
            # init phase doesn't use few-shot learning
            # ad-hoc decision but makes models more comparable between each other
            loss = task_loss
        else:
            loss = task_loss + self.lamb_da * adv_loss

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(
            get_metrics_from_parameter_dict(
                self.get_parameters_watch_list(), loss.device
            )
        )
        log_metrics["T_total_loss"] = loss
        log_metrics["T_task_loss"] = task_loss

        return {
            "loss": loss,  # required, for backward pass
            "progress_bar": {"class_loss": task_loss},
            "log": log_metrics,
        }

    def validation_step(self, batch, batch_nb):
        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="V")
        loss = task_loss + self.lamb_da * adv_loss
        log_metrics["val_loss"] = loss
        return log_metrics

    def _validation_epoch_end(self, outputs, metrics_at_valid):
        log_dict = get_aggregated_metrics(metrics_at_valid, outputs)
        device = outputs[0].get("val_loss").device
        log_dict.update(
            get_metrics_from_parameter_dict(self.get_parameters_watch_list(), device)
        )
        avg_loss = log_dict["val_loss"]
        return {
            "val_loss": avg_loss,  # for callbacks (eg early stopping)
            "progress_bar": {"val_loss": avg_loss},
            "log": log_dict,
        }

    def on_validation_epoch_end(self, outputs):
        metrics_to_log = (
            "val_loss",
            "V_source_acc",
            "V_target_acc",
        )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_step(self, batch, batch_nb):
        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="Te")
        loss = task_loss + self.lamb_da * adv_loss
        log_metrics["test_loss"] = loss
        return log_metrics

    def test_epoch_end(self, outputs):
        metrics_at_test = (
            "test_loss",
            "Te_source_acc",
            "Te_target_acc",
        )
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        return {
            "avg_test_loss": log_dict["test_loss"],
            "progress_bar": log_dict,
            "log": log_dict,
        }

    def _configure_optimizer(self, parameters):
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(
                parameters,
                lr=self._init_lr,
                betas=(0.8, 0.999),
                weight_decay=1e-5,
            )
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )

            if self._adapt_lr:
                feature_sched = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: self._lr_fact
                )
                return [optimizer], [feature_sched]
            return [optimizer]
        raise NotImplementedError(
            f"Unknown optimizer type {self._optimizer_params['type']}"
        )

    def configure_optimizers(self):
        return self._configure_optimizer(self.parameters())

    def train_dataloader(self):
        dataloader = self._dataset.get_domain_loaders(
            split="train", batch_size=self._batch_size
        )
        self._nb_training_batches = len(dataloader)
        return dataloader

    def val_dataloader(self):
        return self._dataset.get_domain_loaders(
            split="valid", batch_size=self._batch_size
        )

    def test_dataloader(self):
        return self._dataset.get_domain_loaders(
            split="test", batch_size=self._batch_size
        )


class BaseMMDLike(BaseAdaptTrainer):
    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        kernel_mul=2.0,
        kernel_num=5,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)

        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        class_output = self.classifier(x)
        return x, class_output

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        raise NotImplementedError("You need to implement a MMD-loss")

    def compute_loss(self, batch, split_name="V"):
        if len(batch) == 3:
            raise NotImplementedError("MMD does not support semi-supervised setting.")
        (x_s, y_s), (x_tu, y_tu) = batch

        phi_s, y_hat = self.forward(x_s)
        phi_t, y_t_hat = self.forward(x_tu)

        loss_cls, ok_src = cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = cross_entropy_logits(y_t_hat, y_tu)

        mmd = self._compute_mmd(phi_s, phi_t, y_hat, y_t_hat)
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_mmd": mmd,
        }
        return task_loss, mmd, log_metrics

    def on_validation_epoch_end(self, outputs):
        metrics_to_log = (
            "val_loss",
            "V_source_acc",
            "V_target_acc",
            "V_mmd",
        )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        metrics_at_test = (
            "test_loss",
            "Te_source_acc",
            "Te_target_acc",
            "Te_mmd",
        )
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        return {
            "avg_test_loss": log_dict["test_loss"],
            "progress_bar": log_dict,
            "log": log_dict,
        }


class DANtrainer(BaseMMDLike):
    """
    This is an implementation of DAN
    Long, Mingsheng, et al.
    "Learning Transferable Features with Deep Adaptation Networks."
    International Conference on Machine Learning. 2015.
    http://proceedings.mlr.press/v37/long15.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(self, dataset, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = gaussian_kernel(
            phi_s,
            phi_t,
            kernel_mul=self._kernel_mul,
            kernel_num=self._kernel_num,
        )
        return compute_mmd_loss(kernels, batch_size)


class JANtrainer(BaseMMDLike):
    """
    This is an implementation of JAN
    Long, Mingsheng, et al.
    "Deep transfer learning with joint adaptation networks."
    International Conference on Machine Learning, 2017.
    https://arxiv.org/pdf/1605.06636.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        kernel_mul=(2.0, 2.0),
        kernel_num=(5, 1),
        **base_params,
    ):
        super().__init__(
            dataset,
            feature_extractor,
            task_classifier,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            **base_params,
        )

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        softmax_layer = torch.nn.Softmax(dim=-1)
        source_list = [phi_s, softmax_layer(y_hat)]
        target_list = [phi_t, softmax_layer(y_t_hat)]
        batch_size = int(phi_s.size()[0])

        joint_kernels = None
        for source, target, k_mul, k_num, sigma in zip(
            source_list, target_list, self._kernel_mul, self._kernel_num, [None, 1.68]
        ):
            kernels = gaussian_kernel(
                source, target, kernel_mul=k_mul, kernel_num=k_num, fix_sigma=sigma
            )
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return compute_mmd_loss(joint_kernels, batch_size)