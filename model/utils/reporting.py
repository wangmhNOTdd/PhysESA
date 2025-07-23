import torch
import torchmetrics


def get_regr_metrics_pt(y_true, y_pred):
    try:
        y_pred = torch.from_numpy(y_pred)
    except:
        pass

    return {
        "MAE": torchmetrics.functional.mean_absolute_error(y_pred, y_true),
        "RMSE": torchmetrics.functional.mean_squared_error(y_pred, y_true, squared=False),
        "R2": torchmetrics.functional.r2_score(y_pred, y_true),
        "SMAPE": torchmetrics.functional.symmetric_mean_absolute_percentage_error(y_pred, y_true),
        # 'ConcordanceCorrCoef': torchmetrics.functional.concordance_corrcoef(y_pred, y_true),
        # 'ExplainedVariance': torchmetrics.functional.explained_variance(y_pred, y_true),
        # 'KendallRankCorrCoef': torchmetrics.functional.kendall_rank_corrcoef(y_pred, y_true),
        # 'MSE': torchmetrics.functional.mean_squared_error(y_pred, y_true),
        # 'PearsonCorrCoef': r,
        # 'PearsonCorrCoefSquared': r ** 2,
        # 'SpearmanCorrCoef': torchmetrics.functional.spearman_corrcoef(y_pred, y_true),
        # 'SMAPE': torchmetrics.functional.symmetric_mean_absolute_percentage_error(y_pred, y_true)
    }


def get_cls_metrics_binary_pt(y_true, y_pred):
    y_true = y_true.to(torch.long)

    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(y_pred, y_true, threshold=0.5)
    auroc = torchmetrics.functional.classification.binary_auroc(y_pred, y_true, thresholds=None)
    accuracy = torchmetrics.functional.classification.binary_accuracy(y_pred, y_true, threshold=0.5)
    f1 = torchmetrics.functional.classification.binary_f1_score(y_pred, y_true, threshold=0.5)

    return auroc, mcc, accuracy, f1


def get_cls_metrics_multilabel_pt(y_true, y_pred, num_cls):
    y_true = y_true.to(torch.long)

    mcc = torchmetrics.functional.classification.multilabel_matthews_corrcoef(
        y_pred, y_true, num_labels=num_cls, threshold=0.5
    )
    auroc = torchmetrics.functional.classification.multilabel_auroc(
        y_pred, y_true, num_labels=num_cls, average="macro", thresholds=None
    )
    accuracy = torchmetrics.functional.classification.multilabel_accuracy(
        y_pred,
        y_true,
        num_labels=num_cls,
        average="macro",
        threshold=0.5,
    )
    f1 = torchmetrics.functional.classification.multilabel_f1_score(
        y_pred, y_true, num_labels=num_cls, average="macro", threshold=0.5
    )

    return auroc, mcc, accuracy, f1


def get_cls_metrics_multiclass_pt(y_true, y_pred, num_cls):
    y_true = y_true.to(torch.long)

    if 0 not in set(y_true.tolist()):
        y_true = y_true - 1

    mcc = torchmetrics.functional.classification.multiclass_matthews_corrcoef(y_pred, y_true, num_classes=num_cls)
    auroc = torchmetrics.functional.classification.multiclass_auroc(
        y_pred, y_true, num_classes=num_cls, average="macro", thresholds=None
    )
    accuracy = torchmetrics.functional.classification.multiclass_accuracy(
        y_pred,
        y_true,
        num_classes=num_cls,
        average="macro",
    )
    f1 = torchmetrics.functional.classification.multiclass_f1_score(
        y_pred, y_true, num_classes=num_cls, average="macro"
    )
    ap = torchmetrics.functional.classification.multiclass_average_precision(
        y_pred,
        y_true,
        num_classes=num_cls,
        average="macro",
        thresholds=None
    )

    return auroc, mcc, accuracy, f1, ap
