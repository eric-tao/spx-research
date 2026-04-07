from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureRow


@dataclass
class ProbabilityBand:
    label: str
    lower_price: float | None
    upper_price: float | None
    probability: float


@dataclass
class DirectionalPrediction:
    date: str
    predicted_return: float
    predicted_close: float
    actual_close: float
    open_price: float
    sigma_return: float
    probability_up: float
    probability_down: float
    event_risk_flag: bool
    probability_bands: List[ProbabilityBand]
    abs_error: float
    signed_error: float
    metadata: Dict[str, float]


@dataclass
class FitResult:
    model: Pipeline
    direction_calibrator: LogisticRegression
    sigma_return: float
    train_rows: List[FeatureRow]
    backtest_rows: List[FeatureRow]
    input_feature_count: int
    reduced_feature_count: int
    explained_variance_ratio: float


@dataclass
class ClassifierFitResult:
    model: Pipeline
    train_rows: List[FeatureRow]
    backtest_rows: List[FeatureRow]
    input_feature_count: int
    reduced_feature_count: int
    explained_variance_ratio: float


@dataclass
class RangePrediction:
    date: str
    open_price: float
    predicted_high_return: float
    predicted_low_return: float
    actual_close_return: float
    actual_high_return: float
    actual_low_return: float
    predicted_high: float
    predicted_low: float
    sigma_high_return: float
    sigma_low_return: float
    actual_high: float
    actual_low: float
    high_abs_error: float
    low_abs_error: float
    high_signed_error: float
    low_signed_error: float
    event_risk_flag: bool
    metadata: Dict[str, float]


@dataclass
class RangeFitResult:
    high_model: Pipeline
    low_model: Pipeline
    sigma_high_return: float
    sigma_low_return: float
    train_rows: List[FeatureRow]
    backtest_rows: List[FeatureRow]
    input_feature_count: int
    high_reduced_feature_count: int
    high_explained_variance_ratio: float
    low_reduced_feature_count: int
    low_explained_variance_ratio: float


@dataclass
class TwoStageFitResult:
    move_model: Pipeline
    direction_model: Pipeline
    train_rows: List[FeatureRow]
    backtest_rows: List[FeatureRow]
    move_threshold: float
    input_feature_count: int
    move_reduced_feature_count: int
    move_explained_variance_ratio: float
    direction_reduced_feature_count: int
    direction_explained_variance_ratio: float


def _make_pipeline(
    ridge_lambda: float,
    pca_components: int | None = None,
    pca_variance_ratio: float | None = None,
) -> Pipeline:
    steps: List[tuple[str, object]] = [("scaler", StandardScaler())]
    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components)))
    elif pca_variance_ratio is not None and 0.0 < pca_variance_ratio < 1.0:
        steps.append(("pca", PCA(n_components=pca_variance_ratio)))
    steps.append(("regressor", Ridge(alpha=ridge_lambda)))
    return Pipeline(steps=steps)


def _make_classifier_pipeline(
    pca_components: int | None = None,
    pca_variance_ratio: float | None = None,
) -> Pipeline:
    steps: List[tuple[str, object]] = [("scaler", StandardScaler())]
    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components)))
    elif pca_variance_ratio is not None and 0.0 < pca_variance_ratio < 1.0:
        steps.append(("pca", PCA(n_components=pca_variance_ratio)))
    steps.append(("classifier", LogisticRegression(max_iter=5000, C=0.2)))
    return Pipeline(steps=steps)


def _extract_reduction_stats(model: Pipeline, x_train: np.ndarray) -> tuple[int, int, float]:
    input_feature_count = int(x_train.shape[1])
    reduced_feature_count = input_feature_count
    explained_variance_ratio = 1.0
    pca = model.named_steps.get("pca")
    if pca is not None:
        reduced_feature_count = int(getattr(pca, "n_components_", input_feature_count))
        explained_variance_ratio = float(np.sum(getattr(pca, "explained_variance_ratio_", [1.0])))
    return input_feature_count, reduced_feature_count, explained_variance_ratio


def split_train_backtest(
    rows: Sequence[FeatureRow],
    train_end_date: str | None = None,
    train_ratio: float = 0.75,
) -> Tuple[List[FeatureRow], List[FeatureRow]]:
    if not rows:
        return [], []
    if train_end_date is not None:
        cutoff = date.fromisoformat(train_end_date)
        train = [row for row in rows if row.date <= cutoff]
        backtest = [row for row in rows if row.date > cutoff]
        if train and backtest:
            return train, backtest
    split_idx = max(1, int(len(rows) * train_ratio))
    split_idx = min(split_idx, len(rows) - 1)
    return list(rows[:split_idx]), list(rows[split_idx:])


def fit_train_backtest_model(
    rows: Sequence[FeatureRow],
    ridge_lambda: float = 1.0,
    train_end_date: str | None = None,
    train_ratio: float = 0.75,
    pca_components: int | None = None,
    pca_variance_ratio: float | None = 0.95,
) -> FitResult:
    train_rows, backtest_rows = split_train_backtest(rows, train_end_date=train_end_date, train_ratio=train_ratio)
    if not train_rows or not backtest_rows:
        raise ValueError("Need non-empty train and backtest partitions")
    x_train = np.asarray([row.values for row in train_rows], dtype=float)
    y_train = np.asarray([row.target_return for row in train_rows], dtype=float)
    model = _make_pipeline(
        ridge_lambda,
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    model.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    y_direction = np.asarray([1 if row.actual_close >= row.open_price else 0 for row in train_rows], dtype=int)
    direction_calibrator = LogisticRegression(max_iter=5000, C=1.0)
    direction_calibrator.fit(train_preds.reshape(-1, 1), y_direction)
    sigma_return = float(np.std(y_train - train_preds, ddof=1)) if len(train_rows) > 1 else 1e-4
    sigma_return = max(sigma_return, 1e-4)
    input_feature_count, reduced_feature_count, explained_variance_ratio = _extract_reduction_stats(model, x_train)
    return FitResult(
        model=model,
        direction_calibrator=direction_calibrator,
        sigma_return=sigma_return,
        train_rows=list(train_rows),
        backtest_rows=list(backtest_rows),
        input_feature_count=input_feature_count,
        reduced_feature_count=reduced_feature_count,
        explained_variance_ratio=explained_variance_ratio,
    )


def fit_direction_classifier(
    rows: Sequence[FeatureRow],
    train_end_date: str | None = None,
    train_ratio: float = 0.75,
    pca_components: int | None = None,
    pca_variance_ratio: float | None = 0.95,
) -> ClassifierFitResult:
    train_rows, backtest_rows = split_train_backtest(rows, train_end_date=train_end_date, train_ratio=train_ratio)
    if not train_rows or not backtest_rows:
        raise ValueError("Need non-empty train and backtest partitions")
    x_train = np.asarray([row.values for row in train_rows], dtype=float)
    y_train = np.asarray([1 if row.actual_close >= row.open_price else 0 for row in train_rows], dtype=int)
    model = _make_classifier_pipeline(
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    model.fit(x_train, y_train)
    input_feature_count, reduced_feature_count, explained_variance_ratio = _extract_reduction_stats(model, x_train)
    return ClassifierFitResult(
        model=model,
        train_rows=list(train_rows),
        backtest_rows=list(backtest_rows),
        input_feature_count=input_feature_count,
        reduced_feature_count=reduced_feature_count,
        explained_variance_ratio=explained_variance_ratio,
    )


def fit_train_backtest_range_model(
    rows: Sequence[FeatureRow],
    ridge_lambda: float = 1.0,
    train_end_date: str | None = None,
    train_ratio: float = 0.75,
    pca_components: int | None = None,
    pca_variance_ratio: float | None = 0.95,
) -> RangeFitResult:
    train_rows, backtest_rows = split_train_backtest(rows, train_end_date=train_end_date, train_ratio=train_ratio)
    if not train_rows or not backtest_rows:
        raise ValueError("Need non-empty train and backtest partitions")
    x_train = np.asarray([row.values for row in train_rows], dtype=float)
    y_high = np.asarray([row.target_high_return for row in train_rows], dtype=float)
    y_low = np.asarray([row.target_low_return for row in train_rows], dtype=float)

    high_model = _make_pipeline(
        ridge_lambda,
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    low_model = _make_pipeline(
        ridge_lambda,
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    high_model.fit(x_train, y_high)
    low_model.fit(x_train, y_low)
    high_train_preds = high_model.predict(x_train)
    low_train_preds = low_model.predict(x_train)
    sigma_high_return = float(np.std(y_high - high_train_preds, ddof=1)) if len(train_rows) > 1 else 1e-4
    sigma_low_return = float(np.std(y_low - low_train_preds, ddof=1)) if len(train_rows) > 1 else 1e-4
    sigma_high_return = max(sigma_high_return, 1e-4)
    sigma_low_return = max(sigma_low_return, 1e-4)
    input_feature_count, high_reduced_feature_count, high_explained_variance_ratio = _extract_reduction_stats(high_model, x_train)
    _, low_reduced_feature_count, low_explained_variance_ratio = _extract_reduction_stats(low_model, x_train)
    return RangeFitResult(
        high_model=high_model,
        low_model=low_model,
        sigma_high_return=sigma_high_return,
        sigma_low_return=sigma_low_return,
        train_rows=list(train_rows),
        backtest_rows=list(backtest_rows),
        input_feature_count=input_feature_count,
        high_reduced_feature_count=high_reduced_feature_count,
        high_explained_variance_ratio=high_explained_variance_ratio,
        low_reduced_feature_count=low_reduced_feature_count,
        low_explained_variance_ratio=low_explained_variance_ratio,
    )


def fit_two_stage_classifier(
    rows: Sequence[FeatureRow],
    move_threshold: float,
    train_end_date: str | None = None,
    train_ratio: float = 0.75,
    pca_components: int | None = None,
    pca_variance_ratio: float | None = 0.95,
) -> TwoStageFitResult:
    train_rows, backtest_rows = split_train_backtest(rows, train_end_date=train_end_date, train_ratio=train_ratio)
    if not train_rows or not backtest_rows:
        raise ValueError("Need non-empty train and backtest partitions")

    x_train = np.asarray([row.values for row in train_rows], dtype=float)
    y_move = np.asarray([1 if abs(row.target_return) >= move_threshold else 0 for row in train_rows], dtype=int)
    if y_move.min() == y_move.max():
        raise ValueError("Move threshold creates a degenerate training target")

    move_model = _make_classifier_pipeline(
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    move_model.fit(x_train, y_move)
    input_feature_count, move_reduced_feature_count, move_explained_variance_ratio = _extract_reduction_stats(move_model, x_train)

    directional_rows = [row for row in train_rows if abs(row.target_return) >= move_threshold]
    if len(directional_rows) < 20:
        raise ValueError("Not enough large-move training rows for second-stage direction model")
    x_dir = np.asarray([row.values for row in directional_rows], dtype=float)
    y_dir = np.asarray([1 if row.actual_close >= row.open_price else 0 for row in directional_rows], dtype=int)
    if y_dir.min() == y_dir.max():
        raise ValueError("Directional target is degenerate for second-stage model")

    direction_model = _make_classifier_pipeline(
        pca_components=pca_components,
        pca_variance_ratio=pca_variance_ratio,
    )
    direction_model.fit(x_dir, y_dir)
    _, direction_reduced_feature_count, direction_explained_variance_ratio = _extract_reduction_stats(direction_model, x_dir)

    return TwoStageFitResult(
        move_model=move_model,
        direction_model=direction_model,
        train_rows=list(train_rows),
        backtest_rows=list(backtest_rows),
        move_threshold=move_threshold,
        input_feature_count=input_feature_count,
        move_reduced_feature_count=move_reduced_feature_count,
        move_explained_variance_ratio=move_explained_variance_ratio,
        direction_reduced_feature_count=direction_reduced_feature_count,
        direction_explained_variance_ratio=direction_explained_variance_ratio,
    )


def _make_probability_bands(open_price: float, predicted_return: float, sigma_return: float) -> List[ProbabilityBand]:
    dist = NormalDist(mu=predicted_return, sigma=max(sigma_return, 1e-6))
    edges = [
        ("tail_down", None, predicted_return - sigma_return),
        ("downside", predicted_return - sigma_return, predicted_return - 0.25 * sigma_return),
        ("center", predicted_return - 0.25 * sigma_return, predicted_return + 0.25 * sigma_return),
        ("upside", predicted_return + 0.25 * sigma_return, predicted_return + sigma_return),
        ("tail_up", predicted_return + sigma_return, None),
    ]
    bands: List[ProbabilityBand] = []
    for label, lower_return, upper_return in edges:
        lower_cdf = dist.cdf(lower_return) if lower_return is not None else 0.0
        upper_cdf = dist.cdf(upper_return) if upper_return is not None else 1.0
        bands.append(
            ProbabilityBand(
                label=label,
                lower_price=open_price * (1.0 + lower_return) if lower_return is not None else None,
                upper_price=open_price * (1.0 + upper_return) if upper_return is not None else None,
                probability=max(upper_cdf - lower_cdf, 0.0),
            )
        )
    return bands


def predict_backtest(
    fit_result: FitResult,
    event_vol_multiplier: float = 1.5,
) -> List[DirectionalPrediction]:
    predictions: List[DirectionalPrediction] = []
    for row in fit_result.backtest_rows:
        predicted_return = float(fit_result.model.predict(np.asarray([row.values], dtype=float))[0])
        sigma_return = fit_result.sigma_return
        event_risk_flag = bool(row.metadata.get("event_any", 0.0))
        if event_risk_flag:
            sigma_return *= event_vol_multiplier
        probability_up = float(fit_result.direction_calibrator.predict_proba(np.asarray([[predicted_return]], dtype=float))[0][1])
        probability_down = 1.0 - probability_up
        predicted_close = row.open_price * (1.0 + predicted_return)
        error = predicted_close - row.actual_close
        predictions.append(
            DirectionalPrediction(
                date=row.date.isoformat(),
                predicted_return=predicted_return,
                predicted_close=predicted_close,
                actual_close=row.actual_close,
                open_price=row.open_price,
                sigma_return=sigma_return,
                probability_up=probability_up,
                probability_down=probability_down,
                event_risk_flag=event_risk_flag,
                probability_bands=_make_probability_bands(row.open_price, predicted_return, sigma_return),
                abs_error=abs(error),
                signed_error=error,
                metadata=row.metadata,
            )
        )
    return predictions


def predict_backtest_classifier(
    fit_result: ClassifierFitResult,
    sigma_return: float,
    event_vol_multiplier: float = 1.5,
) -> List[DirectionalPrediction]:
    predictions: List[DirectionalPrediction] = []
    for row in fit_result.backtest_rows:
        probability_up = float(fit_result.model.predict_proba(np.asarray([row.values], dtype=float))[0][1])
        probability_down = 1.0 - probability_up
        adjusted_sigma = sigma_return
        event_risk_flag = bool(row.metadata.get("event_any", 0.0))
        if event_risk_flag:
            adjusted_sigma *= event_vol_multiplier
        predicted_return = (probability_up - 0.5) * 2.0 * adjusted_sigma
        predicted_close = row.open_price * (1.0 + predicted_return)
        error = predicted_close - row.actual_close
        predictions.append(
            DirectionalPrediction(
                date=row.date.isoformat(),
                predicted_return=predicted_return,
                predicted_close=predicted_close,
                actual_close=row.actual_close,
                open_price=row.open_price,
                sigma_return=adjusted_sigma,
                probability_up=probability_up,
                probability_down=probability_down,
                event_risk_flag=event_risk_flag,
                probability_bands=_make_probability_bands(row.open_price, predicted_return, adjusted_sigma),
                abs_error=abs(error),
                signed_error=error,
                metadata=row.metadata,
            )
        )
    return predictions


def predict_backtest_range(
    fit_result: RangeFitResult,
) -> List[RangePrediction]:
    predictions: List[RangePrediction] = []
    for row in fit_result.backtest_rows:
        features = np.asarray([row.values], dtype=float)
        predicted_high_return = float(fit_result.high_model.predict(features)[0])
        predicted_low_return = float(fit_result.low_model.predict(features)[0])
        predicted_high_return = max(predicted_high_return, 0.0)
        predicted_low_return = min(predicted_low_return, 0.0)
        predicted_high = row.open_price * (1.0 + predicted_high_return)
        predicted_low = row.open_price * (1.0 + predicted_low_return)
        high_error = predicted_high - row.high_price
        low_error = predicted_low - row.low_price
        predictions.append(
            RangePrediction(
                date=row.date.isoformat(),
                open_price=row.open_price,
                predicted_high_return=predicted_high_return,
                predicted_low_return=predicted_low_return,
                actual_close_return=row.target_return,
                actual_high_return=row.target_high_return,
                actual_low_return=row.target_low_return,
                predicted_high=predicted_high,
                predicted_low=predicted_low,
                sigma_high_return=fit_result.sigma_high_return,
                sigma_low_return=fit_result.sigma_low_return,
                actual_high=row.high_price,
                actual_low=row.low_price,
                high_abs_error=abs(high_error),
                low_abs_error=abs(low_error),
                high_signed_error=high_error,
                low_signed_error=low_error,
                event_risk_flag=bool(row.metadata.get("event_any", 0.0)),
                metadata=row.metadata,
            )
        )
    return predictions


def predict_backtest_two_stage(
    fit_result: TwoStageFitResult,
    sigma_return: float,
    event_vol_multiplier: float = 1.5,
) -> List[DirectionalPrediction]:
    predictions: List[DirectionalPrediction] = []
    for row in fit_result.backtest_rows:
        features = np.asarray([row.values], dtype=float)
        probability_move = float(fit_result.move_model.predict_proba(features)[0][1])
        probability_up_given_move = float(fit_result.direction_model.predict_proba(features)[0][1])
        probability_up = (probability_move * probability_up_given_move) + ((1.0 - probability_move) * 0.5)
        probability_down = 1.0 - probability_up
        adjusted_sigma = sigma_return
        event_risk_flag = bool(row.metadata.get("event_any", 0.0))
        if event_risk_flag:
            adjusted_sigma *= event_vol_multiplier
        signed_move_strength = (probability_up_given_move - 0.5) * 2.0
        predicted_return = probability_move * signed_move_strength * fit_result.move_threshold
        predicted_close = row.open_price * (1.0 + predicted_return)
        error = predicted_close - row.actual_close
        metadata = dict(row.metadata)
        metadata["probability_move"] = probability_move
        metadata["probability_up_given_move"] = probability_up_given_move
        metadata["move_threshold"] = fit_result.move_threshold
        predictions.append(
            DirectionalPrediction(
                date=row.date.isoformat(),
                predicted_return=predicted_return,
                predicted_close=predicted_close,
                actual_close=row.actual_close,
                open_price=row.open_price,
                sigma_return=adjusted_sigma,
                probability_up=probability_up,
                probability_down=probability_down,
                event_risk_flag=event_risk_flag,
                probability_bands=_make_probability_bands(row.open_price, predicted_return, adjusted_sigma),
                abs_error=abs(error),
                signed_error=error,
                metadata=metadata,
            )
        )
    return predictions


def regression_metrics(predictions: Sequence[DirectionalPrediction]) -> Dict[str, float]:
    if not predictions:
        return {}
    mae = sum(item.abs_error for item in predictions) / len(predictions)
    mse = sum(item.signed_error ** 2 for item in predictions) / len(predictions)
    rmse = mse ** 0.5
    actuals = [item.actual_close for item in predictions]
    mean_actual = sum(actuals) / len(actuals)
    ss_tot = sum((value - mean_actual) ** 2 for value in actuals)
    ss_res = sum(item.signed_error ** 2 for item in predictions)
    directional_hits = 0
    for item in predictions:
        predicted_sign = 1 if item.probability_up >= 0.5 else -1
        actual_sign = 1 if item.actual_close >= item.open_price else -1
        directional_hits += 1 if predicted_sign == actual_sign else 0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": 1.0 - (ss_res / ss_tot) if ss_tot else 0.0,
        "directional_accuracy": directional_hits / len(predictions),
        "samples": float(len(predictions)),
    }


def metrics_by_flag(predictions: Sequence[DirectionalPrediction], flag_name: str) -> Dict[str, Dict[str, float]]:
    flagged = [item for item in predictions if item.metadata.get(flag_name, 0.0) == 1.0]
    unflagged = [item for item in predictions if item.metadata.get(flag_name, 0.0) == 0.0]
    return {
        "flagged": regression_metrics(flagged),
        "unflagged": regression_metrics(unflagged),
    }


def directional_metrics(predictions: Sequence[DirectionalPrediction], confidence_threshold: float = 0.70) -> Dict[str, float]:
    if not predictions:
        return {}
    actuals = np.asarray([1 if item.actual_close >= item.open_price else 0 for item in predictions], dtype=int)
    probabilities = np.asarray([item.probability_up for item in predictions], dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    accuracy = float(np.mean(predicted == actuals))
    brier = float(np.mean((probabilities - actuals) ** 2))
    base_rate = float(np.mean(actuals))
    auc = 0.0
    pos = probabilities[actuals == 1]
    neg = probabilities[actuals == 0]
    if len(pos) and len(neg):
        wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
        auc = wins / (len(pos) * len(neg))
    confident_mask = (probabilities >= confidence_threshold) | (probabilities <= 1.0 - confidence_threshold)
    confident_accuracy = float(np.mean(predicted[confident_mask] == actuals[confident_mask])) if np.any(confident_mask) else 0.0
    return {
        "samples": float(len(predictions)),
        "base_rate_up": base_rate,
        "accuracy": accuracy,
        "auc": auc,
        "brier": brier,
        "confidence_coverage": float(np.mean(confident_mask)),
        "confidence_accuracy": confident_accuracy,
    }


def range_metrics(predictions: Sequence[RangePrediction]) -> Dict[str, float]:
    if not predictions:
        return {}
    high_mae = float(np.mean([item.high_abs_error for item in predictions]))
    low_mae = float(np.mean([item.low_abs_error for item in predictions]))
    high_rmse = float(np.sqrt(np.mean([item.high_signed_error ** 2 for item in predictions])))
    low_rmse = float(np.sqrt(np.mean([item.low_signed_error ** 2 for item in predictions])))
    containment = float(
        np.mean(
            [
                1.0 if (item.predicted_high >= item.actual_high and item.predicted_low <= item.actual_low) else 0.0
                for item in predictions
            ]
        )
    )
    average_predicted_range = float(np.mean([item.predicted_high - item.predicted_low for item in predictions]))
    average_actual_range = float(np.mean([item.actual_high - item.actual_low for item in predictions]))
    high_return_mae = float(np.mean([abs(item.predicted_high_return - item.actual_high_return) for item in predictions]))
    low_return_mae = float(np.mean([abs(item.predicted_low_return - item.actual_low_return) for item in predictions]))
    return {
        "samples": float(len(predictions)),
        "high_mae": high_mae,
        "low_mae": low_mae,
        "high_rmse": high_rmse,
        "low_rmse": low_rmse,
        "high_return_mae": high_return_mae,
        "low_return_mae": low_return_mae,
        "range_containment": containment,
        "avg_predicted_range": average_predicted_range,
        "avg_actual_range": average_actual_range,
    }


def excursion_threshold_metrics(
    predictions: Sequence[RangePrediction],
    thresholds: Sequence[float],
) -> Dict[str, List[Dict[str, float | str]]]:
    if not predictions:
        return {"upside": [], "downside": []}
    upside_rows: List[Dict[str, float | str]] = []
    downside_rows: List[Dict[str, float | str]] = []
    for threshold in thresholds:
        predicted_up = np.asarray([item.predicted_high_return >= threshold for item in predictions], dtype=bool)
        actual_up = np.asarray([item.actual_high_return >= threshold for item in predictions], dtype=bool)
        predicted_down = np.asarray([item.predicted_low_return <= -threshold for item in predictions], dtype=bool)
        actual_down = np.asarray([item.actual_low_return <= -threshold for item in predictions], dtype=bool)

        def _row(side: str, predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float | str]:
            tp = float(np.sum(predicted & actual))
            fp = float(np.sum(predicted & ~actual))
            fn = float(np.sum(~predicted & actual))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            coverage = float(np.mean(predicted))
            actual_rate = float(np.mean(actual))
            return {
                "side": side,
                "threshold_return": threshold,
                "threshold_pct": threshold * 100.0,
                "coverage": coverage,
                "actual_rate": actual_rate,
                "precision": precision,
                "recall": recall,
                "samples": float(len(predictions)),
            }

        upside_rows.append(_row("upside", predicted_up, actual_up))
        downside_rows.append(_row("downside", predicted_down, actual_down))
    return {"upside": upside_rows, "downside": downside_rows}


def excursion_probability_backtest(
    predictions: Sequence[RangePrediction],
    thresholds: Sequence[float],
) -> Dict[str, List[Dict[str, float | str]]]:
    if not predictions:
        return {"upside": [], "downside": []}

    def _auc(probabilities: np.ndarray, actuals: np.ndarray) -> float:
        pos = probabilities[actuals == 1]
        neg = probabilities[actuals == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    upside_rows: List[Dict[str, float | str]] = []
    downside_rows: List[Dict[str, float | str]] = []
    for threshold in thresholds:
        upside_probabilities = np.asarray(
            [
                1.0 - NormalDist(mu=item.predicted_high_return, sigma=item.sigma_high_return).cdf(threshold)
                for item in predictions
            ],
            dtype=float,
        )
        downside_probabilities = np.asarray(
            [
                NormalDist(mu=item.predicted_low_return, sigma=item.sigma_low_return).cdf(-threshold)
                for item in predictions
            ],
            dtype=float,
        )
        upside_actuals = np.asarray([1 if item.actual_high_return >= threshold else 0 for item in predictions], dtype=int)
        downside_actuals = np.asarray([1 if item.actual_low_return <= -threshold else 0 for item in predictions], dtype=int)

        def _row(side: str, probabilities: np.ndarray, actuals: np.ndarray) -> Dict[str, float | str]:
            predicted = probabilities >= 0.5
            tp = float(np.sum(predicted & (actuals == 1)))
            fp = float(np.sum(predicted & (actuals == 0)))
            fn = float(np.sum((~predicted) & (actuals == 1)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            return {
                "side": side,
                "threshold_return": threshold,
                "threshold_pct": threshold * 100.0,
                "samples": float(len(predictions)),
                "avg_predicted_probability": float(np.mean(probabilities)),
                "actual_rate": float(np.mean(actuals)),
                "brier": float(np.mean((probabilities - actuals) ** 2)),
                "auc": _auc(probabilities, actuals),
                "precision_at_50": precision,
                "recall_at_50": recall,
                "coverage_at_50": float(np.mean(predicted)),
            }

        upside_rows.append(_row("upside", upside_probabilities, upside_actuals))
        downside_rows.append(_row("downside", downside_probabilities, downside_actuals))
    return {"upside": upside_rows, "downside": downside_rows}


def conditional_close_given_touch(
    predictions: Sequence[RangePrediction],
    thresholds: Sequence[float],
) -> Dict[str, List[Dict[str, float | str]]]:
    if not predictions:
        return {"upside": [], "downside": []}

    def _quantiles(values: np.ndarray) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0
        return float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))

    def _build_row(side: str, threshold: float) -> Dict[str, float | str]:
        if side == "upside":
            touched = [item for item in predictions if item.actual_high_return >= threshold]
            if not touched:
                return {
                    "side": side,
                    "threshold_return": threshold,
                    "threshold_pct": threshold * 100.0,
                    "samples": 0.0,
                    "touch_rate": 0.0,
                    "avg_close_return": 0.0,
                    "median_close_return": 0.0,
                    "close_return_q25": 0.0,
                    "close_return_q75": 0.0,
                    "avg_finish_vs_touch": 0.0,
                    "avg_extension_beyond_touch": 0.0,
                    "close_above_open_rate": 0.0,
                    "close_back_below_open_rate": 0.0,
                    "close_above_touch_rate": 0.0,
                }
            close_returns = np.asarray([item.actual_close_return for item in touched], dtype=float)
            high_returns = np.asarray([item.actual_high_return for item in touched], dtype=float)
            close_return_q25, close_return_q75 = _quantiles(close_returns)
            avg_finish_vs_touch = float(np.mean(close_returns - threshold))
            avg_extension_beyond_touch = float(np.mean(high_returns - threshold))
            return {
                "side": side,
                "threshold_return": threshold,
                "threshold_pct": threshold * 100.0,
                "samples": float(len(touched)),
                "touch_rate": float(len(touched) / len(predictions)),
                "avg_close_return": float(np.mean(close_returns)),
                "median_close_return": float(np.median(close_returns)),
                "close_return_q25": close_return_q25,
                "close_return_q75": close_return_q75,
                "avg_finish_vs_touch": avg_finish_vs_touch,
                "avg_extension_beyond_touch": avg_extension_beyond_touch,
                "close_above_open_rate": float(np.mean(close_returns >= 0.0)),
                "close_back_below_open_rate": float(np.mean(close_returns < 0.0)),
                "close_above_touch_rate": float(np.mean(close_returns >= threshold)),
            }

        touched = [item for item in predictions if item.actual_low_return <= -threshold]
        if not touched:
            return {
                "side": side,
                "threshold_return": threshold,
                "threshold_pct": threshold * 100.0,
                "samples": 0.0,
                "touch_rate": 0.0,
                "avg_close_return": 0.0,
                "median_close_return": 0.0,
                "close_return_q25": 0.0,
                "close_return_q75": 0.0,
                "avg_finish_vs_touch": 0.0,
                "avg_extension_beyond_touch": 0.0,
                "close_below_open_rate": 0.0,
                "close_back_above_open_rate": 0.0,
                "close_below_touch_rate": 0.0,
            }
        close_returns = np.asarray([item.actual_close_return for item in touched], dtype=float)
        low_returns = np.asarray([item.actual_low_return for item in touched], dtype=float)
        close_return_q25, close_return_q75 = _quantiles(close_returns)
        avg_finish_vs_touch = float(np.mean(close_returns + threshold))
        avg_extension_beyond_touch = float(np.mean((-threshold) - low_returns))
        return {
            "side": side,
            "threshold_return": threshold,
            "threshold_pct": threshold * 100.0,
            "samples": float(len(touched)),
            "touch_rate": float(len(touched) / len(predictions)),
            "avg_close_return": float(np.mean(close_returns)),
            "median_close_return": float(np.median(close_returns)),
            "close_return_q25": close_return_q25,
            "close_return_q75": close_return_q75,
            "avg_finish_vs_touch": avg_finish_vs_touch,
            "avg_extension_beyond_touch": avg_extension_beyond_touch,
            "close_below_open_rate": float(np.mean(close_returns <= 0.0)),
            "close_back_above_open_rate": float(np.mean(close_returns > 0.0)),
            "close_below_touch_rate": float(np.mean(close_returns <= -threshold)),
        }

    return {
        "upside": [_build_row("upside", threshold) for threshold in thresholds],
        "downside": [_build_row("downside", threshold) for threshold in thresholds],
    }


def conditional_close_given_touch_by_regime(
    train_rows: Sequence[FeatureRow],
    backtest_rows: Sequence[FeatureRow],
    predictions: Sequence[RangePrediction],
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, List[Dict[str, float | str]]]]:
    if not train_rows or not backtest_rows or not predictions:
        return {"upside": {}, "downside": {}}

    name_index = {name: idx for idx, name in enumerate(train_rows[0].feature_names)}

    def _feature(row: FeatureRow, name: str) -> float:
        return row.values[name_index[name]]

    train_vix = np.asarray([_feature(row, "current_vix_open") for row in train_rows], dtype=float)
    train_prev_range = np.asarray([_feature(row, "spx_range_pct_lag_1") for row in train_rows], dtype=float)
    train_abs_gap = np.asarray([abs(_feature(row, "current_spx_overnight_gap")) for row in train_rows], dtype=float)

    vix_low_upper, vix_mid_upper = np.quantile(train_vix, [1.0 / 3.0, 2.0 / 3.0])
    range_low_upper, range_mid_upper = np.quantile(train_prev_range, [1.0 / 3.0, 2.0 / 3.0])
    gap_flat_upper = float(np.quantile(train_abs_gap, 0.25))
    gap_small_upper = float(np.quantile(train_abs_gap, 0.60))
    gap_medium_upper = float(np.quantile(train_abs_gap, 0.85))

    enriched: List[Dict[str, float | str]] = []
    for row, prediction in zip(backtest_rows, predictions):
        current_vix_open = _feature(row, "current_vix_open")
        prev_range_pct = _feature(row, "spx_range_pct_lag_1")
        abs_gap = abs(_feature(row, "current_spx_overnight_gap"))
        if current_vix_open <= vix_low_upper:
            vix_regime = "low_vix"
        elif current_vix_open <= vix_mid_upper:
            vix_regime = "mid_vix"
        else:
            vix_regime = "high_vix"
        if prev_range_pct <= range_low_upper:
            range_regime = "low_prev_range"
        elif prev_range_pct <= range_mid_upper:
            range_regime = "mid_prev_range"
        else:
            range_regime = "high_prev_range"
        if abs_gap <= gap_flat_upper:
            gap_regime = "flat_gap"
        elif abs_gap <= gap_small_upper:
            gap_regime = "small_gap"
        elif abs_gap <= gap_medium_upper:
            gap_regime = "medium_gap"
        else:
            gap_regime = "large_gap"
        enriched.append(
            {
                "vix_regime": vix_regime,
                "range_regime": range_regime,
                "gap_regime": gap_regime,
                "combo_regime": f"{vix_regime}|{range_regime}|{gap_regime}",
                "actual_high_return": prediction.actual_high_return,
                "actual_low_return": prediction.actual_low_return,
                "actual_close_return": prediction.actual_close_return,
            }
        )

    def _summarize(side: str, threshold: float, regime_name: str, regime_value: str) -> Dict[str, float | str]:
        subset = [row for row in enriched if row[regime_name] == regime_value]
        if side == "upside":
            touched = [row for row in subset if row["actual_high_return"] >= threshold]
            close_returns = np.asarray([float(row["actual_close_return"]) for row in touched], dtype=float)
            close_return_q25 = float(np.quantile(close_returns, 0.25)) if len(close_returns) else 0.0
            close_return_q75 = float(np.quantile(close_returns, 0.75)) if len(close_returns) else 0.0
            return {
                "regime_family": regime_name,
                "regime_value": regime_value,
                "threshold_return": threshold,
                "threshold_pct": threshold * 100.0,
                "samples": float(len(touched)),
                "regime_days": float(len(subset)),
                "touch_rate_within_regime": float(len(touched) / len(subset)) if subset else 0.0,
                "avg_close_return": float(np.mean(close_returns)) if len(close_returns) else 0.0,
                "median_close_return": float(np.median(close_returns)) if len(close_returns) else 0.0,
                "close_return_q25": close_return_q25,
                "close_return_q75": close_return_q75,
                "close_above_open_rate": float(np.mean(close_returns >= 0.0)) if len(close_returns) else 0.0,
                "close_above_touch_rate": float(np.mean(close_returns >= threshold)) if len(close_returns) else 0.0,
                "close_back_below_open_rate": float(np.mean(close_returns < 0.0)) if len(close_returns) else 0.0,
            }
        touched = [row for row in subset if row["actual_low_return"] <= -threshold]
        close_returns = np.asarray([float(row["actual_close_return"]) for row in touched], dtype=float)
        close_return_q25 = float(np.quantile(close_returns, 0.25)) if len(close_returns) else 0.0
        close_return_q75 = float(np.quantile(close_returns, 0.75)) if len(close_returns) else 0.0
        return {
            "regime_family": regime_name,
            "regime_value": regime_value,
            "threshold_return": threshold,
            "threshold_pct": threshold * 100.0,
            "samples": float(len(touched)),
            "regime_days": float(len(subset)),
            "touch_rate_within_regime": float(len(touched) / len(subset)) if subset else 0.0,
            "avg_close_return": float(np.mean(close_returns)) if len(close_returns) else 0.0,
            "median_close_return": float(np.median(close_returns)) if len(close_returns) else 0.0,
            "close_return_q25": close_return_q25,
            "close_return_q75": close_return_q75,
            "close_below_open_rate": float(np.mean(close_returns <= 0.0)) if len(close_returns) else 0.0,
            "close_below_touch_rate": float(np.mean(close_returns <= -threshold)) if len(close_returns) else 0.0,
            "close_back_above_open_rate": float(np.mean(close_returns > 0.0)) if len(close_returns) else 0.0,
        }

    regime_buckets = {
        "vix_regime": ["low_vix", "mid_vix", "high_vix"],
        "range_regime": ["low_prev_range", "mid_prev_range", "high_prev_range"],
        "gap_regime": ["flat_gap", "small_gap", "medium_gap", "large_gap"],
        "combo_regime": sorted({str(row["combo_regime"]) for row in enriched}),
    }
    output: Dict[str, Dict[str, List[Dict[str, float | str]]]] = {"upside": {}, "downside": {}}
    for side in ["upside", "downside"]:
        for regime_name, regime_values in regime_buckets.items():
            rows: List[Dict[str, float | str]] = []
            for threshold in thresholds:
                for regime_value in regime_values:
                    rows.append(_summarize(side, threshold, regime_name, regime_value))
            output[side][regime_name] = rows
    return output
