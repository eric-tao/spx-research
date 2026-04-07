from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .model import DirectionalPrediction


@dataclass
class DebitSpreadConfig:
    width: float = 5.0
    premium: float = 3.0
    confidence_threshold: float = 0.70


@dataclass
class StrategyTrade:
    date: str
    direction: str
    long_strike: float
    short_strike: float
    premium: float
    width: float
    probability_up: float
    probability_down: float
    predicted_close: float
    actual_close: float
    pnl: float


def _bull_call_payoff(close_price: float, long_strike: float, short_strike: float, premium: float) -> float:
    intrinsic = max(min(close_price - long_strike, short_strike - long_strike), 0.0)
    return intrinsic - premium


def _bear_put_payoff(close_price: float, long_strike: float, short_strike: float, premium: float) -> float:
    intrinsic = max(min(long_strike - close_price, long_strike - short_strike), 0.0)
    return intrinsic - premium


def _bull_call_strikes(open_price: float, config: DebitSpreadConfig) -> tuple[float, float]:
    half_width = config.width / 2.0
    long_strike = open_price - half_width
    short_strike = open_price + half_width
    return long_strike, short_strike


def _bear_put_strikes(open_price: float, config: DebitSpreadConfig) -> tuple[float, float]:
    half_width = config.width / 2.0
    long_strike = open_price + half_width
    short_strike = open_price - half_width
    return long_strike, short_strike


def trade_from_prediction(
    prediction: DirectionalPrediction,
    config: DebitSpreadConfig,
) -> StrategyTrade | None:
    if prediction.probability_up >= config.confidence_threshold:
        long_strike, short_strike = _bull_call_strikes(prediction.open_price, config)
        pnl = _bull_call_payoff(prediction.actual_close, long_strike, short_strike, config.premium)
        return StrategyTrade(
            date=prediction.date,
            direction="bull_call_debit",
            long_strike=long_strike,
            short_strike=short_strike,
            premium=config.premium,
            width=config.width,
            probability_up=prediction.probability_up,
            probability_down=prediction.probability_down,
            predicted_close=prediction.predicted_close,
            actual_close=prediction.actual_close,
            pnl=pnl,
        )
    if prediction.probability_down >= config.confidence_threshold:
        long_strike, short_strike = _bear_put_strikes(prediction.open_price, config)
        pnl = _bear_put_payoff(prediction.actual_close, long_strike, short_strike, config.premium)
        return StrategyTrade(
            date=prediction.date,
            direction="bear_put_debit",
            long_strike=long_strike,
            short_strike=short_strike,
            premium=config.premium,
            width=config.width,
            probability_up=prediction.probability_up,
            probability_down=prediction.probability_down,
            predicted_close=prediction.predicted_close,
            actual_close=prediction.actual_close,
            pnl=pnl,
        )
    return None


def backtest_strategy(
    predictions: Sequence[DirectionalPrediction],
    config: DebitSpreadConfig,
) -> List[StrategyTrade]:
    trades: List[StrategyTrade] = []
    for prediction in predictions:
        trade = trade_from_prediction(prediction, config)
        if trade is not None:
            trades.append(trade)
    return trades


def strategy_metrics(
    trades: Sequence[StrategyTrade],
    total_backtest_days: int,
) -> Dict[str, float]:
    if not trades:
        return {
            "backtest_days": float(total_backtest_days),
            "trades": 0.0,
            "trade_rate": 0.0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    wins = [trade for trade in trades if trade.pnl > 0]
    losses = [trade for trade in trades if trade.pnl < 0]
    gross_profit = sum(trade.pnl for trade in wins)
    gross_loss = abs(sum(trade.pnl for trade in losses))
    return {
        "backtest_days": float(total_backtest_days),
        "trades": float(len(trades)),
        "trade_rate": len(trades) / total_backtest_days if total_backtest_days else 0.0,
        "win_rate": len(wins) / len(trades),
        "total_pnl": sum(trade.pnl for trade in trades),
        "avg_pnl": sum(trade.pnl for trade in trades) / len(trades),
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
    }
