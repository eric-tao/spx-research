from __future__ import annotations

import pandas as pd


def attach_regime_columns(per_checkpoint: pd.DataFrame) -> pd.DataFrame:
    frame = per_checkpoint.copy()
    frame["move_bucket"] = pd.cut(
        frame["move_from_open_pct"],
        bins=[-10.0, -0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01, 10.0],
        labels=["<=-1.0%", "-1.0%..-0.5%", "-0.5%..-0.25%", "-0.25%..0.25%", "0.25%..0.5%", "0.5%..1.0%", ">=1.0%"],
        include_lowest=True,
    )
    frame["abs_move_bucket"] = pd.cut(
        frame["move_from_open_pct"].abs(),
        bins=[0.0, 0.0025, 0.005, 0.01, 10.0],
        labels=["0..0.25%", "0.25%..0.5%", "0.5%..1.0%", ">=1.0%"],
        include_lowest=True,
    )
    frame["vol_regime"] = pd.cut(
        frame["expected_move_proxy_pct"],
        bins=[0.0, 0.01, 0.02, 10.0],
        labels=["low_vol", "mid_vol", "high_vol"],
        include_lowest=True,
    )
    return frame
