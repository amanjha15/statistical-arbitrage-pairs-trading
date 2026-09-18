import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import data, signals


def test_zscore_has_no_lookahead_nans_then_stabilizes():
    prices = data.synthetic_cointegrated_pair(n=300, seed=3)
    est = signals.build_rolling_estimate(prices["A"], prices["B"], window=50)

    assert est.zscore.iloc[:50].isna().all()
    assert est.zscore.iloc[100:].notna().all()


def test_generate_signals_entry_exit_stop_logic():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    z = pd.Series([np.nan, 0.0, 2.5, 2.6, 0.3, 0.0, -2.5, -0.4, 4.0, 0.0], index=idx)

    position = signals.generate_signals(z, entry=2.0, exit=0.5, stop=3.5)

    assert position.iloc[0] == 0
    assert position.iloc[2] == -1
    assert position.iloc[3] == -1
    assert position.iloc[4] == 0
    assert position.iloc[6] == 1
    assert position.iloc[7] == 0


def test_generate_signals_never_exceeds_unit_position():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    z = pd.Series(rng.normal(0, 1.5, size=200), index=idx)

    position = signals.generate_signals(z)

    assert position.isin([-1, 0, 1]).all()
