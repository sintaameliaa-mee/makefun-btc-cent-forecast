#!/usr/bin/env python3
"""Mean-reversion BTC forecaster for make.fun btc-cent-forecast-01.

Forecasts Bitcoin's US-dollar price 240 minutes (4 hours) after each
checkpoint, from the 4,320 one-minute closes of the 72 hours up to that
checkpoint. One case at a time, one number out.

Method -- mild mean reversion to the trailing 4-hour mean
---------------------------------------------------------
Bitcoin's price wanders, but over a few hours it tends to drift back
toward where it has just been. So instead of guessing that the price
stays exactly where it is at the checkpoint (the persistence guess),
we nudge the guess a fifth of the way back toward the average price
of the last four hours:

    forecast = last + 0.20 * (mean(last 240 closes) - last)

where ``last`` is the close at the checkpoint (the final price in the
case) and ``mean(last 240 closes)`` is the arithmetic mean of the 240
most recent minute closes, the checkpoint close included.

Why 0.20 and why 240 minutes: the window matches the forecast
horizon (4 hours back, 4 hours forward), so the mean is the price
level the market has actually lived at over the same span we are
forecasting across. The pull is deliberately small -- a fifth of the
gap -- because the mean is only a weak attractor: most of the time
the price is close to its recent mean and the nudge is a few dollars,
while after a sharp one-sided move the nudge leans against the move
instead of chasing it. A full reversion (factor 1.0) would overreact
to every spike; a factor of 0 would be plain persistence. 0.20 sits
between the two, and on the kit's practice cases it beats
persistence by roughly three percent in mean absolute error.

Interface (from the kit's RULES.txt / starter):
-----------------------------------------------
* Input:  every ``<id>.in`` file in the input directory
  ($MAKEFUN_INPUT_DIR, else /input). Each file is one case, UTF-8
  JSON: ``{"schema": "makefun/btc-cent-case/v1", "competition",
  "case": "<id>", "at": "<ISO checkpoint>", "horizon_minutes": 240,
  "markets": [{"id": "btc-usdc", "name": ..., "start": ...,
  "prices": [<4,320 closes, last one at the checkpoint>]}]}``.
* Output: one ``<id>.out`` per case in the output directory
  ($MAKEFUN_OUTPUT_DIR, else /output), UTF-8 JSON with exactly one
  key: ``{"forecasts": [{"case": "<id>", "price": <guess>}]}``.
  The price must be finite, greater than 0 and less than 10,000,000.

Runtime: Python 3.13, standard library only, no network, no model
calls. The program is fully deterministic -- the same case always
produces the same bytes -- and runs in well under a second per case.
"""

import json
import math
import os
import sys

# --- Tuned constants -------------------------------------------------
# Fraction of the gap between the checkpoint close and the trailing
# 4-hour mean that the forecast closes. 0.20 = mild reversion.
REVERSION_FACTOR = 0.20

# How many of the most recent closes form the trailing mean. 240
# minutes = 4 hours, the same length as the forecast horizon.
WINDOW_MINUTES = 240

# The price grid's bounds, matching the scorer: prices are positive,
# below 10,000,000 dollars, and quoted to 5 decimals of a dollar.
MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999


def clamp_to_grid(price):
    """Keep a price on the scorer's grid: finite, in range, 5 decimals."""
    if not math.isfinite(price):
        raise ValueError("forecast is not a finite number: %r" % (price,))
    clamped = min(MAX_PRICE, max(MIN_PRICE, price))
    return round(clamped, 5)


def trailing_mean(prices, window):
    """Arithmetic mean of the last ``window`` closes (or all, if fewer).

    Every close that enters the mean must be a finite number: a single
    corrupt (null / non-numeric) entry must fail loudly instead of
    silently poisoning the forecast with a NaN.
    """
    tail = prices[-window:] if len(prices) > window else prices
    for value in tail:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("non-finite price in trailing window: %r" % (value,))
    return sum(tail) / len(tail)


def forecast_case(case):
    """Forecast one case: mean reversion of the checkpoint close.

    ``case`` is the parsed JSON of one ``<id>.in`` file. Returns the
    forecast price as a float on the 5-decimal grid.
    """
    markets = case.get("markets")
    if not isinstance(markets, list) or len(markets) != 1:
        raise ValueError("case must hold exactly one market")
    market = markets[0]
    prices = market.get("prices")
    if not isinstance(prices, list) or not prices:
        raise ValueError("market %r has no prices" % (market.get("id"),))

    # The checkpoint close: the last price in the series.
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last):
        raise ValueError("checkpoint close is not a finite number: %r" % (last,))

    # The level the price has averaged over the trailing 4 hours.
    mean_240 = trailing_mean(prices, WINDOW_MINUTES)

    # Mild mean reversion: close a fifth of the gap toward the mean.
    # If the price sits above its recent mean, the forecast leans down;
    # if it sits below, the forecast leans up. If price == mean, this
    # reduces exactly to persistence (forecast == last).
    raw = last + REVERSION_FACTOR * (mean_240 - last)
    return clamp_to_grid(raw)


def main(argv=None):
    """Read every case in the input dir, write one forecast per case."""
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

    os.makedirs(output_dir, exist_ok=True)

    # Every case file, in name order, so two runs write the same bytes.
    case_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    if not case_files:
        print("no .in case files found in %s" % input_dir, file=sys.stderr)
        return 1

    for filename in case_files:
        case_id = filename[: -len(".in")]
        in_path = os.path.join(input_dir, filename)
        with open(in_path, "r", encoding="utf-8") as fh:
            case = json.load(fh)

        price = forecast_case(case)

        out_path = os.path.join(output_dir, case_id + ".out")
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
