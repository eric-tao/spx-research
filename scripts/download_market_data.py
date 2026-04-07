from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


def _in_range(day: str, start: str, end: str, fmt: str) -> bool:
    parsed = datetime.strptime(day, fmt).date().isoformat()
    return start <= parsed <= end


def download_stooq_spx(start: str, end: str) -> list[dict[str, str]]:
    request = Request("https://stooq.com/q/d/l/?s=%5Espx&i=d", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        if not _in_range(row["Date"], start, end, "%Y-%m-%d"):
            continue
        rows.append(
            {
                "date": row["Date"],
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row.get("Volume", "0") or "0",
            }
        )
    return rows


def download_yahoo_spx(start: str, end: str) -> list[dict[str, str]]:
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    request = Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={start_ts}&period2={end_ts}&interval=1d&includePrePost=false&events=div%2Csplits",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    rows: list[dict[str, str]] = []
    for idx, ts in enumerate(timestamps):
        open_value = opens[idx]
        high_value = highs[idx]
        low_value = lows[idx]
        close_value = closes[idx]
        if None in (open_value, high_value, low_value, close_value):
            continue
        bar_date = datetime.utcfromtimestamp(ts).date().isoformat()
        rows.append(
            {
                "date": bar_date,
                "open": f"{open_value:.2f}",
                "high": f"{high_value:.2f}",
                "low": f"{low_value:.2f}",
                "close": f"{close_value:.2f}",
                "volume": str(int(volumes[idx] or 0)),
            }
        )
    return rows


def download_cboe_vix(start: str, end: str) -> list[dict[str, str]]:
    request = Request("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        if not _in_range(row["DATE"], start, end, "%m/%d/%Y"):
            continue
        parsed_date = datetime.strptime(row["DATE"], "%m/%d/%Y").date().isoformat()
        rows.append(
            {
                "date": parsed_date,
                "open": row["OPEN"],
                "high": row["HIGH"],
                "low": row["LOW"],
                "close": row["CLOSE"],
                "volume": "0",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SPX and VIX daily OHLC from public historical sources")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--spx-out", default="data/spx_daily.csv")
    parser.add_argument("--vix-out", default="data/vix_daily.csv")
    args = parser.parse_args()

    spx_rows = download_stooq_spx(args.start, args.end)
    if not spx_rows:
        spx_rows = download_yahoo_spx(args.start, args.end)
    vix_rows = download_cboe_vix(args.start, args.end)
    write_csv(Path(args.spx_out), spx_rows, ["date", "open", "high", "low", "close", "volume"])
    write_csv(Path(args.vix_out), vix_rows, ["date", "open", "high", "low", "close", "volume"])
    print(f"Downloaded {len(spx_rows)} SPX rows to {args.spx_out}")
    print(f"Downloaded {len(vix_rows)} VIX rows to {args.vix_out}")


if __name__ == "__main__":
    main()
