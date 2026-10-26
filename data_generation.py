"""This script downloads the capital bike share dataset and combines it with the Open-Meteo weather data.

It script was created with the help of LLMS in order to create one consistent dataset for the AI and Non-AI pairs.
"""
from __future__ import annotations

import json
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

root = Path.cwd()
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2027-01-01")

def download_trips() -> None:
    base = "https://s3.amazonaws.com/capitalbikeshare-data"
    today = date.today()
    for year in range(2020, 2027):
        for month in range(1, 13):
            if (year, month) > (today.year, today.month):
                continue
            name = f"{year}{month:02d}-capitalbikeshare-tripdata.zip"
            target = root / name
            if target.exists() and target.stat().st_size > 0:
                continue
            try:
                print("download", base + "/" + name, flush=True)
                urllib.request.urlretrieve(f"{base}/{name}", target)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise

download_trips()
files = sorted(
    p for p in root.iterdir()
    if p.suffix.lower() in {".csv", ".zip"}
    and ("tripdata" in p.name.lower() or "capitalbikeshare" in p.name.lower())
)
if not files:
    raise FileNotFoundError("Put the official Capital Bikeshare trip CSV/ZIP files in this folder.")

aliases = {
    "started_at": ("started_at", "start date", "starttime", "start time", "started at"),
    "ended_at": ("ended_at", "end date", "endtime", "end time", "ended at"),
    "rideable_type": ("rideable_type", "bike type", "biketype"),
    "member_casual": ("member_casual", "member type", "usertype", "user type", "customer type"),
    "start_station_id": ("start_station_id", "start station number", "start station id", "start_station_number"),
    "start_station_name": ("start_station_name", "start station", "start station name"),
    "end_station_id": ("end_station_id", "end station number", "end station id", "end_station_number"),
    "end_station_name": ("end_station_name", "end station", "end station name"),
    "start_lat": ("start_lat", "start latitude"),
    "start_lng": ("start_lng", "start longitude"),
    "end_lat": ("end_lat", "end latitude"),
    "end_lng": ("end_lng", "end longitude"),
}
required = list(aliases)


def rename_columns(columns) -> dict:
    lower = {str(c).strip().lower(): c for c in columns}
    out = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in lower:
                out[lower[name]] = canonical
                break
    return out


def distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fetch_weather(start, end) -> pd.DataFrame:
    end = min(end, date.today() - timedelta(days=5))
    params = {
        "latitude": 38.9072,
        "longitude": -77.0369,
        "start_date": str(start),
        "end_date": str(end),
        "timezone": "America/New_York",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "snowfall", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
        ]),
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    payload = json.load(urllib.request.urlopen(url, timeout=120))
    wx = pd.DataFrame(payload["hourly"]).rename(columns={
        "time": "timestamp",
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "precipitation": "precip",
        "snowfall": "snowfall",
        "cloud_cover": "cloudcover",
        "wind_speed_10m": "windspeed",
        "wind_gusts_10m": "windgust",
    })
    wx["timestamp"] = pd.to_datetime(wx.timestamp)
    wx["is_rain"] = (wx.precip > 0.2).astype(int)
    wx["is_heavy_rain"] = (wx.precip > 10).astype(int)
    wx["is_snow"] = (wx.snowfall > 0).astype(int)
    wx["weather_regime"] = np.select(
        [wx.is_snow == 1, wx.is_heavy_rain == 1, wx.is_rain == 1],
        ["snow", "heavy rain", "rain"],
        default="dry",
    )
    return wx


def read_chunks(path: Path):
    if path.suffix.lower() != ".zip":
        yield from pd.read_csv(path, chunksize=1_000_000, low_memory=False)
        return
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        ]
        for name in members:
            with archive.open(name) as handle:
                yield from pd.read_csv(handle, chunksize=1_000_000, low_memory=False)

start_parts, station_start_parts, station_end_parts = [], [], []
for path in files:
    for chunk in read_chunks(path):
        chunk = chunk.rename(columns=rename_columns(chunk.columns))
        for col in required:
            if col not in chunk:
                chunk[col] = np.nan
        chunk["started_at"] = pd.to_datetime(chunk.started_at, errors="coerce", format="mixed")
        chunk["ended_at"] = pd.to_datetime(chunk.ended_at, errors="coerce", format="mixed")
        chunk = chunk[(chunk.started_at >= START_DATE) & (chunk.started_at < END_DATE)]
        chunk = chunk.dropna(subset=["started_at"])

        text_type = chunk.rideable_type.astype(str).str.lower()
        chunk["classic"] = text_type.str.contains("classic").astype("int8")
        chunk["docked"] = text_type.str.contains("docked").astype("int8")
        chunk["electric"] = text_type.str.contains("electric").astype("int8")
        text_member = chunk.member_casual.astype(str).str.lower()
        chunk["member"] = text_member.str.contains("member|registered").astype("int8")
        chunk["casual"] = text_member.str.contains("casual|customer").astype("int8")

        chunk["start_hour"] = chunk.started_at.dt.floor("h")
        chunk["end_hour"] = chunk.ended_at.dt.floor("h")
        chunk["date"] = chunk.started_at.dt.normalize()
        duration = (chunk.ended_at - chunk.started_at).dt.total_seconds() / 60
        chunk["duration"] = duration.where(duration.between(0, 1440))
        chunk["distance"] = distance(
            chunk.start_lat, chunk.start_lng, chunk.end_lat, chunk.end_lng
        ).where(lambda s: s.between(0, 100))

        start_parts.append(chunk.groupby("start_hour").agg(
            rides_started=("member", "size"),
            member_rides=("member", "sum"),
            casual_rides=("casual", "sum"),
            classic_rides=("classic", "sum"),
            docked_rides=("docked", "sum"),
            electric_rides=("electric", "sum"),
            duration_sum=("duration", "sum"),
            duration_n=("duration", "count"),
            distance_sum=("distance", "sum"),
            distance_n=("distance", "count"),
        ))
        station_start_parts.append(
            chunk.dropna(subset=["start_station_id"]).groupby(["date", "start_station_id"]).agg(
                pickup_counts=("member", "size"),
                member_pickups=("member", "sum"),
                casual_pickups=("casual", "sum"),
                classic_pickups=("classic", "sum"),
                docked_pickups=("docked", "sum"),
                electric_pickups=("electric", "sum"),
                duration_sum=("duration", "sum"),
                duration_n=("duration", "count"),
                station_name=("start_station_name", "first"),
            ).rename_axis(["date", "station_id"])
        )
        station_end_parts.append(
            chunk.dropna(subset=["end_station_id"]).groupby(["date", "end_station_id"]).size()
            .rename("dropoff_counts").rename_axis(["date", "station_id"])
        )

hourly = pd.concat(start_parts).groupby(level=0).sum()
hourly = hourly.reindex(
    pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name="timestamp"),
    fill_value=0,
)
hourly = hourly.reset_index()
hourly["unknown_rides"] = hourly.rides_started - hourly.classic_rides - hourly.docked_rides - hourly.electric_rides
hourly["avg_duration_min"] = hourly.duration_sum / hourly.duration_n.replace(0, np.nan)
hourly["avg_distance_km"] = hourly.distance_sum / hourly.distance_n.replace(0, np.nan)
hourly["date"] = hourly.timestamp.dt.normalize()
hourly["hour"] = hourly.timestamp.dt.hour
hourly["dow"] = hourly.timestamp.dt.dayofweek
hourly["month"] = hourly.timestamp.dt.month
hourly["is_weekend"] = (hourly.dow >= 5).astype(int)

weather = fetch_weather(hourly.timestamp.min().date(), hourly.timestamp.max().date())
weather.to_csv(root / "weather_hourly.csv", index=False)
hourly = hourly.merge(weather, on="timestamp", how="left")

station_start = pd.concat(station_start_parts).reset_index()
station_start = station_start.groupby(["date", "station_id"]).agg(
    pickup_counts=("pickup_counts", "sum"),
    member_pickups=("member_pickups", "sum"),
    casual_pickups=("casual_pickups", "sum"),
    classic_pickups=("classic_pickups", "sum"),
    docked_pickups=("docked_pickups", "sum"),
    electric_pickups=("electric_pickups", "sum"),
    duration_sum=("duration_sum", "sum"),
    duration_n=("duration_n", "sum"),
    station_name=("station_name", "first"),
).reset_index()
station_start["avg_duration_min"] = station_start.duration_sum / station_start.duration_n.replace(0, np.nan)
station_start["station_id"] = station_start.station_id.astype(str)
station_end = pd.concat(station_end_parts).groupby(level=[0, 1]).sum().reset_index()
station_end["station_id"] = station_end.station_id.astype(str)
station = station_start.merge(station_end, on=["date", "station_id"], how="outer")
for col in ["pickup_counts", "dropoff_counts", "member_pickups", "casual_pickups",
            "classic_pickups", "docked_pickups", "electric_pickups"]:
    station[col] = station[col].fillna(0)
station["station_name"] = station.station_name.fillna("")
station["turnover"] = station.pickup_counts + station.dropoff_counts
station["unknown_pickups"] = station.pickup_counts - station.classic_pickups - station.docked_pickups - station.electric_pickups

daily = weather.set_index("timestamp").resample("D").agg({
    "temp": "mean", "precip": "sum", "snowfall": "sum", "humidity": "mean",
    "windspeed": "mean", "windgust": "max", "cloudcover": "mean",
}).reset_index()
daily["is_rain"] = (daily.precip > 0.2).astype(int)
daily["is_heavy_rain"] = (daily.precip > 10).astype(int)
daily["is_snow"] = (daily.snowfall > 0).astype(int)
daily["weather_regime"] = np.select(
    [daily.is_snow == 1, daily.is_heavy_rain == 1, daily.is_rain == 1],
    ["snow", "heavy rain", "rain"],
    default="dry",
)
station = station.merge(daily.rename(columns={"timestamp": "date"}), on="date", how="left")

holidays = set(USFederalHolidayCalendar().holidays("2020-01-01", "2026-12-31"))
for frame, date_col in [(hourly, "timestamp"), (station, "date")]:
    frame["is_holiday"] = frame[date_col].dt.normalize().isin(holidays).astype(int)

hourly[[
    "timestamp", "rides_started", "member_rides", "casual_rides",
    "classic_rides", "docked_rides", "electric_rides", "unknown_rides",
    "avg_duration_min", "avg_distance_km", "date", "hour", "dow", "month",
    "is_weekend", "is_holiday", "temp", "precip", "snowfall", "humidity",
    "windspeed", "windgust", "cloudcover", "is_rain", "is_heavy_rain",
    "is_snow", "weather_regime",
]].to_csv(root / "capital_hourly.csv", index=False)
station[[
    "date", "station_id", "station_name", "pickup_counts", "dropoff_counts", "turnover",
    "member_pickups", "casual_pickups", "classic_pickups", "docked_pickups",
    "electric_pickups", "unknown_pickups", "avg_duration_min", "is_holiday",
    "temp", "precip", "snowfall", "humidity", "windspeed", "windgust",
    "cloudcover", "is_rain", "is_heavy_rain", "is_snow", "weather_regime",
]].to_csv(root / "capital_station_daily.csv", index=False)
print("done")
