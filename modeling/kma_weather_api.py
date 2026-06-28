from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import urllib.parse
import urllib.request


APIHUB_BASE_URL = "https://apihub.kma.go.kr"
DEFAULT_CAMPUS_LAT = 35.8338
DEFAULT_CAMPUS_LON = 128.7546
DEFAULT_ENV_PATH = Path(".env")


@dataclass(frozen=True)
class KmaGridPoint:
    x: int
    y: int


def load_env_file(path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_kma_auth_key(env_path: Path = DEFAULT_ENV_PATH) -> str:
    env_values = load_env_file(env_path)
    key = os.environ.get("KMA_APIHUB_AUTH_KEY") or env_values.get("KMA_APIHUB_AUTH_KEY")
    if not key:
        raise RuntimeError(
            "KMA_APIHUB_AUTH_KEY is missing. Set it in .env or the process environment.",
        )
    return key


def kma_grid_from_lonlat(lon: float, lat: float) -> KmaGridPoint:
    re = 6371.00877
    grid = 5.0
    slat1 = math.radians(30.0)
    slat2 = math.radians(60.0)
    olon = math.radians(126.0)
    olat = math.radians(38.0)
    xo = 43.0
    yo = 136.0

    ra = re / grid
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(
        math.pi * 0.25 + slat1 * 0.5,
    )
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf**sn * math.cos(slat1)) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = ra * sf / (ro**sn)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    rn = math.tan(math.pi * 0.25 + lat_rad * 0.5)
    rn = ra * sf / (rn**sn)
    theta = lon_rad - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = int(math.floor(rn * math.sin(theta) + xo + 0.5))
    y = int(math.floor(ro - rn * math.cos(theta) + yo + 0.5))
    return KmaGridPoint(x=x, y=y)


def build_apihub_url(
    endpoint: str,
    params: dict[str, object],
    auth_key: str,
) -> str:
    query = {key: value for key, value in params.items() if value is not None}
    query["authKey"] = auth_key
    return f"{APIHUB_BASE_URL}{endpoint}?{urllib.parse.urlencode(query)}"


def fetch_text(url: str, timeout: int = 30) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        if error.code == 403:
            raise RuntimeError(
                "KMA APIHub returned HTTP 403. Check that the authKey is valid and "
                "that this API is approved for the account.",
            ) from error
        raise
