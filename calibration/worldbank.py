"""
worldbank.py — Descarga de datos Gini del Banco Mundial.

Usa la API REST v2 del Banco Mundial para obtener el indicador
SI.POV.GINI (Gini index from household surveys).

Documentación: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
"""

import time
import requests
import pandas as pd


# Indicador Gini del Banco Mundial
GINI_INDICATOR = "SI.POV.GINI"
WB_API_BASE = "https://api.worldbank.org/v2"

# Países de referencia para calibración
DEFAULT_COUNTRIES = [
    "WLD",   # Mundo
    "USA",   # EE. UU.
    "DEU",   # Alemania
    "BRA",   # Brasil
    "ZAF",   # Sudáfrica
    "SWE",   # Suecia
    "CHN",   # China
    "IND",   # India
    "MEX",   # México
    "ESP",   # España
]


def fetch_gini(
    countries: list = None,
    start_year: int = 2010,
    end_year: int = 2023,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Descarga datos Gini del Banco Mundial.

    Parámetros
    ----------
    countries : list[str]
        Códigos ISO3 de países. Defecto: DEFAULT_COUNTRIES.
    start_year : int
        Año de inicio de la serie.
    end_year : int
        Año final de la serie.
    max_retries : int
        Intentos máximos por petición.
    retry_delay : float
        Segundos de espera entre reintentos.

    Retorna
    -------
    pd.DataFrame con columnas: country_code, country_name, year, gini
    """
    if countries is None:
        countries = DEFAULT_COUNTRIES

    country_str = ";".join(countries)
    url = (
        f"{WB_API_BASE}/country/{country_str}/indicator/{GINI_INDICATOR}"
        f"?format=json&per_page=500&mrv=20&date={start_year}:{end_year}"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Error descargando datos del Banco Mundial: {e}"
                ) from e

    # La respuesta tiene formato [metadata, [records...]]
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Respuesta inesperada del Banco Mundial: {data}")

    records = data[1]
    if records is None:
        return pd.DataFrame(columns=["country_code", "country_name", "year", "gini"])

    rows = []
    for rec in records:
        if rec.get("value") is not None:
            rows.append({
                "country_code": rec["countryiso3code"],
                "country_name": rec["country"]["value"],
                "year": int(rec["date"]),
                "gini": float(rec["value"]) / 100.0,  # Banco Mundial da 0-100, normalizar a 0-1
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    return df


def latest_gini_by_country(
    countries: list = None,
    **kwargs
) -> pd.Series:
    """
    Retorna el Gini más reciente disponible para cada país.

    Retorna
    -------
    pd.Series indexed by country_code, values = gini (0-1)
    """
    df = fetch_gini(countries=countries, **kwargs)
    if df.empty:
        return pd.Series(dtype=float)
    latest = df.sort_values("year").groupby("country_code").last()
    return latest["gini"]


def world_gini_range(countries: list = None, **kwargs) -> dict:
    """
    Retorna estadísticas del rango de Gini observado en los países.

    Útil para definir targets de calibración realistas.
    """
    series = latest_gini_by_country(countries=countries, **kwargs)
    if series.empty:
        return {"min": 0.25, "max": 0.65, "mean": 0.40, "median": 0.38}
    return {
        "min":    float(series.min()),
        "max":    float(series.max()),
        "mean":   float(series.mean()),
        "median": float(series.median()),
        "std":    float(series.std()),
        "n":      int(len(series)),
    }
