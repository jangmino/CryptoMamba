"""
comp_tools.py

이 모듈은 암호화폐 데이터 처리 및 분석을 위한 유틸리티 함수들을 제공합니다.
주요 기능은 다음과 같습니다:
    - 타임스탬프와 datetime 간 변환
    - 원시 데이터 로드
    - 차트 데이터 생성 (기술적 지표 포함)
"""

import pandas as pd
import numpy as np


def timestamp_to_datetime(series):
    """pd.Series 내 리눅스 타임스탬프를 pandas datetime으로 변환합니다.

    Args:
      series (pd.Series): 리눅스 타임스탬프를 담은 pd.Series

    Returns:
      pd.Series: pandas datetime으로 변환된 pd.Series
    """
    return pd.to_datetime(series, unit="s")  # 초 단위 타임스탬프 가정


import pandas as pd


def datetime_to_timestamp(series):
    """pd.Series 내 pandas datetime을 리눅스 타임스탬프로 변환합니다.

    Args:
      series (pd.Series): pandas datetime을 담은 pd.Series

    Returns:
      pd.Series: 리눅스 타임스탬프로 변환된 pd.Series
    """
    return series.astype("int64") // 10**9  # 나노초 -> 초 변환


def get_raw_dfs(args):
    """
    지정된 해상도에 따라 원시 데이터 DataFrame을 로드합니다.

    Args:
        args: 다음 속성을 가진 객체:
            - base_path_for_raw_data (str): 원시 데이터가 저장된 기본 경로.
            - symbol (str): 데이터에 대한 기호 (예: 'BTCUSDT').
            - raw_data_period (str): 데이터의 기간 (예: '1d').
            - resolutions (list): 로드할 해상도 목록 (예: ['1h', '4h', '1d']).
    Returns:
        dict: 해상도를 키로, 해당 해상도의 DataFrame을 값으로 하는 딕셔너리.
    """

    base_path = args.base_path_for_raw_data
    raw_data_paths = {
        resolution: f"{base_path}/{args.symbol}_{args.raw_data_period}_{resolution}.pkl"
        for resolution in args.resolutions
    }
    return {k: pd.read_pickle(v) for k, v in raw_data_paths.items()}


def make_charted_data(in_df: pd.DataFrame, use_log_scale: bool = False) -> pd.DataFrame:
    """
    원시 데이터 DataFrame을 차트 데이터로 변환합니다.

    Args:
        in_df (pd.DataFrame): 원시 데이터 DataFrame. 'Open', 'High', 'Low', 'Close', 'Volume' 컬럼을 포함해야 합니다.
        use_log_scale (bool, optional): 로그 스케일을 사용할지 여부. Defaults to False.

    Returns:
        pd.DataFrame: 차트 데이터 DataFrame.
            - 'open', 'high', 'low', 'close', 'volume' 컬럼을 포함합니다.
            - SMA(5, 10, 20), VWAP, Bollinger Bands, RSI(7, 14, 21), MACD 지표를 포함합니다.
            - 필요에 따라 로그 스케일이 적용됩니다.
            - 'date' 컬럼을 포함합니다.
    """
    col_conversion = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    df = in_df[list(col_conversion.keys())].copy()

    # 모든 컬럼을 float으로 변환
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError as e:
            print(f"Error converting column '{col}' to float: {e}")
            return None

    df.rename(columns=col_conversion, inplace=True)
    df.reset_index().rename(columns={"Open time": "date"}, inplace=True)

    # SMA
    for l in [5, 10, 20]:
        df.ta.sma(length=l, append=True)

    # VWAP, Bollinger Bands
    df.ta.vwap(append=True)
    df.ta.bbands(length=20, std=2, append=True)

    # RSI
    for l in [7, 14, 21]:
        df.ta.rsi(length=l, append=True)

    # MACD
    df.ta.macd(append=True)

    col_conversion_derived = {
        "VWAP_D": "vwap",
        "BBU_20_2.0": "bbu",
        "BBL_20_2.0": "bbl",
        "BBM_20_2.0": "bbm",
    }
    df.rename(columns=col_conversion_derived, inplace=True)
    df.dropna(inplace=True)

    log_columns = [
        # "open",
        # "high",
        # "low",
        # "close",
        "volume",
        # "vwap",
        # "bbu",
        # "bbl",
        # "bbm",
    ]
    if use_log_scale:
        for col in log_columns:
            df[col] = np.log1p(df[col])

    df.reset_index(inplace=True)
    df.rename(columns={"Open time": "date"}, inplace=True)
    return df
