"""
Неделя 2-3: сборка обучающей выборки и построение признаков.

Правило №1 проекта: признаки строятся ТОЛЬКО из данных до точки отсчёта
(snapshot), таргет — ТОЛЬКО из данных после. Нарушение этого правила — data
leakage.

Функции ниже — скелеты. Заполните тело под свой датасет, не меняя сигнатуры
(так будет проще ревьюить и сравнивать между студентами).
"""
import numpy as np
import pandas as pd

OBSERVATION_MONTHS = 6  # окно наблюдения: сколько месяцев до snapshot
FORECAST_MONTHS = 3      # окно прогноза: сколько месяцев после snapshot


def build_snapshot(usage: pd.DataFrame, clients: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Собирает обучающую выборку для одной точки отсчёта.

    Parameters
    ----------
    usage : помесячная таблица (client_id, month, revenue, ...)
    clients : справочник клиентов (client_id, segment, product, ...)
    snapshot : строка вида '2025-01' — точка отсчёта

    Returns
    -------
    DataFrame с признаками, таргетом и метаданными (segment/product/snapshot_date)

    TODO (неделя 2):
    1. Отфильтровать клиентов, активных на конец окна наблюдения ("мёртвые души")
    2. Посчитать признаки по окну наблюдения через make_features()
    3. Определить таргет по окну прогноза
    4. Склеить признаки + таргет + метаданные клиента
    """

    snapshot_month = pd.Period(snapshot, freq="M")
    start_observ = snapshot_month - 6
    end_observ = snapshot_month - 1
    start_predict = snapshot_month
    end_predict = snapshot_month + 2

    usage["month"] = pd.PeriodIndex(usage["month"], freq="M")

    active_clients = usage.loc[
        (usage["month"] == end_observ)
        & (usage["revenue"] > 0),
        "client_id"
    ].unique()

    history_active_clients = usage.loc[
        usage["client_id"].isin(active_clients)
        & (usage["month"].between(start_observ, end_observ))
    ]
    features = make_features(history_active_clients, end_observ)

    predict_data = usage.loc[
        usage["client_id"].isin(active_clients)
        & usage["month"].between(start_predict, end_predict)
    ]

    last_revenue = (
        history_active_clients.loc[
            history_active_clients["month"] == end_observ
        ].set_index("client_id")["revenue"]
    )

    future_revenue = (
        predict_data.groupby("client_id")["revenue"].mean()
    )

    future_revenue = (
        future_revenue.reindex(last_revenue.index).fillna(0)
    )

    target = (
        future_revenue < 0.2 * last_revenue
    ).astype(int)
    target.name = "target"

    result = features.join(target).reset_index()

    result = result.merge(
        clients[["client_id", "segment", "product", "region"]],
        on="client_id",
        how="left",
    )
    result["snapshot"] = str(snapshot_month)
    return result


def make_features(obs: pd.DataFrame, obs_end) -> pd.DataFrame:
    """Строит признаки по данным окна наблюдения.

    Parameters
    ----------
    obs : помесячные данные окна наблюдения (client_id, month, revenue, ...)
    obs_end : последний месяц окна наблюдения

    Returns
    -------
    DataFrame признаков с индексом client_id

    TODO (неделя 3): реализовать группы признаков:
    - платежи (среднее/мин/макс, отношение последний/среднее)
    - динамика (тренд, волатильность, месяцев подряд снижения)
    - объём отношений (число услуг, срок жизни)
    - сигналы боли (обращения в поддержку, задолженность)
    """


    obs = obs.sort_values(["client_id", "month"])
    obs_grouped = obs.groupby("client_id")

    features = obs_grouped.agg(
        rev_mean = ("revenue", "mean"),
        rev_last = ("revenue", "last"),
        traffic_mean = ("traffic_gb", "mean"),
        sim_last = ("n_sim", "last")
    )
    return features


def trend_slope(s: pd.Series) -> float:
    """Наклон линейного тренда: >0 растёт, <0 падает."""
    y = s.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def months_declining(s: pd.Series) -> int:
    """Сколько последних месяцев подряд значение снижалось."""
    diffs = s.diff().to_numpy()[1:]
    cnt = 0
    for d in diffs[::-1]:
        if d < 0:
            cnt += 1
        else:
            break
    return cnt
