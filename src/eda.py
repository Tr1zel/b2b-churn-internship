"""
Неделя 1: инструменты для разведочного анализа (EDA).
"""
import pandas as pd


def explore(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Быстрый профиль таблицы: типы, пропуски, уникальные значения.

    Parameters
    ----------
    df : исходная таблица
    name : имя таблицы для заголовка вывода

    Returns
    -------
    DataFrame со сводкой по столбцам (тип, % пропусков, число уникальных)
    """
    print(f"===== {name} =====")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")

    summary = pd.DataFrame({
        "тип": df.dtypes,
        "пропусков": df.isna().sum(),
        "пропусков_%": (df.isna().mean() * 100).round(1),
        "уникальных": df.nunique(),
    })
    print(summary)

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        print("\n--- Числовые столбцы ---")
        print(numeric.describe().T.round(2))

    return summary


def churn_rate_by(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    """Доля оттока в разбивке по группе (сегмент/продукт/регион).

    TODO (неделя 1): использовать для первой проверки гипотезы
    "отток зависит от сегмента".
    """
    return (
        df.groupby(group_col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "доля_оттока", "count": "клиентов"})
        .sort_values("доля_оттока", ascending=False)
    )

def missing_months(usage: pd.DataFrame) -> pd.DataFrame:
    """Дыры в помесячных данных клиента.

    Для каждого клиента: первый месяц, последний месяц, сколько месяцев
    фактически есть и сколько должно быть между первым и последним.

    Отсутствие месяцев В СЕРЕДИНЕ истории клиента — это дыра в данных,
    а не уход. Отличать одно от другого понадобится на неделе 2.

    Returns
    -------
    DataFrame с индексом client_id и столбцами:
        first_month, last_month, n_months_actual, n_months_expected, has_gap

    TODO (неделя 1)
    """

    cp_usage = usage.copy()
    cp_usage["month"] = pd.PeriodIndex(
        cp_usage["month"],
        freq="M"
    )

    grouped = cp_usage.groupby("client_id")

    result = grouped.agg(
        first_month = ("month", "min"),
        last_month = ("month", "max"),
        n_months_actual = ("month", "nunique")
    )

    result["n_months_expected"] = (
        result["last_month"].astype("int64") - result["first_month"].astype("int64") + 1
    )

    result["has_gap"] = (
        result["n_months_actual"] < result["n_months_expected"]
    )

    return result



