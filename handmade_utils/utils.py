import numpy as np
import pandas as pd

def get_class_intervals_info(array_number_limits: list, array_labels: list, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:

    #This is part of the library, we assign new labels behind.
    cut_series = pd.cut(
        dataframe[column_name],
        bins=array_number_limits,
        labels=array_labels,
        include_lowest=True
    )

    #We obtain a sub-dataframe.
    class_intervals_count = cut_series.value_counts().sort_index()

    #We calculate midpoints.
    midpoints = []
    for i in range(len(array_number_limits) - 1):
        lower = array_number_limits[i]
        upper = array_number_limits[i + 1]
        midpoints.append((lower + upper) / 2)

    #We obtain class averages.
    class_average = []
    for label in array_labels:
        class_values = dataframe[column_name][cut_series == label]
        if len(class_values) > 0:
            class_average.append(class_values.mean())
        else:
            class_average.append(np.nan)  # No data in the interval

    df_result = pd.DataFrame({
        'Frequency': class_intervals_count.values,
        'Relative Frequency (%)': (class_intervals_count / class_intervals_count.sum() * 100).round(2),
        'Class Midpoint': midpoints,
        'Class Average': class_average
    })

    return df_result


def variance_of_sample(data : list) -> float:
    #See variance_of_sample.jpg
    if len(data) == 0:
        raise ValueError("The data list cannot be empty.")

    number_total_elements = len(data);
    average = sum(data) / len(data)
    variance = (sum((x - average) ** 2 for x in data)) / (number_total_elements - 1)
    return variance

