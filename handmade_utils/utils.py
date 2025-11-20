from math import comb
from typing import List, Dict, Tuple

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

def print_centered(text: str, width: int, fill_char: str = ' ') -> None:
    if not fill_char or len(fill_char) != 1:
        raise ValueError("fill_char must be a single character.")

    centered_text = text.center(width, fill_char)
    print(centered_text)

def variance_of_sample(data : list) -> float:
    #See variance_of_sample.jpg
    if len(data) == 0:
        raise ValueError("The data list cannot be empty.")

    number_total_elements = len(data);
    average = sum(data) / len(data)
    variance = (sum((x - average) ** 2 for x in data)) / (number_total_elements - 1)
    return variance

def split_into_quantile_groups(df: pd.DataFrame, column: str, n_groups: int) -> List[pd.DataFrame]:
    quantile_labels = pd.qcut(df[column],q=n_groups,labels=False,duplicates='drop')
    df_copy = df.copy()
    df_copy['quantile_group'] = quantile_labels

    groups: List[pd.DataFrame] = [
        df_copy[df_copy['quantile_group'] == i].drop(columns=['quantile_group'])
        for i in range(quantile_labels.max() + 1)
    ]
    return groups

def count_value_frequencies( df: pd.DataFrame, column: str) -> Dict[str, int]:
    counts = df[column].value_counts().to_dict()
    return counts

def get_group_ranges(df: pd.DataFrame, column: str) -> Tuple[float, float]:
    return df[column].min(), df[column].max()

def prob_intersection(probability_a : float, probability_b : float) -> float:
    PA = probability_a / 100
    PB = probability_b / 100
    intersection = PA * PB
    return intersection * 100

def prob_union(probability_a: float, probability_b: float, intersection: float = None) -> float:
    PA = probability_a / 100
    PB = probability_b / 100

    if intersection is None:
        P_intersection = PA * PB
    else:
        P_intersection = intersection / 100

    union = PA + PB - P_intersection
    return union * 100

def repetition_in_muestral_space(total_count : int, subgroup_total : int, amount_to_remove : int, desired_amount : int):
    if desired_amount > amount_to_remove or desired_amount > subgroup_total:
        return 0.0

    prob = (comb(subgroup_total, desired_amount) * comb(total_count - subgroup_total, amount_to_remove - desired_amount)) / comb(total_count, amount_to_remove)
    return prob * 100

def joint_standard_deviation(n1 : int, std_dev1 : float, n2: int, std_dev2 : float) -> float:
    joint_variance : float = ((((n1 -1) * (std_dev1**2)) + ((n2 -1) * (std_dev2**2))) / (n1 + n2 - 2))
    return joint_variance ** 0.5

def joint_margin_of_error(z : float, joint_standard_deviation : float, n1 : int, n2 : int) -> float:
    return (z * joint_standard_deviation) * (((1 / n1) + (1 / n2)) ** 0.5)
