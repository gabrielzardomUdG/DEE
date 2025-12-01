from itertools import zip_longest
import handmade_utils.utils as hmu
from typing import List

import numpy as np
import pandas as pd

from scipy.stats import f, norm, chi2, kurtosis, skew

NORMAL_DISTRIBUTION_ACC_TABLE : np.array = np.array([
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0359, 0.0228, 0.0162, 0.0122, 0.0099, 0.0082, 0.0070, 0.0062],
    [0.2743, 0.1587, 0.1003, 0.0668, 0.0475, 0.0359, 0.0281, 0.0228],
    [0.7257, 0.5000, 0.3372, 0.2266, 0.1587, 0.1151, 0.0869, 0.0668],
    [0.9641, 0.8413, 0.6628, 0.5000, 0.3707, 0.2743, 0.2062, 0.1587],
    [1.0000, 0.9772, 0.8997, 0.7734, 0.6293, 0.5000, 0.3934, 0.3085],
    [np.nan, 1.0000, 0.9838, 0.9332, 0.8413, 0.7257, 0.6066, 0.5000],
    [np.nan, np.nan, 1.0000, 0.9878, 0.9525, 0.8849, 0.7938, 0.6915],
    [np.nan, np.nan, np.nan, 1.0000, 0.9901, 0.9641, 0.9131, 0.8413],
    [np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9918, 0.9719, 0.9332],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9930, 0.9772],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9938],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000],
], dtype=np.float64)

IMAGE_BASED_ACC_TABLE_NORMAL_DISTRIBUTION: np.array = np.array([
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.2000, 0.1667, 0.1429, 0.1250, 0.1111, 0.1000, 0.0909, 0.0833],
    [0.4000, 0.3333, 0.2857, 0.2500, 0.2222, 0.2000, 0.1818, 0.1667],
    [0.6000, 0.5000, 0.4286, 0.3750, 0.3333, 0.3000, 0.2727, 0.2500],
    [0.8000, 0.6667, 0.5714, 0.5000, 0.4444, 0.4000, 0.3636, 0.3333],
    [1.0000, 0.8333, 0.7143, 0.6250, 0.5555, 0.5000, 0.4545, 0.4167],
    [np.nan, 1.0000, 0.8571, 0.7500, 0.6666, 0.6000, 0.5454, 0.5000],
    [np.nan, np.nan, 1.0000, 0.8750, 0.7777, 0.7000, 0.6363, 0.5833],
    [np.nan, np.nan, np.nan, 1.0000, 0.8888, 0.8000, 0.7272, 0.6667],
    [np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9090, 0.8181, 0.7500],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9090, 0.8333],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000, 0.9167],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0000],
], dtype=np.float64)

NH_VARIANCE     : str = "CHI CALCULATED"
NH_CHICRIT    : str = "CHI CRIT"
NH_IS_ACCEPTED  : str = "IS THE NH ACCEPTED?"

CHI_COMP_HEADERS    : list = [NH_VARIANCE, NH_CHICRIT, NH_IS_ACCEPTED]
def produce_example_cumulative_intervals()-> np.ndarray :
    return np.array([8, 10, 15, 22, 20, 13, 7, 5])

def cumulated_area_of_normal_distribution(number_of_intervals: int, section_index: int) -> np.float64:
    return NORMAL_DISTRIBUTION_ACC_TABLE[section_index][number_of_intervals - 5]

def calculate_element_element_of_expected_frequency(n : int, number_of_intervals: int, section_index: int) -> np.float64:
    a : np.float64 = cumulated_area_of_normal_distribution(number_of_intervals, section_index)
    b : np.float64 = cumulated_area_of_normal_distribution(number_of_intervals, section_index + 1)
    return np.round(n * (b - a))

def calculate_expected_frequency_for_normal_distribution_array(input_array : np.ndarray) -> np.ndarray:
    output : np.array() = np.array([])
    n : int = input_array.sum()
    n_intervals : int = len(input_array)

    for i in range(0, len(input_array)):
        output = np.append(output, calculate_element_element_of_expected_frequency(n, n_intervals, i))

    return output

def calculate_chi(observed_frequency : np.ndarray, expected_frequency : np.ndarray) -> np.float64:
    chi_calculated : np.float64 = 0
    for i in range(0, len(observed_frequency)):
        chi_calculated += ((observed_frequency[i] - expected_frequency[i]) ** 2) / expected_frequency[i]

    return chi_calculated

def calculate_chi_uniform_distribution(observed_frequency : np.ndarray) -> np.float64:
    chi_calculated : np.float64 = 0
    n : int = observed_frequency.sum()
    k : int = len(observed_frequency)
    fe : np.float64 = n/k
    for i in range(0, len(observed_frequency)):
        chi_calculated += ((observed_frequency[i] - fe) ** 2) / fe

    return chi_calculated

def produce_null_hypothesis_table(observed_frequency : np.ndarray, expected_frequency : np.ndarray, alpha : np.float64) -> pd.DataFrame:
    degrees_of_freedom : int = (len(observed_frequency) - 1)
    chi2_calculated : np.float64 = calculate_chi(observed_frequency,expected_frequency)
    chi2_crit : np.float64 = chi2.ppf(1 - alpha, degrees_of_freedom)

    output_data : pd.DataFrame = pd.DataFrame(columns=CHI_COMP_HEADERS)
    decision : str = "Accepted" if chi2_calculated < chi2_crit else "Denied"
    row : list = [chi2_calculated, chi2_crit, decision]
    output_data.loc[len(output_data)] = row
    return output_data

def produce_null_hypothesis_table_is_uniform_distribution(observed_frequency : np.ndarray, alpha : np.float64) -> pd.DataFrame:
    degrees_of_freedom : int = (len(observed_frequency) - 1)
    chi2_calculated : np.float64 = calculate_chi_uniform_distribution(observed_frequency)
    chi2_crit : np.float64 = chi2.ppf(1 - alpha, degrees_of_freedom)

    output_data : pd.DataFrame = pd.DataFrame(columns=CHI_COMP_HEADERS)
    decision : str = "Accepted" if chi2_calculated < chi2_crit else "Denied"
    row : list = [chi2_calculated, chi2_crit, decision]
    output_data.loc[len(output_data)] = row
    return output_data


def count_values_in_intervals(data: np.ndarray | pd.Series, n_intervals: int):

    data = np.asarray(data)

    min_val = data.min()
    max_val = data.max()

    intervals = np.linspace(min_val, max_val, n_intervals + 1)

    counts, _ = np.histogram(data, bins=intervals)

    return {
        "min": min_val,
        "max": max_val,
        "intervals": intervals,
        "counts": counts
    }


def produce_null_hypothesis_table_is_uniform_distribution(observed_frequency : np.ndarray, alpha : np.float64) -> pd.DataFrame:
    degrees_of_freedom : int = (len(observed_frequency) - 1)
    chi2_calculated : np.float64 = calculate_chi_uniform_distribution(observed_frequency)
    chi2_crit : np.float64 = chi2.ppf(1 - alpha, degrees_of_freedom)

    output_data : pd.DataFrame = pd.DataFrame(columns=CHI_COMP_HEADERS)
    decision : str = "Accepted" if chi2_calculated < chi2_crit else "Denied"
    row : list = [chi2_calculated, chi2_crit, decision]
    output_data.loc[len(output_data)] = row
    return output_data

def example_practice(alpha : np.float64 = 0.05):
    observed_frequency : np.ndarray = produce_example_cumulative_intervals()
    expected_frequency : np.ndarray = calculate_expected_frequency_for_normal_distribution_array(observed_frequency)
    print()
    hmu.print_centered("Exercise : ", 108, "*")
    print("Observed Frequency : ", observed_frequency)
    print("Expected Frequency : ", expected_frequency)
    print(produce_null_hypothesis_table(observed_frequency, expected_frequency, alpha))
    return

def is_normal_distribution(subject : str,  data: np.ndarray | pd.Series, n_intervals: int , alpha : np.float64 = 0.05):
    result = count_values_in_intervals(data, n_intervals)
    observed_frequency : np.ndarray = result["counts"]
    expected_frequency : np.ndarray = calculate_expected_frequency_for_normal_distribution_array(observed_frequency)
    print()
    hmu.print_centered("Subject : " + subject, 108, "*")
    print("Min:", result["min"])
    print("Max:", result["max"])
    print("Intervals:", result["intervals"])
    print("Observed Frequency : ", observed_frequency)
    print("Expected Frequency : ", expected_frequency)
    print()
    print(produce_null_hypothesis_table(observed_frequency, expected_frequency, alpha))

    intervals  = np.array(result["intervals"])
    freq = np.array(observed_frequency)

    midpoints = (intervals[:-1] + intervals[1:]) / 2

    data = np.repeat(midpoints, freq)

    skew_value = skew(data)
    kurt_excess = kurtosis(data)
    kurt_total = kurtosis(data, fisher=False)

    print()
    print("Midpoints:", midpoints)
    print("Skewness:", skew_value)
    print("Kurtosis Excess:", kurt_excess)
    print("Kurtosis Total:", kurt_total)
    return

def is_uniform_distribution(subject : str,  data: np.ndarray | pd.Series, n_intervals: int , alpha : np.float64 = 0.05):
    result = count_values_in_intervals(data, n_intervals)
    observed_frequency : np.ndarray = result["counts"]
    print()
    hmu.print_centered("Subject : " + subject, 108, "*")
    print("Min:", result["min"])
    print("Max:", result["max"])
    print("Intervals:", result["intervals"])
    print("Observed Frequency : ", observed_frequency)
    print()
    print(produce_null_hypothesis_table_is_uniform_distribution(observed_frequency, alpha))
    return