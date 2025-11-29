import numpy as np
import pandas as pd
import handmade_utils.utils as hmu
from scipy.stats import chi2

#Summary Header SH
CJ   : str = "CJ"
FI   : str = "FI"

NH_VARIANCE     : str = "CHI CALCULATED"
NH_NLOGOFVAR    : str = "CHI CRIT"
NH_IS_ACCEPTED  : str = "IS THE NH ACCEPTED?"

CHI_COMP_HEADERS    : list = [NH_VARIANCE, NH_NLOGOFVAR, NH_IS_ACCEPTED]

def produce_example_df()-> pd.DataFrame:
    return pd.DataFrame(
        {
            "Homicidios": [np.float64(7), np.float64(17), np.float64(12)],
            "Robos":      [np.float64(163), np.float64(239), np.float64(98)],
            "Asaltos":    [np.float64(191), np.float64(109), np.float64(278)],
            "Secuestros": [np.float64(110), np.float64(201), np.float64(56)],
            "Otros":      [np.float64(47), np.float64(54), np.float64(17)]
        },
        index=["I", "II", "III"],
        dtype=np.float64
    )


def add_sum_row_n_col(input_table: pd.DataFrame) -> pd.DataFrame:
    input_table.loc[CJ] = input_table.sum()
    input_table[FI]     = input_table.sum(axis=1)
    return input_table

def create_expected_frequencies_tables(input_table: pd.DataFrame) -> pd.DataFrame:
    output_data : pd.DataFrame = pd.DataFrame(columns=input_table.columns[:-1], index=input_table.index[:-1], dtype=np.float64)
    divisor : np.float64 = input_table.at[CJ,FI]

    for column in output_data.columns :
        for row in output_data.index :
            output_data.at[row, column] = (input_table.at[row, FI] * input_table.at[CJ, column]) / divisor

    return output_data

def calculate_chi(observed_frequency : pd.DataFrame, expected_frequency : pd.DataFrame) -> np.float64:

    chi_calculated : np.float64 = 0

    for row in expected_frequency.index :
        for column in expected_frequency.columns :
            fo : np.float64 = observed_frequency.at[row, column]
            fe : np.float64 = expected_frequency.at[row, column]
            chi_calculated += (fo - fe) ** 2 / fe

    return chi_calculated

def produce_null_hypothesis_table(observed_frequency : pd.DataFrame, expected_frequency : pd.DataFrame, alpha : np.float64) -> pd.DataFrame:
    degrees_of_freedom : int = (len(observed_frequency.index) - 2) * (len(observed_frequency.columns) - 2)
    chi2_calculated : np.float64 = calculate_chi(observed_frequency,expected_frequency)
    chi2_crit : np.float64 = chi2.ppf(1 - alpha, degrees_of_freedom)

    output_data : pd.DataFrame = pd.DataFrame(columns=CHI_COMP_HEADERS)
    decision : str = "Accepted" if chi2_calculated < chi2_crit else "Denied"
    row : list = [chi2_calculated, chi2_crit, decision]
    output_data.loc[len(output_data)] = row
    return output_data

def example_practice(alpha : np.float64 = 0.05) :

    print()
    hmu.print_centered("Observed Frequency : ", 108, "*")
    input_example : pd.DataFrame = produce_example_df()
    add_sum_row_n_col(input_example)
    print(input_example)

    print()
    hmu.print_centered("Expected Frequency : ", 108, "*")
    expected_frequencies_table : pd.DataFrame = create_expected_frequencies_tables(input_example)
    add_sum_row_n_col(expected_frequencies_table)
    print(expected_frequencies_table)

    print()
    hmu.print_centered("Hypothesis Table : ", 108, "*")
    null_hypothesis_table : pd.DataFrame = produce_null_hypothesis_table(input_example, expected_frequencies_table, alpha)
    print(null_hypothesis_table)

    return


def random_analysis(observed_frequency : pd.DataFrame, alpha : np.float64 = 0.05) -> None:


    print()
    hmu.print_centered("Observed Frequency : ", 108, "*")
    add_sum_row_n_col(observed_frequency)
    print(observed_frequency)

    print()
    hmu.print_centered("Expected Frequency : ", 108, "*")
    expected_frequencies_table : pd.DataFrame = create_expected_frequencies_tables(observed_frequency)
    add_sum_row_n_col(expected_frequencies_table)
    print(expected_frequencies_table)

    print()
    hmu.print_centered("Hypothesis Table : ", 108, "*")
    null_hypothesis_table : pd.DataFrame = produce_null_hypothesis_table(observed_frequency, expected_frequencies_table, alpha)
    print(null_hypothesis_table)

    return