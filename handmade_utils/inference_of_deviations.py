from itertools import zip_longest
import handmade_utils.utils as hmu
from typing import List
from scipy.stats import chi2


import numpy as np
import pandas as pd

from scipy.stats import f

#Summary Header SH
SH_IDENTIFIER   : str = "IDENTIFIER"
SH_SUMMATORY    : str = "SUMMATORY"
SH_CARDINALITY  : str = "CARDINALITY"
SH_AVERAGE      : str = "AVERAGE"
SH_TOTAL_SUMOFSQUARES    : str = "TOTAL_SUM_OF_SQUARES"
SH_DGRSOFFRE    : str = "GRADES_OF_FREEDOM"
SH_VARIANCE     : str = "VARIANCE"
SH_NLOGOFVAR    : str = "LOG N OF VARIANCE"
SH_V_LNS_PWR2   : str = "v(lnS)^2"

NH_VARIANCE     : str = "CHI CALCULATED"
NH_NLOGOFVAR    : str = "CHI CRIT"
NH_IS_ACCEPTED  : str = "IS THE NH ACCEPTED?"


SUMMARY_HEADERS     : list = [SH_IDENTIFIER, SH_SUMMATORY, SH_CARDINALITY, SH_AVERAGE, SH_TOTAL_SUMOFSQUARES, SH_DGRSOFFRE, SH_VARIANCE, SH_NLOGOFVAR, SH_V_LNS_PWR2]
CHI_COMP_HEADERS    : list = [NH_VARIANCE, NH_NLOGOFVAR, NH_IS_ACCEPTED]
def produce_example_df() -> pd.DataFrame:
    data = list(zip_longest(
        [6, 9, 3],
        [10, 4, 9, 12, 8, 11],
        [2, 6, 4],
        fillvalue=None
    ))
    return pd.DataFrame(data, columns=["n1", "n2", "n3"])

def produce_math_of_summary_row(numeric_data : np.ndarray) -> List:
    numeric_data = numeric_data[~np.isnan(numeric_data)]
    summatory   : np.float64 = numeric_data.sum()
    cardinality : int = len(numeric_data)
    average     : np.float64 = numeric_data.mean()
    sum_squares : np.float64 = np.sum(numeric_data ** 2)
    total_sum_squares : np.float64 = sum_squares - (summatory**2 / cardinality)
    degrees_of_freedom : int = len(numeric_data) - 1
    variance : np.float64 = np.var(numeric_data, ddof=1)
    nat_log_variance : np.float64 = np.log(variance)
    v_lns_pwr2 : np.float64 = degrees_of_freedom * nat_log_variance
    output : List = [summatory, cardinality, average, total_sum_squares, degrees_of_freedom, variance, nat_log_variance, v_lns_pwr2]
    return output

def produce_summary_row(identifier : str, numeric_data : np.ndarray) -> List:
    output : list = produce_math_of_summary_row(numeric_data)
    output.insert(0, identifier)
    return output

def produce_last_summary_row(summary_table : pd.DataFrame) -> List:
    system_summatory : np.float64 = summary_table[SH_SUMMATORY].to_numpy().sum()
    system_cardinality : int = summary_table[SH_CARDINALITY].to_numpy().sum()
    system_average : int = system_summatory / system_cardinality
    system_tss : np.float64 = summary_table[SH_TOTAL_SUMOFSQUARES].to_numpy().sum()
    system_degfreed : int = summary_table[SH_DGRSOFFRE].to_numpy().sum()
    system_variance : int = summary_table[SH_TOTAL_SUMOFSQUARES].to_numpy().sum() / system_degfreed
    system_nat_log_variance : np.float64 = np.log(system_variance)
    system_v_lns_pwr2 : np.float64 = summary_table[SH_V_LNS_PWR2].to_numpy().sum()
    return ["Design/System Sums, Mean And Logs", system_summatory, system_cardinality, system_average, system_tss, system_degfreed, 
            system_variance, system_nat_log_variance, system_v_lns_pwr2]
    return None

def produce_summary_table_from(input_table : pd.DataFrame) -> pd.DataFrame:
    output_data : pd.DataFrame = pd.DataFrame(columns=SUMMARY_HEADERS)

    for col_name in input_table.columns:
        output_data.loc[len(output_data)] = produce_summary_row(col_name, input_table[col_name].to_numpy())
        
    system_row : List = produce_last_summary_row(output_data)
    output_data.loc[len(output_data)] = system_row

    return output_data

def produce_null_hypothesis_table(summary_table : pd.DataFrame, alpha : np.float64) -> pd.DataFrame:

    output_data : pd.DataFrame = pd.DataFrame(columns=CHI_COMP_HEADERS)

    ngroups : int = len(summary_table) - 1
    ngroups_degrees_of_freedom : int = ngroups - 1
    chi2_crit : np.float64 = chi2.ppf(1 - alpha, ngroups_degrees_of_freedom)
    chi2_calculated : np.float64 = (summary_table[SH_DGRSOFFRE].iloc[-1] * summary_table[SH_NLOGOFVAR].iloc[-1]) - summary_table[SH_V_LNS_PWR2].iloc[-1]
    decision : str = "Accepted" if chi2_calculated < chi2_crit else "Denied"
    row : list = [chi2_calculated, chi2_crit, decision]
    output_data.loc[len(output_data)] = row

    return output_data

def example_practice(alpha : np.float64 = 0.05):

    input_example : pd.DataFrame = produce_example_df()
    summary_table : pd.DataFrame = produce_summary_table_from(input_example)
    null_hypothesis_table : pd.DataFrame = produce_null_hypothesis_table(summary_table, alpha)

    hmu.print_centered("Summary Table : ", 108, "*")
    print(summary_table,"\n")

    hmu.print_centered("CHI COMPARISON : ", 108, "*")
    print(null_hypothesis_table,"\n")

    return

def anova_adeva_analysis(input_data : pd.DataFrame, subject : str, alpha : np.float64 = 0.05, print_input_data : bool = True) -> None:


    summary_table : pd.DataFrame = produce_summary_table_from(input_data)
    null_hypothesis_table : pd.DataFrame = produce_null_hypothesis_table(summary_table, alpha)

    hmu.print_centered(subject, 120, "=")

    if print_input_data :
        print()
        hmu.print_centered("Input Data : " +  subject, 108, "*")
        print(input_data)
        print()
    hmu.print_centered("Summary Table : " +  subject, 108, "*")
    print(summary_table,"\n")
    hmu.print_centered("CHI COMPARISON : " +  subject, 108, "*")
    print(null_hypothesis_table, "\n")

    return