from itertools import zip_longest
import handmade_utils.utils as hmu
from typing import List

import numpy as np
import pandas as pd

from scipy.stats import f

#Summary Header SH
SH_IDENTIFIER : str = "IDENTIFIER"
SH_SUMMATORY : str = "SUMMATORY"
SH_ELEMENTS : str = "N_ELEMENTS"
SH_MEAN : str = "MEAN"
#Suma de Cuadrados / Sum of Squares
SH_SC_SS_SUMSQRS : str = "SC_SS"
#Correction Factor / COrrecion de Medias
SH_COM_CF : str = "CF_COM"
#Suma del total de cuadrados / Tosal sum of squares
SH_STC_TSS : str = "STC_TSS"
SH_SMPVRNC : str = "SAMPLE_VARIANCE"

AH_COM : str = "CF_COM"
#Population Sum of Squares / Suma de Cuadrados de la poblacion
AH_PSS_SCP : str = "PSS_SCP"
#Sum Of Suqares Error/ Residual Sum Of Squares/ Suma de Cuadroados Error
AH_SSE_RSS_SCE : str = "SSE_RSS_SCE"
#Total Sum of Suqares / Suma de Cuadrados Total Con Variabilidad del Sistema (Y)
AH_TSS_SCY : str = "TSS_SCY"
#Mean Square Between / Cuadrado Media de la Poblacion
AH_MSB_CMP : str = "MSB_CMP"
#Mean Square Error / Cuadrado Medio del Error
AH_MSE_CME : str = "MSE_CME"
AH_SYSTEM_STDDEV : str = "SYSTEM_STD"
AH_CALCULATEDF_FCALCULADA : str = "F_CAL"


ANDVH_ORIGIN : str = "ORIGIN"
#Degrees of freedom / Grados de Libertad
ANDVH_DF_GL : str = "DF_GL"
#Sum of squares / Suma de Cuadrados
ANDVH_SS_SC : str = "SS_SC"
#Mean Square Between / Cuadrado Medio
ANDVH_MS_CM : str = "MS_CM"
ANDVH_CALCULATEDF_FCALCULADA : str = "F_CAL"
ANDVH_BASE : str = "BASE"
ANDVH_DECISION : str = "DECISION"


SUMMARY_HEADERS     : list = [SH_IDENTIFIER, SH_SUMMATORY, SH_ELEMENTS, SH_MEAN, SH_SC_SS_SUMSQRS, SH_COM_CF, SH_STC_TSS, SH_SMPVRNC]
ANALYSIS_HEADERS    : list = [AH_COM, AH_PSS_SCP, AH_SSE_RSS_SCE, AH_TSS_SCY, AH_MSB_CMP, AH_MSE_CME, AH_SYSTEM_STDDEV, AH_CALCULATEDF_FCALCULADA]
ANODEVA_HEADERS     : list = [ANDVH_ORIGIN, ANDVH_DF_GL, ANDVH_SS_SC, ANDVH_MS_CM, ANDVH_CALCULATEDF_FCALCULADA, ANDVH_BASE, ANDVH_DECISION]

def produce_example_df() -> pd.DataFrame:
    data = list(zip_longest(
        [77, 29, 40, 13, 81],
        [19, 66, 15, 32, 20, 22, 23, 19, 23, 79],
        [21, 68, 89, 99, 56, 68, 73, 37],
        [34, 59, 45, 34, 10, 40, 37],
        fillvalue=None
    ))
    return pd.DataFrame(data, columns=["A", "B", "C", "D"])


def produce_math_of_summary_row(numeric_data : np.ndarray) -> List:
    #This data is being explained, that is the reason for the verbosity

    #Remove NaNs
    numeric_data = numeric_data[~np.isnan(numeric_data)]

    summatory : np.float64 = numeric_data.sum()
    n_elements : int = len(numeric_data)
    mean : np.float64 = numeric_data.mean()
    #Sum of squares
    ss : np.float64 = np.sum(numeric_data ** 2)
    #Correction factor
    cf_com : np.float64 = (summatory**2) / n_elements
    #Total sum of squares
    local_tss : np.float64 = ss - (summatory**2 / n_elements)
    #Sample variance
    local_sample_variance : np.float64 = ((ss - cf_com) / (n_elements - 1))**0.5
    output : List = [summatory, n_elements, mean, ss, cf_com, local_tss, local_sample_variance]
    return output

def produce_summary_row(identifier : str, numeric_data : np.ndarray) -> List:
    output : list = produce_math_of_summary_row(numeric_data)
    output.insert(0, identifier)
    return output

def produce_last_summary_row(summary_table : pd.DataFrame) -> List:
    system_summatory : np.float64 = summary_table[SH_SUMMATORY].to_numpy().sum()
    system_nlen : np.float64 = summary_table[SH_ELEMENTS].to_numpy().sum()
    system_mean : np.float64 = summary_table[SH_MEAN].to_numpy().mean()
    system_sumsqrs : np.float64 = summary_table[SH_SC_SS_SUMSQRS].to_numpy().sum()
    system_cf_com : np.float64 = summary_table[SH_COM_CF].to_numpy().sum()
    system_tss : np.float64 = summary_table[SH_STC_TSS].to_numpy().sum()
    return ["System Sums And Mean", system_summatory, system_nlen, system_mean, system_sumsqrs, system_cf_com, system_tss, np.nan]

def produce_summary_table_from(input_example : pd.DataFrame) -> pd.DataFrame:
    output_data : pd.DataFrame = pd.DataFrame(columns=SUMMARY_HEADERS)

    for col_name in input_example.columns:
        output_data.loc[len(output_data)] = produce_summary_row(col_name, input_example[col_name].to_numpy())
        
    system_row : List = produce_last_summary_row(output_data)
    output_data.loc[len(output_data)] = system_row

    return output_data

def produce_anova_adeva_analysis(summary_table : pd.DataFrame) -> pd.DataFrame:
    output_data : pd.DataFrame = pd.DataFrame(columns=ANALYSIS_HEADERS)
    last_row : pd.Series = summary_table.iloc[-1]

    #Number of real groups es the number of rows of the table minus one, that minus one is because the last row is a summary. Not a real group.
    number_of_real_groups : int = len(summary_table) - 1
    #Formula for degree of freedoms in groups is "n - 1"
    degrees_of_freedom_groups : int = number_of_real_groups - 1
    #Formula for degree of freedoms in total elements is "n_total_elements - number_of_real_groups"
    degres_of_freedom_n_elements : int  = last_row[SH_ELEMENTS] - number_of_real_groups

    cf_com : np.float64 = (last_row[SH_SUMMATORY]**2) / last_row[SH_ELEMENTS]
    pss_ss_scp : np.float64 = last_row[SH_COM_CF] - cf_com
    ssr_ress_sce : np.float64 = last_row[SH_STC_TSS]
    tss_scy : np.float64 = last_row[SH_SC_SS_SUMSQRS] - cf_com
    cmp_msb : np.float64 = pss_ss_scp / degrees_of_freedom_groups
    mse_cme : np.float64 = ssr_ress_sce / degres_of_freedom_n_elements
    system_stddev : np.float64 = (tss_scy / (last_row[SH_ELEMENTS] - 1))**0.5
    f_calculated : np.float64 = (cmp_msb / mse_cme)

    row : List = [cf_com, pss_ss_scp, ssr_ress_sce, tss_scy, cmp_msb, mse_cme, system_stddev, f_calculated]
    output_data.loc[len(output_data)] = row

    return output_data

def generate_anodeva_table(anodeva_analysis : pd.DataFrame, confidance_coeficient0to1 : np.float64, total_groups : int, total_elements : int) -> pd.DataFrame:
    output_data : pd.DataFrame = pd.DataFrame(columns=ANODEVA_HEADERS)

    degrees_of_freedom_groups : int = total_groups - 1
    degrees_of_freedom_elements : int = total_elements - total_groups
    f_critical = f.ppf(1 - confidance_coeficient0to1, degrees_of_freedom_groups, degrees_of_freedom_elements)
    f_calculada = anodeva_analysis[AH_CALCULATEDF_FCALCULADA].iloc[-1]
    decision : str = "Accepted" if f_calculada < f_critical else "Denied"

    population_row : list = ["Population", degrees_of_freedom_groups, anodeva_analysis[AH_PSS_SCP].iloc[-1],anodeva_analysis[AH_MSB_CMP].iloc[-1], 
                             f_calculada, f_critical,  decision ]
    error_row : list = ["Error", degrees_of_freedom_elements, anodeva_analysis[AH_SSE_RSS_SCE].iloc[-1], anodeva_analysis[AH_MSE_CME].iloc[-1], " ", " ", " "]
    system_row : list = ["System", degrees_of_freedom_groups + degrees_of_freedom_elements, " ", " ", " ", " ", " "]
    
    output_data.loc[len(output_data)] = population_row
    output_data.loc[len(output_data)] = error_row
    output_data.loc[len(output_data)] = system_row

    return output_data
    
def example_practice(confidance_coeficient0to1 : np.float64 = 0.05):

    input_example : pd.DataFrame = produce_example_df()
    summary_table : pd.DataFrame = produce_summary_table_from(input_example)
    anodeva_analysis : pd.DataFrame = produce_anova_adeva_analysis(summary_table)
    anodeva_table : pd.DataFrame = generate_anodeva_table(anodeva_analysis, confidance_coeficient0to1, len(summary_table) - 1, summary_table[SH_ELEMENTS].iloc[-1])
    print()
    hmu.print_centered("Input Data", 108, "*")
    print(input_example)
    print()
    hmu.print_centered("Summary Table", 108, "*")
    print(summary_table,"\n")
    hmu.print_centered("Auxiliar Analysis Table", 108, "*")
    print(anodeva_analysis, "\n")
    hmu.print_centered("ANOVA / ADEVA Table", 108, "*")
    print(anodeva_table, "\n")
    return

def anova_adeva_analysis(input_data : pd.DataFrame, subject : str, confidance_coeficient0to1 : np.float64 = 0.05) -> None:
    summary_table : pd.DataFrame = produce_summary_table_from(input_data)
    anodeva_analysis : pd.DataFrame = produce_anova_adeva_analysis(summary_table)
    anodeva_table : pd.DataFrame = generate_anodeva_table(anodeva_analysis, confidance_coeficient0to1, len(summary_table) - 1, summary_table[SH_ELEMENTS].iloc[-1])
    print()
    hmu.print_centered("Input Data : " +  subject, 108, "*")
    print(input_data)
    print()
    hmu.print_centered("Summary Table : " +  subject, 108, "*")
    print(summary_table,"\n")
    hmu.print_centered("Auxiliar Analysis Table : " +  subject, 108, "*")
    print(anodeva_analysis, "\n")
    hmu.print_centered("ANOVA / ADEVA Table : " +  subject, 108, "*")
    print(anodeva_table, "\n")
    return
