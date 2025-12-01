import numpy as np

import pandas as pd
import handmade_utils.goodness_of_fit_analysis_of_a_normal_distribution as gad
import handmade_utils.utils as hmu
import handmade_utils.anova_adeva as anodeva
import handmade_utils.sudents_agrupator as sa


def visit_all_subjects_is_normal_distribution(all_students : pd.DataFrame, partition_number : int = 8, alpha : np.float64 = 0.05) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        result_of_subject : np.ndarray = all_students[subject].to_numpy()
        gad.is_normal_distribution(subject, result_of_subject,  partition_number, alpha)

    return

def visit_all_subjects_is_uniform_distribution(all_students : pd.DataFrame, partition_number : int = 8, alpha : np.float64 = 0.05) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        result_of_subject : np.ndarray = all_students[subject].to_numpy()
        gad.is_uniform_distribution(subject, result_of_subject,  partition_number, alpha)

    return

def visit_all_subjects_anova(all_students    : pd.DataFrame, alpha : np.float64 = 0.05, print_input_table : bool = False) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        hmu.print_centered(subject, 120, "=")
        students_by_mark : pd.DataFrame = sa.get_subject_marks_by_location(all_students, subject)
        anodeva.anova_adeva_analysis(students_by_mark, subject, alpha, print_input_table)

    return