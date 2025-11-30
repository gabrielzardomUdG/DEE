import numpy as np

import handmade_utils.utils as hmu

import pandas as pd
import handmade_utils.goodness_of_fit_analysis_of_a_normal_distribution as gad

#Pandas display options :
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # no max width for output
pd.set_option('display.max_colwidth', None)  # prevent column text trimming

def visit_all_subjects_is_normal_distribution(all_students : pd.DataFrame, print_input_table : bool = True) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        result_of_subject : np.ndarray = all_students[subject].to_numpy()
        gad.is_normal_distribution(subject, result_of_subject,  8, 0.05)

    return

def visit_all_subjects_is_uniform_distribution(all_students : pd.DataFrame, print_input_table : bool = True) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        result_of_subject : np.ndarray = all_students[subject].to_numpy()
        gad.is_uniform_distribution(subject, result_of_subject,  8, 0.05)

    return

# Setting variables for easy change :
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')

#This is the class example, just uf you want to see it in python, remove the #
#gad.example_practice()
print()
hmu.print_centered("IS_NORMAL_DISTRIBUTION", 108, "=")
visit_all_subjects_is_normal_distribution(all_students)
hmu.print_centered("IS_UNIFORM_DISTRIBUTION", 108, "=")
visit_all_subjects_is_uniform_distribution(all_students)