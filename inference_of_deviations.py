import handmade_utils.inference_of_deviations as iod
import handmade_utils.sudents_agrupator as sa

import pandas as pd

def visit_all_subjects(all_students    : pd.DataFrame, print_input_table : bool = True) -> None:
    cols = list(all_students.columns[3:])

    for subject in cols:
        students_by_mark : pd.DataFrame = sa.get_subject_marks_by_location(all_students, subject)
        iod.anova_adeva_analysis(students_by_mark, subject, 0.05, print_input_table)

    return

#Pandas display options :
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # no max width for output
pd.set_option('display.max_colwidth', None)  # prevent column text trimming


# Setting variables for easy change :
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')

#This is the class example, just uf you want to see it in python, remove the #
#iod.example_practice()

visit_all_subjects(all_students, False)