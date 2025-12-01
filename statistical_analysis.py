import numpy as np

import handmade_utils.sudents_agrupator as sa
import handmade_utils.last_analysis as la
import handmade_utils.graph_maker as gm
import handmade_utils.utils as hmu
import handmade_utils.randomness_analysis_of_the_sample as raos


import pandas as pd

from handmade_utils.data_holder import DataHolder

#Pandas display options :
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # no max width for output
pd.set_option('display.max_colwidth', None)  # prevent column text trimming

all_students                :   pd.DataFrame        = pd.read_csv('source/data_science_student_marks.csv')
#students_by_location        :   list[DataHolder]    = sa.get_dh_list_of_students_by_location(all_students)
subject_marks_by_location   :   list[DataHolder]    = sa.get_dh_list_of_subject_by_location(all_students)
observed_frequency_for_randomness_analysis : pd.DataFrame = sa.create_subject_x_location_average_table(all_students)

alpha : np.float64 = 0.05
graph_lower_y : int = 0
graph_upper_y : int = 50

hmu.print_centered("ANOVA ANALYSIS", 108, "=")
la.visit_all_subjects_anova(all_students, alpha,False)
hmu.print_centered("RANDOM ANALYSIS", 108, "=")
raos.random_analysis(observed_frequency_for_randomness_analysis, alpha)
print()
hmu.print_centered("IS_NORMAL_DISTRIBUTION", 108, "=")
la.visit_all_subjects_is_normal_distribution(all_students, 8 , alpha)
hmu.print_centered("", 108, "=")
hmu.print_centered("IS_UNIFORM_DISTRIBUTION", 108, "=")
la.visit_all_subjects_is_uniform_distribution(all_students, 8, alpha)
hmu.print_centered("", 108, "=")
#gm.graph_dh_list(students_by_location, 70, 100, 4)
gm.graph_dh_list(subject_marks_by_location, 70, 100, 8, graph_lower_y, graph_upper_y)

