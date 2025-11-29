import handmade_utils.randomness_analysis_of_the_sample as raos
import handmade_utils.sudents_agrupator as sa

import pandas as pd


#Pandas display options :
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # no max width for output
pd.set_option('display.max_colwidth', None)  # prevent column text trimming


# Setting variables for easy change :
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')
observed_frequency : pd.DataFrame = sa.create_subject_x_location_average_table(all_students)

#raos.example_practice()
raos.random_analysis(observed_frequency, 0.05)