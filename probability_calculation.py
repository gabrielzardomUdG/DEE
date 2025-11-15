from typing import List, Dict

import pandas as pd
import handmade_utils.utils as hmu

def probability(numerator : float, denominator : float) -> float:
    return (100/denominator) * numerator

def probability_dict(dict_input: dict) -> dict:
    total = sum(dict_input.values())
    return {key: (value / total) * 100 for key, value in dict_input.items()}

def print_probabilities(prob_dict: dict):
    formatted = {k: f"{v:.2f}" for k, v in prob_dict.items()}
    print(formatted)

def print_quartile(q_Marks: pd.DataFrame, title : str, totalStudents : int, column : str) -> None:
    hmu.print_centered(title, 50, "*")
    group_limits =  hmu.get_group_ranges(q_Marks, column)
    numberOfStudentsInThisGroup : int = len(q_Marks)
    print("Number of students in this group: " + str(numberOfStudentsInThisGroup) + " , " + "lowest mark : " + str(group_limits[0])
          + ", highest mark : " + str(group_limits[1]) + ", propability of a student to be part of this group : " +
          str(round(probability(float(numberOfStudentsInThisGroup), float(totalStudents)), 2)) + "\n")
    return


def count_tokyo_bottom25(df):
    q1 = df['english_marks'].quantile(0.25)

    filtered_df = df[
        (df['location'] == "Tokyo") &
        (df['english_marks'] <= q1)
        ]

    return len(filtered_df)

# Load data
df : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')

numberOfStudents = len(df)
percentile_quarters_english : List[pd.DataFrame] = hmu.split_into_quantile_groups(df, 'english_marks', 4)
percentile_quarters_python : List[pd.DataFrame] = hmu.split_into_quantile_groups(df, 'python_marks', 4)
location_counts : Dict[str, int] = hmu.count_value_frequencies(df, "location")
location_probability : Dict[str, float] = probability_dict(location_counts)

hmu.print_centered("Total number of students", 50, "-")
print("Number of students : " + str(numberOfStudents) + "\n")

_1q_eMarks : pd.DataFrame = percentile_quarters_english[0]
_1q_pMarks : pd.DataFrame = percentile_quarters_python[0]

print_quartile(_1q_eMarks, "_Students bottom 25% quartile english_", numberOfStudents, "english_marks");
print_quartile(_1q_pMarks, "_Students bottom 25% quartile python_", numberOfStudents, "python_marks");

hmu.print_centered("Location count and probability", 50, "*")
print(location_counts)
print_probabilities(location_probability)
print()

hmu.print_centered("Probability exercises", 50, "*")
print(f"Probability of intersection of two independent events with the same occurrence probability (Student being in Los Angeles / Tokyo locations): {hmu.prob_intersection(location_probability['Tokyo'], location_probability['Los Angeles']): .2f}")
raw_probability_of_being_in_the_bottom_25_english : float = probability(float(len(_1q_eMarks)), float(numberOfStudents))
raw_probability_of_being_in_tokyo : float = location_probability['Tokyo']
union_of_two_independent_events : float = hmu.prob_union(raw_probability_of_being_in_tokyo, raw_probability_of_being_in_the_bottom_25_english, None)
print(f"The union of two or more independent events (Student being from Tokyo or be in the bottom 25% in the english subject): ", round(union_of_two_independent_events, 2))
intersection_of_two_independent_events : float = hmu.prob_intersection(raw_probability_of_being_in_tokyo, raw_probability_of_being_in_the_bottom_25_english)
print(f"The intersection of two or more independent events (Student being from Tokyo and be in the bottom 25% in the english subject): ", round(intersection_of_two_independent_events, 2))
raw_probability_of_being_in_the_bottom_25_python : float = probability(float(len(_1q_pMarks)), float(numberOfStudents))
occurrence_of_everything : float = hmu.prob_intersection(intersection_of_two_independent_events, raw_probability_of_being_in_the_bottom_25_python)
print(f"The intersection of two or more events with ocurrance of everything (Student being from Tokyo and be in the bottom 25% in the english subject and the python subject): ", round(occurrence_of_everything, 2))
print(f"Repetition of an event in a muestral space (Picking 4 students of the bottom 25% in the english subject): ", round(hmu.repetition_in_muestral_space(numberOfStudents, len(_1q_eMarks), 4, 4), 2))
print("Real number of students in Tokyo in the bottom 25% of english marks : ", count_tokyo_bottom25(df))
