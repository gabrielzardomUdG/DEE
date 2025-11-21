from math import comb

import pandas as pd
import handmade_utils.utils as hmu


def print_binomial_table(n: int, positive_probability: float) -> None:

    p = positive_probability
    q = 1 - p

    print(f"{'Y':<3} {'C(N,Y)':<8} {'p^Y':<12} {'q^(N-Y)':<12} {'P(Y)':<12}")
    print("-" * 50)

    for y in range(n + 1):
        c = comb(n, y)
        p_pow = p ** y
        q_pow = q ** (n - y)
        probability = c * p_pow * q_pow

        print(f"{y:<3} {c:<8} {p_pow:<12.6f} {q_pow:<12.6f} {probability:<12.6f}")

#Load data
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')
students_in_non_english_speaking_cities : pd.DataFrame = all_students[all_students['location'].isin(['Berlin','Paris','Tokyo'])]

nStudents : int = len(all_students)
nStudents_inNonSpeakingEnglishCities : int = len(students_in_non_english_speaking_cities)
nStudents_inSpeakingEnglishCities : int = nStudents - nStudents_inNonSpeakingEnglishCities

hmu.print_centered("Number of students",40,"*")
print("Number of students : " + str(nStudents))
print("Number of students in Non-Speaking-English-Cities: " + str(nStudents_inNonSpeakingEnglishCities))
print("Number of students in Speaking-English-Cities: " + str(nStudents_inSpeakingEnglishCities))

percentageOfStudentsInNonEnglishSpeakingCountries : float = (100/nStudents) * nStudents_inNonSpeakingEnglishCities

print("Percentage of students in Non English Speaking Cities : " + str(round(percentageOfStudentsInNonEnglishSpeakingCountries/100,2)) + "%")

hmu.print_centered("Binomial Table of Students in Non english Speaking Cities",40,"*")
print_binomial_table(nStudents, round(percentageOfStudentsInNonEnglishSpeakingCountries/100,2))

#print_binomial_table(8, 0.35)
print(hmu.count_by_group_in_range())