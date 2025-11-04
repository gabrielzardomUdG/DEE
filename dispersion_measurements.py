import pandas as pd
import matplotlib.pyplot as plt
import handmade_utils.utils as hmu

df = pd.read_csv('source/data_science_student_marks.csv')

#--------------------------------------------------------------------
print("variance_of_sample : english_marks")

data = df['english_marks'].to_numpy()
number_total_elements = len(data);
average = sum(data) / len(data)
variance_of_sample = (sum((x - average) ** 2 for x in data)) / (number_total_elements - 1)

hmu.print_centered("variance_of_sample",80,"*")
print(variance_of_sample)

hmu.print_centered("standard_deviation",80,"*")
standard_deviation = variance_of_sample ** 0.5
print(standard_deviation)

hmu.print_centered("variance_coefficient",80,"*")
variance_coefficient = standard_deviation/average
print(variance_coefficient)
