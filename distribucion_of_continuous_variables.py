import pandas as pd
from scipy.stats import norm
import handmade_utils.utils as hmu

#Subject being examined
subject : str = 'english_marks'

#Load data
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')
nStudent : int = len(all_students)

#Data
average : float = all_students[subject].mean()
std_dvt : float = all_students[subject].std()

#average = 32
#std_dvt = 4

lower_limit : float = 75
upper_limit : float = 90
slctd_number : float = 95

#lower_limit = 34
#upper_limit = 38
#slctd_number = 28

z_lower : float = (lower_limit - average) / std_dvt
z_upper : float = (upper_limit - average) / std_dvt
z_selected : float = (slctd_number - average) / std_dvt

p_lower : float =   norm.cdf(z_lower)
p_upper : float =   norm.cdf(z_upper)
p_slctd : float =   norm.cdf(z_selected)

py : float = p_upper - p_lower

hmu.print_centered("Values of the practice",40,"*")
print("Average of the students in", subject, ":" ,average)
print("Standard deviation :", std_dvt)
#print("Lower limit :", lower_limit)
#print("Upper limit :", upper_limit)
#print("Lower Z :", z_lower)
#print("Upper Z :", z_upper)
#print("Selected Z :", z_selected)
#print("Lower P :", p_lower)
#print("Upper P :", p_upper)
#print("Selected P :", p_slctd)
print("P(Y), Probability of a number being in the range  :",  lower_limit,  "-", upper_limit, "es de:",py)
print("Probability of a number being below :",  slctd_number, "is", p_slctd)
print("Probability of a number being above :",  slctd_number, "is", 1 - p_slctd)
print("Probability of a number being below :",  upper_limit, "is", p_upper)
print("Probability of a number being above :",  upper_limit, "is", 1 - p_upper)
print("Probability of a number being below :",  lower_limit, "is", p_lower)
print("Probability of a number being above :",  lower_limit, "is", 1 - p_lower)



hmu.print_centered("Students being inside the range",40,"*")
print(hmu.count_by_group_in_range(all_students, subject, lower_limit, upper_limit, 'location'))
hmu.print_centered("Students being below the range",40,"*")
print(hmu.count_by_group_below(all_students, subject, lower_limit, 'location'))
hmu.print_centered("Students being above the range",40,"*")
print(hmu.count_by_group_above(all_students, subject, upper_limit, 'location'))
