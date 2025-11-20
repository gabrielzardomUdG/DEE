import pandas as pd
import math
import handmade_utils.utils as hmu

#Load data
all_students    : pd.DataFrame = pd.read_csv('source/data_science_student_marks.csv')
non_berlin       : pd.DataFrame = all_students[all_students['location'] != 'Berlin']
berlin_students  : pd.DataFrame = all_students[all_students['location'] == 'Berlin']

#English marks average em_average
all_students_em_average     :   float   =   all_students['english_marks'].mean()
non_berlin_students_em_average :   float   =   non_berlin['english_marks'].mean()
berlin_students_em_average   :   float   =   berlin_students['english_marks'].mean()

nAllStudents    : int = len(all_students)
nNonBerlin       : int = len(non_berlin)
nBerlin          : int = len(berlin_students)

#Averages
hmu.print_centered("English Marks Averages",40,"*")
print(f"All students english mark average : {all_students_em_average:.6f}, N : {nAllStudents}")
print(f"Non-Berlin Students english mark average: {non_berlin_students_em_average:.6f}, N : {nNonBerlin}")
print(f"Berlin students english mark average: {berlin_students_em_average:.6f}, N : {nBerlin}")
print()

#Average Differences
allStudents_nonBerlinStudents_avgDifference      :   float   =   math.fabs(all_students_em_average - non_berlin_students_em_average)
allStudents_berlinStudents_avgDifference         :   float   =   math.fabs(all_students_em_average - berlin_students_em_average)
berlinStudents_nonBerlinStudents_avgDifference    :   float   =   math.fabs(berlin_students_em_average - non_berlin_students_em_average)

hmu.print_centered("English Marks Average Differences",40,"*")
print(f"All students and non-berlin students average difference : {allStudents_nonBerlinStudents_avgDifference:.6f}")
print(f"All students and berlin students average difference  : {allStudents_berlinStudents_avgDifference:.6f}")
print(f"Berlin students and non-berlin students : {berlinStudents_nonBerlinStudents_avgDifference:.6f}")
print()

#Standard deviation
all_students_em_standard_deviation              :   float   =   all_students['english_marks'].std()
non_berlin_em_average_standard_deviation         :   float   =   non_berlin['english_marks'].std()
berlin_students_em_average_standard_deviation    :   float   =   berlin_students['english_marks'].std()

hmu.print_centered("Standard Deviation",40,"*")
print(f"All students english mark standard deviation : {all_students_em_standard_deviation:.6f}")
print(f"Non-Berlin Students english mark standard deviation: {non_berlin_em_average_standard_deviation:.6f}")
print(f"Berlin students english mark standard deviation: {berlin_students_em_average_standard_deviation:.6f}")
print()

#Joint Standard Deviation
allStudents_nonBerlinStudents_jointStandardDeviation     :   float   =   (hmu.joint_standard_deviation
             (nAllStudents, all_students_em_standard_deviation, nNonBerlin, non_berlin_em_average_standard_deviation))

allStudents_berlinStudents_jointStandardDeviation        :   float   =   (hmu.joint_standard_deviation
             (nAllStudents, all_students_em_standard_deviation, nBerlin, berlin_students_em_average_standard_deviation))

berlinStudents_nonBerlinStudents_jointStandardDeviation   :   float   =   (hmu.joint_standard_deviation
             (nBerlin, berlin_students_em_average_standard_deviation, nNonBerlin, non_berlin_em_average_standard_deviation))

hmu.print_centered("English Marks Joint Standard Deviation",40,"*")
print(f"All students and non-berlin students joint standard deviation : {allStudents_nonBerlinStudents_jointStandardDeviation:.6f}")
print(f"All students and berlin students joint standard deviation  : {allStudents_berlinStudents_jointStandardDeviation:.6f}")
print(f"Berlin students and non-berlin students joint standard deviation: {berlinStudents_nonBerlinStudents_jointStandardDeviation:.6f}")
print()

#Z value at 95% value between two sets
z : float = 1.96

#Joint margin of error
allStudents_nonBerlinStudents_jointMarginOfError     :   float   =   hmu.joint_margin_of_error(z, allStudents_nonBerlinStudents_jointStandardDeviation, nAllStudents, nNonBerlin)
allStudents_berlinStudents_jointMarginOfError        :   float   =   hmu.joint_margin_of_error(z, allStudents_berlinStudents_jointStandardDeviation, nAllStudents, nBerlin)
berlinStudents_nonBerlinStudents_jointMarginOfError   :   float   =   hmu.joint_margin_of_error(z, berlinStudents_nonBerlinStudents_jointStandardDeviation, nBerlin, nNonBerlin)

hmu.print_centered("English Marks Joint Margin Of Error",40,"*")
print(f"All students and non-berlin students joint margin of error : {allStudents_nonBerlinStudents_jointMarginOfError:.6f}")
print(f"All students and berlin students joint margin of error  : {allStudents_berlinStudents_jointMarginOfError:.6f}")
print(f"Berlin students and non-berlin students joint margin of error: {berlinStudents_nonBerlinStudents_jointMarginOfError:.6f}")
print()

#Average of averages and limits
allStudents_nonBerlinStudents_aoa : float = (all_students_em_average + non_berlin_students_em_average) / 2
allStudents_berlinStudents_aoa : float = (all_students_em_average + berlin_students_em_average) / 2
berlinStudents_nonBerlinStudents_aoa : float = (berlin_students_em_average + berlin_students_em_average) / 2

aS_nT_LcL : float   =   allStudents_nonBerlinStudents_aoa - allStudents_nonBerlinStudents_jointMarginOfError
aS_nT_ScL : float   =   allStudents_nonBerlinStudents_aoa + allStudents_nonBerlinStudents_jointMarginOfError

aS_t_LcL : float = allStudents_berlinStudents_aoa - allStudents_berlinStudents_jointMarginOfError
aS_t_ScL : float = allStudents_berlinStudents_aoa + allStudents_berlinStudents_jointMarginOfError

t_nt_LcL : float = berlinStudents_nonBerlinStudents_aoa - berlinStudents_nonBerlinStudents_jointMarginOfError
t_nt_ScL : float = berlinStudents_nonBerlinStudents_aoa + berlinStudents_nonBerlinStudents_jointMarginOfError

hmu.print_centered("Average Of Averages",40,"*")
print(f"All students and non-berlin students average of averages : {allStudents_nonBerlinStudents_aoa:.6f}, LcL : {aS_nT_LcL : .2f} , UcL : {aS_nT_ScL : .2f}")
print(f"All students and berlin students average of averages  : {allStudents_berlinStudents_aoa:.6f}, LcL : {aS_t_LcL : .2f} , UcL : {aS_t_ScL : .2f}")
print(f"Berlin students and non-berlin students average of averages: {berlinStudents_nonBerlinStudents_aoa:.6f}, LcL : {t_nt_LcL : .2f} , UcL : {t_nt_ScL : .2f}")
print()

#print(hmu.joint_margin_of_error(z, 0.77135133, 40, 35))
#print(hmu.joint_standard_deviation(40, 0.75871, 35, 0.7856))