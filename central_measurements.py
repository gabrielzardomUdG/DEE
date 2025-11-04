import pandas as pd
import matplotlib.pyplot as plt
import handmade_utils.utils as hmu

df = pd.read_csv('source/data_science_student_marks.csv')

#------------------------------Central Measurements--------------------------------------
hmu.print_centered("Central Measurements",32,"*")

"""
The array 'bins' defines our class limits, from the minimum to the maximum values.
The 'labels' array provides readable labels that represent these limits in our data graphs.
The 'class_intervals' object stores the grouped frequency data and is used to plot our graphs.
"""

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0–10', '11–20', '21–30', '31–40', '41–50', '51–60', '61–70', '71–80', '81–90', '91–100']
class_intervals = hmu.get_class_intervals_info(bins, labels, df, "english_marks")

#------------------------------Data average--------------------------------------
hmu.print_centered("Data average",32,"*")
average = (df['english_marks'].sum() / len(df['english_marks']))
print("General Average")
print(average.round(2))
#------------------------------Class Information--------------------------------------
hmu.print_centered("Class Information",32,"*")
print("Class Information")
print(class_intervals)


#------------------------------Graphs--------------------------------------
# --- Combined Histogram and Frequency Polygon ---
plt.figure(figsize=(10, 6))

# Histogram
plt.bar(labels, class_intervals['Frequency'].to_numpy(), color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')

# Frequency polygon (line on top of bars)
plt.plot(labels, class_intervals['Frequency'].to_numpy(), marker='o', color='red', linewidth=2, label='Frequency Polygon')

plt.xlabel("Mark Ranges")
plt.ylabel("Number of Students")
plt.title("Distribution of English Marks by Range")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()