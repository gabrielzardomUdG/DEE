import math
import pandas as pd
import numpy as np
import handmade_utils.utils as hmu
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data
df = pd.read_csv('source/data_science_student_marks.csv')
data = df['english_marks'].to_numpy()

# Find values
hmu.print_centered("Values",80,"*")
global_min_value : float = np.min(data)
global_max_value : float = np.max(data)

#19.31 and 7.78 in text book
average : float = np.mean(data)
standard_deviation : float = np.std(data)

#7.04 and 31.6 in textbook
agustin_proportion_low_value : float = average - standard_deviation
agustin_proportion_high_value : float = average + standard_deviation

agustin_lowZ_first_form: float = (agustin_proportion_low_value - average) / standard_deviation
agustin_highZ_first_form: float = (agustin_proportion_high_value - average) / standard_deviation
Pmin : float = 1 - ((1 / 2) ** agustin_highZ_first_form)
Pmax : float = 1 - ((1 / 8) ** agustin_highZ_first_form)

p = 0.65
min_log_term = math.log(1 / (1 - p)) / math.log(2)
min_low_range = average - (standard_deviation * min_log_term)
min_high_range = average + (standard_deviation * min_log_term)

max_log_term = math.log(1 / (1 - p)) / math.log(8)
max_low_range = average - (standard_deviation * max_log_term)
max_high_range = average + (standard_deviation * max_log_term)

print("Statistical Summary")
print("-" * 40)
print(f"Minimum global value in array: {global_min_value:.2f}")
print(f"Maximum global value in array: {global_max_value:.2f}")
print(f"Average value: {average:.2f}")
print(f"Standard deviation: {standard_deviation:.2f}")
print()

print("Agustin's Proportion Details")
print("-" * 40)
print(f"Min/Max proportion range values: {agustin_proportion_low_value:.2f} to {agustin_proportion_high_value:.2f}")
print(f"Range z values: {agustin_lowZ_first_form:.2f} to {agustin_highZ_first_form:.2f}")
print(f"Agustin minimum proportion (Pmin): {Pmin:.2f}")
print(f"Agustin maximum proportion (Pmax): {Pmax:.2f}")
print(f"Agustin min proportion at 65%: {min_low_range:.2f} to {min_high_range:.2f}")
print(f"Agustin max proportion at 65%: {max_low_range:.2f} to {max_high_range:.2f}")
print(f"Agustin normal proportion range: {agustin_proportion_low_value:.2f} to {agustin_proportion_high_value:.2f}")

# Curva normal teórica
x = np.linspace(global_min_value, global_max_value, 100)
y = norm.pdf(x, average, standard_deviation)

# Visualización
plt.figure(figsize=(8,5))
plt.hist(data, bins=10, density=True, alpha=0.6, color='lightblue', edgecolor='black', label='Observed Data')
plt.plot(x, y, 'r-', linewidth=2, label='Normal Curve')
plt.title('Score Distribution (English Marks)')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
