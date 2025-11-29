from typing import Hashable
import pandas as pd


def get_subject_marks_by_location(all_students : pd.DataFrame, subject_column : str) -> pd.DataFrame:
    result : pd.DataFrame = pd.DataFrame()
    
    for location, df in all_students.groupby("location"):
        col = df[[subject_column]].reset_index(drop=True)
        col.columns = [location]
        result = pd.concat([result, col], axis=1)
    return result


def create_subject_x_location_average_table(all_students: pd.DataFrame) -> pd.DataFrame:
    result = all_students.groupby("location").mean(numeric_only=True).iloc[:, 3:]
    return result