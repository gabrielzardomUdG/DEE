from typing import Hashable
import pandas as pd


def get_subject_marks_by_location(all_students : pd.DataFrame, subject_column : str) -> pd.DataFrame:
    result : pd.DataFrame = pd.DataFrame()   # final output
    
    for location, df in all_students.groupby("location"):
        col = df[[subject_column]].reset_index(drop=True)
        col.columns = [location]
        result = pd.concat([result, col], axis=1)
    return result