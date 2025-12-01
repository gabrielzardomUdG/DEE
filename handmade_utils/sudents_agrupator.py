import pandas as pd

from handmade_utils.data_holder import DataHolder
from handmade_utils.global_constants import DOC_TYPE_LOCATION, DOC_TYPE_SUBJECT


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

def get_students_of_location(all_students: pd.DataFrame, location: str) -> pd.DataFrame:
    result: pd.DataFrame = all_students[all_students["location"] == location]
    return result

def get_unique_locations(all_students : pd.DataFrame) -> list :
    return all_students["location"].unique()

def get_dh_list_of_students_by_location(all_students : pd.DataFrame) -> list :

    str_locations : list = get_unique_locations(all_students)
    dh_students_by_location : list = []

    for location in str_locations:
        new_element : DataHolder = DataHolder(DOC_TYPE_LOCATION, location, get_students_of_location(all_students, location))
        dh_students_by_location.append(new_element)

    return dh_students_by_location

def get_dh_list_of_subject_by_location(all_students : pd.DataFrame) -> list :

    str_subjects : list = list(all_students.columns[3:])
    dh_students_by_location : list = []

    for subject in str_subjects:
        new_element : DataHolder = DataHolder(DOC_TYPE_SUBJECT, subject, get_subject_marks_by_location(all_students, subject))
        dh_students_by_location.append(new_element)

    return dh_students_by_location