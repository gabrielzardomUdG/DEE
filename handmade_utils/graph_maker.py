import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from handmade_utils.data_holder import DataHolder
from handmade_utils.global_constants import DOC_TYPE_LOCATION

def graph_dataframe_percentage(data_frame : pd.DataFrame, title : str, y_lower_limit : int, y_upper_limit : int) -> None :

    data_frame.plot(kind='bar', figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Score Range")
    plt.ylabel("Percentage")
    plt.ylim(y_lower_limit, y_upper_limit)
    plt.xticks(rotation=0)
    plt.show()

    return

def create_unified_intervals(pure_numeric_df : pd.DataFrame, lower_limit : int, upper_limit : int, partition_number : int) -> pd.DataFrame :
    bins : np.ndarray = np.linspace(lower_limit, upper_limit, partition_number + 1)
    labels : list[str] = [f"{round(bins[i])}–{round(bins[i + 1])}" for i in range(partition_number)]
    result = (pure_numeric_df.apply(lambda col: pd.cut(col, bins=bins, labels=labels).value_counts(normalize=True)).sort_index() * 100).round(2)
    return result


def graph_intervals(data_object : DataHolder, lower_limit : int, upper_limit : int, partition_number : int,
                    y_lower_limit : int, y_upper_limit : int) -> None :

    title : str = "Intervals By Location of Subject : " + data_object.str_value
    pure_numeric_df : pd.DataFrame = data_object.data_holder

    if data_object.doc_type == DOC_TYPE_LOCATION:
        title = "Intervals By Subject of Students from " + data_object.str_value
        columns_to_drop : list = data_object.data_holder.columns[0:3]
        pure_numeric_df = data_object.data_holder.drop(columns=columns_to_drop)

    df_to_graph : pd.DataFrame = create_unified_intervals(pure_numeric_df, lower_limit, upper_limit, partition_number)

    graph_dataframe_percentage(df_to_graph, title, y_lower_limit, y_upper_limit)

    return

def graph_dh_list(dh_list : list[DataHolder], lower_limit : int, upper_limit : int, partition_number : int,
                  y_lower_limit : int, y_upper_limit : int) -> None :

    for element in dh_list:
        graph_intervals(element, lower_limit, upper_limit, partition_number, y_lower_limit, y_upper_limit)

    return

