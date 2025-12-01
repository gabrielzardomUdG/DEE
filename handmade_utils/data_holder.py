from dataclasses import dataclass

import pandas as pd


@dataclass
class DataHolder:
    doc_type    : str
    str_value   : str
    data_holder : pd.DataFrame