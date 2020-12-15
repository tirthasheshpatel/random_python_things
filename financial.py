import numpy as np
import pandas as pd

data = pd.read_excel("financial_data.xlsx", header=[0,1,2])
data.stack(level=[0,1,2]).reset_index()
