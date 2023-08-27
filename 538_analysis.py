import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/pollster-ratings.csv"

try:
    df = pd.read_csv(url)
except:
    print("Could not retrieve the data!")

for c in df.columns:
    print(c, end=', ')

df.info()

df.rename(columns={'Predictive    Plus-Minus':'Predictive Plus-Minus'},inplace=True)

df['Races Called Correctly'][:3]

def percent_to_float(x):
    """
    Converts percentage to float
    """

    return float(x[:-1]/100)

def bias_party_id(x):
    """
    Returns a string indicating partisan bias
    """
    if x is np.nan:
        return "No Data"
    if x > 0 :
        return "Democrat"
    else:
        return "Republican"

def bias_party_degree(x):
    """
    Returns a string indicating partisan bias
    """
    if x is np.nan:
        return np.nan
    x = str(x)
    return float(x)

df['Partisan Bias'] = df['Bias'].apply(bias_party_id)
df['Partisan Bias Degree'] = df['Bias'].apply(bias_party_degree)
