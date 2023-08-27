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
df[['Pollster', 'Partisan Bias', 'Partisan Bias Degree']].sample(5)

df['538 Grade'].unique()

plt.figure(figsize=(12,4))
plt.title("Pollster grade counts",fontsize=18)
plt.bar(x=df['538 Grade'].unique(),
        height=df['538 Grade'].value_counts(),
       color='red',alpha=0.6,edgecolor='k',linewidth=2.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.show()

def grade_numeric(x):
    """
    Quantizes the letter grade
    """
    if x[0] == 'A':
        return 4
    if x[0] == 'B':
        return 3
    if x[0] == 'C':
        return 2
    if x[0] == 'D':
        return 1
    else:
        return 0

df['Numeric grade'] = df['538 Grade'].apply(grade_numeric)
df['Numeric grade'].value_counts()

def custom_boxplot(x,y,rot=90):
    plt.figure(figsize=(12,4))
    plt.title("Boxplot of \"{}\" by \"{}\"".format(y,x),fontsize=17)
    sns.boxplot(x=x,y=y,data=df)
    plt.xticks(rotation=rot,fontsize=12)
    plt.yticks(fontsize=13)
    plt.xlabel(x,fontsize=15)
    plt.ylabel(y+'\n',fontsize=15)
    plt.show()

custom_boxplot(x='Partisan Bias',y='Races Called Correctly',rot=0)

custom_boxplot(x='Partisan Bias',y='Advanced Plus-Minus')

custom_boxplot(x='AAPOR/Roper',y='Races Called Correctly',rot=0)

custom_boxplot(x='AAPOR/Roper',y='Advanced Plus-Minus',rot=0)

def custom_scatter(x,y,data=df,pos=(0,0),regeqn=True):
    """
    Plots customized scatter plots with regression fit using Seaborn
    """
    sns.lmplot(x=x,y=y,data=data,height=4,aspect=1.5,
       scatter_kws={'color':'yellow','edgecolor':'k','s':100},
              line_kws={'linewidth':3,'color':'red','linestyle':'--'})

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(x,fontsize=15)
    plt.ylabel(y+'\n',fontsize=15)
    ax = plt.gca()
    ax.set_title("Regression fit of \"{}\" vs. \"{}\"".format(x,y),fontsize=15)

    if (regeqn):
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x],df[y])
        r_squared = r_value**2
        eqn= "$y$={0:.3f}$x$+{1:.3f},\n$R^2$:{2:.3f}".format(slope,intercept,r_squared)
        plt.annotate(s=eqn,xy=pos,fontsize=13)

custom_scatter(x='Races Called Correctly',
               y='Predictive Plus-Minus',
              pos=(0.05,-1.5))

custom_scatter(x='Numeric grade',
                y='Simple Average Error',
                pos=(0,20))

df.columns

df_2 = df.dropna()
filtered = df_2[df_2['Polls Analyzed']>100]
custom_scatter(x = 'Polls Analyzed',
                y='Partisan Bias Degree',
                data=filtered, regeqn=False)

x = df_2['Polls Analyzed']
y = df_2['Partisan Bias Degree']

plt.scatter(x,y, color = 'yellow', edgecolor='k', s=100)

def func(x, a, b, c):
    return a * np.exp(-b *0.1*x) + c
popt, pcov = curve_fit(func, x, y)
y_fit = func(x,popt[0],popt[1],popt[2])
plt.scatter(x,y_fit,color='red',alpha=0.5)
plt.show()
