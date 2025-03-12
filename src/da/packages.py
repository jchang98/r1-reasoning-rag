import sys
import re
import json

import scipy.stats as stats
import statsmodels

import talib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from streamlit.components.v1 import html
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pyecharts
