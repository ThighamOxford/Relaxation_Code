import numpy as np
import pandas as pd

df_Nest = pd.read_csv("data_Nest.csv")
df_Gd = pd.read_csv("data_GD.csv")
df_Nest.plot(x="t", y="Energy", logy=False)
df_Gd.plot(x = "t", y = "Energy", logy=False)
plt.grid()
plt.show()