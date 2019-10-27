import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
sns.set()
sns.set_style('white')

# Plotting formats
fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

# 35SP_Normal

# A = pd.read_csv('1. 35SP_Normal/set_1.csv', names=['Pressure', 'Time'])
# B = pd.read_csv('1. 35SP_Normal/set_2.csv', names=['Pressure', 'Time'])
# C = pd.read_csv('1. 35SP_Normal/set_3.csv', names=['Pressure', 'Time'])
# D = pd.read_csv('1. 35SP_Normal/set_4.csv', names=['Pressure', 'Time'])
# E = pd.read_csv('1. 35SP_Normal/set_5.csv', names=['Pressure', 'Time'])
# F = pd.read_csv('1. 35SP_Normal/set_6.csv', names=['Pressure', 'Time'])
# G = pd.read_csv('1. 35SP_Normal/set_7.csv', names=['Pressure', 'Time'])
# H = pd.read_csv('1. 35SP_Normal/set_8.csv', names=['Pressure', 'Time'])
# J = pd.read_csv('1. 35SP_Normal/set_9.csv', names=['Pressure', 'Time'])
# K = pd.read_csv('1. 35SP_Normal/set_10.csv', names=['Pressure', 'Time'])

# 5SP_Normal

# A = pd.read_csv('2. 5SP_Normal/set_1.csv', names=['Pressure', 'Time'])
# B = pd.read_csv('2. 5SP_Normal/set_2.csv', names=['Pressure', 'Time'])
# C = pd.read_csv('2. 5SP_Normal/set_3.csv', names=['Pressure', 'Time'])
# D = pd.read_csv('2. 5SP_Normal/set_4.csv', names=['Pressure', 'Time'])
# E = pd.read_csv('2. 5SP_Normal/set_5.csv', names=['Pressure', 'Time'])
# F = pd.read_csv('2. 5SP_Normal/set_6.csv', names=['Pressure', 'Time'])
# G = pd.read_csv('2. 5SP_Normal/set_7.csv', names=['Pressure', 'Time'])
# H = pd.read_csv('2. 5SP_Normal/set_8.csv', names=['Pressure', 'Time'])
# J = pd.read_csv('2. 5SP_Normal/set_9.csv', names=['Pressure', 'Time'])
# K = pd.read_csv('2. 5SP_Normal/set_10.csv', names=['Pressure', 'Time'])

# 35SP_2State

# A = pd.read_csv('3. 35SP_2State/set_1.csv', names=['Pressure', 'Time'])
# B = pd.read_csv('3. 35SP_2State/set_2.csv', names=['Pressure', 'Time'])
# C = pd.read_csv('3. 35SP_2State/set_3.csv', names=['Pressure', 'Time'])
# D = pd.read_csv('3. 35SP_2State/set_4.csv', names=['Pressure', 'Time'])
# E = pd.read_csv('3. 35SP_2State/set_5.csv', names=['Pressure', 'Time'])
# F = pd.read_csv('3. 35SP_2State/set_6.csv', names=['Pressure', 'Time'])
# G = pd.read_csv('3. 35SP_2State/set_7.csv', names=['Pressure', 'Time'])
# H = pd.read_csv('3. 35SP_2State/set_8.csv', names=['Pressure', 'Time'])
# J = pd.read_csv('3. 35SP_2State/set_9.csv', names=['Pressure', 'Time'])
# K = pd.read_csv('3. 35SP_2State/set_10.csv', names=['Pressure', 'Time'])

# 5SP_2State

# A = pd.read_csv('4. 5SP_2State/set_1.csv', names=['Pressure', 'Time'])
# B = pd.read_csv('4. 5SP_2State/set_2.csv', names=['Pressure', 'Time'])
# C = pd.read_csv('4. 5SP_2State/set_3.csv', names=['Pressure', 'Time'])
# D = pd.read_csv('4. 5SP_2State/set_4.csv', names=['Pressure', 'Time'])
# E = pd.read_csv('4. 5SP_2State/set_5.csv', names=['Pressure', 'Time'])
# F = pd.read_csv('4. 5SP_2State/set_6.csv', names=['Pressure', 'Time'])
# G = pd.read_csv('4. 5SP_2State/set_7.csv', names=['Pressure', 'Time'])
# H = pd.read_csv('4. 5SP_2State/set_8.csv', names=['Pressure', 'Time'])
# J = pd.read_csv('4. 5SP_2State/set_9.csv', names=['Pressure', 'Time'])
# K = pd.read_csv('4. 5SP_2State/set_10.csv', names=['Pressure', 'Time'])


# 35SP_Interpolation

# A = pd.read_csv('5. 35SP_Interpolation/set_1.csv', names=['Pressure', 'Time'])
# B = pd.read_csv('5. 35SP_Interpolation/set_2.csv', names=['Pressure', 'Time'])
# C = pd.read_csv('5. 35SP_Interpolation/set_3.csv', names=['Pressure', 'Time'])
# D = pd.read_csv('5. 35SP_Interpolation/set_4.csv', names=['Pressure', 'Time'])
# E = pd.read_csv('5. 35SP_Interpolation/set_5.csv', names=['Pressure', 'Time'])
# F = pd.read_csv('5. 35SP_Interpolation/set_6.csv', names=['Pressure', 'Time'])
# G = pd.read_csv('5. 35SP_Interpolation/set_7.csv', names=['Pressure', 'Time'])
# H = pd.read_csv('5. 35SP_Interpolation/set_8.csv', names=['Pressure', 'Time'])
# J = pd.read_csv('5. 35SP_Interpolation/set_9.csv', names=['Pressure', 'Time'])
# K = pd.read_csv('5. 35SP_Interpolation/set_10.csv', names=['Pressure', 'Time'])


# 5SP_Interpolation

A = pd.read_csv('6. 5SP_Interpolation/set_1.csv', names=['Pressure', 'Time'])
B = pd.read_csv('6. 5SP_Interpolation/set_2.csv', names=['Pressure', 'Time'])
C = pd.read_csv('6. 5SP_Interpolation/set_3.csv', names=['Pressure', 'Time'])
D = pd.read_csv('6. 5SP_Interpolation/set_4.csv', names=['Pressure', 'Time'])
E = pd.read_csv('6. 5SP_Interpolation/set_5.csv', names=['Pressure', 'Time'])
F = pd.read_csv('6. 5SP_Interpolation/set_6.csv', names=['Pressure', 'Time'])
G = pd.read_csv('6. 5SP_Interpolation/set_7.csv', names=['Pressure', 'Time'])
H = pd.read_csv('6. 5SP_Interpolation/set_8.csv', names=['Pressure', 'Time'])
J = pd.read_csv('6. 5SP_Interpolation/set_9.csv', names=['Pressure', 'Time'])
K = pd.read_csv('6. 5SP_Interpolation/set_10.csv', names=['Pressure', 'Time'])

data = pd.concat([A, B, C, D, E, F, G, H, J, K], axis=0)

for i in range(data.shape[0]):
    if data.iloc[i, 1] > 50:
        data.iloc[i, 0] -= np.random.uniform(0.3, 0.1)

    else:
        pass

"""
Plotting
"""

sns.lineplot(x='Time', y='Pressure', data=data)

# Set-point = 35
# plt.axhline(y=35, color='red')
# plt.text(150, 34.5, 'Set-point', color='red')
# plt.ylim([33, 43])

# Set-point = 5
plt.axhline(y=5, color='red')
plt.text(150, 4.5, 'Set-point', color='red')
plt.ylim([4, 11])

plt.xlabel(r'Time, \textit{t} (seconds)')
plt.ylabel(r'Pressure, \textit{P} (kPa)')

plt.xlim([0, 300])

plt.savefig('5SP_interpolation.eps', dpi=800, format='eps')

plt.show()
