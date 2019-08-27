import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fonts = {"family": "serif",
         "weight": "normal",
         "size": "12"}

plt.rc('font', **fonts)
plt.rc('text', usetex=True)

ftc_noiseless = np.loadtxt('noiseless_case1.csv')
ftc_noise = np.loadtxt('noise_case1.csv')
no_ftc_noiseless = np.loadtxt('no_ftc_noiseless_case1.csv')
no_ftc_noise = np.loadtxt('no_ftc_noise_case1.csv')

x = np.linspace(0, ftc_noiseless.shape[0], ftc_noiseless.shape[0] - 50)

plt.plot(x, no_ftc_noiseless[50:, 0], label=r'No FTC (Noiseless)', linestyle='-.', color='black')
plt.plot(x, ftc_noise[50:, 0], label=r'With FTC (Sensor \& Actuator Noise)', linestyle='--',
         color='grey')
plt.plot(x, ftc_noiseless[50:, 0], label=r'With FTC (Noiseless)', color='black')
# plt.plot(no_ftc_noise[50:, 0], label=r'No FTC (Sensor Noise)', color='grey')

plt.xlabel(r'Time, \textit{t} (min)')
plt.ylabel(r'\%MeOH, $\textit{X}_D$ (wt. \%)')

plt.ylim([50, 105])

plt.legend(loc=0, prop={'size': 12}, frameon=False)

plt.savefig('Case1_Plot.eps', format='eps', dpi=1000)

plt.show()
