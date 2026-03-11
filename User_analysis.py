import os
import numpy as np
import pandas as pd
import lib
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fs_emg = 2148.1481
fs_imu = 370.3704

directory_general = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\Projects\Inclined gait\Data\Peaks Data\P1'
ID = os.path.basename(directory_general)
print(ID)

trials = [0, 12, 25]
cols = ['ECG', 'Gstr', 'Quad', 'IMU']

df_number = pd.DataFrame(index=trials, columns=cols)
df_average = pd.DataFrame(index=trials, columns=cols)
df_SD = pd.DataFrame(index=trials, columns=cols)
df_CV = pd.DataFrame(index=trials, columns=cols)
df_DFA = pd.DataFrame(index=trials, columns=cols)
df_SaEn = pd.DataFrame(index=trials, columns=cols)

# Create Plotly figure with 4 subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('ECG', 'Gstr', 'Quad', 'IMU')
)

# Map each signal to subplot position
subplot_positions = {
    'ECG': (1, 1),
    'Gstr': (1, 2),
    'Quad': (2, 1),
    'IMU': (2, 2)
}

for trial in trials:
    directory = os.path.join(directory_general, str(trial))
    os.chdir(directory)

    Peaks_ECG = pd.read_excel(rf'Peaks_ECG_{trial}.xlsx')
    Peaks_Gastr = pd.read_excel(rf'Peaks_Gastr_{trial}.xlsx')
    Peaks_Quad = pd.read_excel(rf'Peaks_Quad_{trial}.xlsx')
    Peaks_IMU = pd.read_excel(rf'Peaks_IMU_{trial}.xlsx')

    Peaks_ECG_time = Peaks_ECG['peak_times_ECG'].to_numpy()
    Peaks_Gastr_time = Peaks_Gastr['peak_times_Gastr'].to_numpy()
    Peaks_Quad_time = Peaks_Quad['peak_times_Quad'].to_numpy()
    Peaks_IMU_time = Peaks_IMU['peak_times_IMU'].to_numpy()

    intervals_ECG = np.diff(Peaks_ECG_time)
    intervals_Gastr = np.diff(Peaks_Gastr_time)
    intervals_Quad = np.diff(Peaks_Quad_time)
    intervals_IMU = np.diff(Peaks_IMU_time)

    signals = {
        'ECG': intervals_ECG,
        'Gstr': intervals_Gastr,
        'Quad': intervals_Quad,
        'IMU': intervals_IMU
    }

    for name, data in signals.items():
        row, col = subplot_positions[name]

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode='lines',
                name=f'{name} - Trial {trial}',  # <- key change
            ),
            row=row, col=col
        )

        df_number.loc[trial, name] = len(data)
        df_average.loc[trial, name] = np.mean(data)
        df_SD.loc[trial, name] = np.std(data)
        df_CV.loc[trial, name] = np.std(data) / np.mean(data)

        scales = np.arange(16, len(data)//9, 1)
        _, _, dfa = lib.dfa(data, scales, plot=False)
        df_DFA.loc[trial, name] = dfa

        df_SaEn.loc[trial, name] = lib.Ent_Samp(data, 2, 0.2)

fig.update_layout(
    title=f'{ID} - Peak Intervals',
    height=800,
    width=1200
)

fig.update_xaxes(title_text='Interval number')
fig.update_yaxes(title_text='Interval duration')

fig.show()

print('number')
print(df_number)

print('')
print('average')
print(df_average)

print('')
print('SD')
print(df_SD)

print('')
print('CV')
print(df_CV)

print('')
print('DFA')
print(df_DFA)

print('')
print('SaEn')
print(df_SaEn)