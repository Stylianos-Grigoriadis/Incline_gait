import math
import scipy.stats
from scipy import signal
import matplotlib.pyplot as plt
import statistics
import numpy as np
from numpy.fft import fft, fftfreq
import colorednoise as cn
from scipy.stats import pearsonr, spearmanr

def FFT(var,fs):
    dt = 1 / fs
    freqs = fftfreq(len(var), dt)
    mask = freqs > 0
    Y = fft(var)
    pSpec = 2 * ((abs(Y) / len(var)) ** 2)

    f = freqs[mask]
    a = pSpec[mask]
    plt.plot(f,a)
    plt.show()
    sumA=[0]
    for i in range(1,len(a)):
        sumA.append(a[i]+sumA[i-1])

    prec90= []
    prec95 = []
    prec99 = []
    percA = [(sumA[i] / sumA[-1])*100 for i in range(len(sumA))]
    for p in percA:
        prec90.append(abs(p - 90))
        prec95.append(abs(p - 95))
        prec99.append(abs(p - 99))
    for i in range(len(percA)):
        if prec90[i]==min(prec90):
            index90 = i
        if prec95[i]==min(prec95):
            index95 = i
        if prec99[i]==min(prec99):
            index99 = i
    return f[index90],f[index95],f[index99]

def q_to_ypr(q):
    if q:
        yaw = (math.atan2(2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[0] ** 2 + 2 * q[1] ** 2 - 1))
        roll = (-1 * math.asin(2 * q[1] * q[3] + 2 * q[0] * q[2]))
        pitch = (math.atan2(2 * q[2] * q[3] - 2 * q[0] * q[1], 2 * q[0] ** 2 + 2 * q[3] ** 2 - 1))
        return [yaw, pitch, roll]

def pyth2d(x1,y1,x2,y2):
    x=x2-x1
    y=y2-y1
    c=math.sqrt(x**2+y**2)
    return c

def compute_cop(X, Y, df_filtered):
    list_X_coordinates_left_plate = []
    list_Y_coordinates_left_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1L'][i] + df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i] + \
                df_filtered['CHANNEL_4L'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2L'][i] + df_filtered['CHANNEL_3L'][i])) / F_all
        list_X_coordinates_left_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3L'][i] + df_filtered['CHANNEL_4L'][i])) / F_all
        list_Y_coordinates_left_plate.append(y_coordinate)

    list_X_coordinates_right_plate = []
    list_Y_coordinates_right_plate = []
    for i in range(len(df_filtered['CHANNEL_1L'])):
        F_all = df_filtered['CHANNEL_1R'][i] + df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i] + \
                df_filtered['CHANNEL_4R'][i]
        x_coordinate = (X * (df_filtered['CHANNEL_2R'][i] + df_filtered['CHANNEL_3R'][i])) / F_all
        list_X_coordinates_right_plate.append(x_coordinate)
        y_coordinate = (Y * (df_filtered['CHANNEL_3R'][i] + df_filtered['CHANNEL_4R'][i])) / F_all
        list_Y_coordinates_right_plate.append(y_coordinate)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform = []
    for i in range(len(list_X_coordinates_left_plate)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_left_plate[i] - X / 2)
        list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_left_plate[i] - Y / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_X_coordinates_right_plate[i] - X / 2)
        list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform.append(
            list_Y_coordinates_right_plate[i] - Y / 2)

    list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms = []
    list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms = []
    for i in range(len(list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform)):
        list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[i] - X / 2)
        list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms.append(
            list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[i] + X / 2)

    list_X_coordinates_both_plates = []
    list_Y_coordinates_both_plates = []
    for i in range(len(list_X_coordinates_right_plate)):
        list_X_coordinates_both_plates.append((list_X_coordinates_left_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i] +
                                               list_X_coordinates_right_plate_with_zero_at_the_middle_of_both_platforms[
                                                   i]) / 2)
        list_Y_coordinates_both_plates.append((list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform[
                                                   i] +
                                               list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform[
                                                   i]) / 2)
    return list_X_coordinates_both_plates,list_Y_coordinates_both_plates
    #return list_X_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_left_plate_with_zero_at_the_middle_of_the_platform, list_X_coordinates_right_plate_with_zero_at_the_middle_of_the_platform, list_Y_coordinates_right_plate_with_zero_at_the_middle_of_the_platform

def peaks(var,distance,height):
    peaks, _ = signal.find_peaks(var, distance=distance, height=height)
    peaksAmp = [var[peak] for peak in peaks]

    return peaks,peaksAmp

def Linear_Interpolation(col, step, plus):
    n = len(col)
    newdf = []
    value = step
    while value < n - 1:
        if math.ceil(value) == math.floor(value):
            num = col[math.ceil(value) + plus]
        else:

            num = ((col[math.ceil(value) + plus] - col[math.floor(value) + plus]) * (value - math.floor(value))) / (
                    math.ceil(value) - math.floor(value)) + col[math.floor(value) + plus]

        newdf.append(num)
        value = value + step

    return newdf

def Butterworth(fs, fc, var):
    """ Parameter:
            fs:     sampling frequency
            fc:     cutoff frequency for example 30Hz
            var:    data series
    """

    b, a = signal.butter(N=2, Wn=fc, btype='low', fs=fs)
    return signal.filtfilt(b, a, var)

def Average(lst):
    return sum(lst) / len(lst)

def Remove_drift(inte,index,cut):
    inte = inte[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]
    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_sl[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Remove_drift2(inte,inte2nd,index,cut):
    inte = inte[cut:]
    inte2nd = inte2nd[cut:]
    index = index[cut:]
    slope, intercept, r, p, stderr = scipy.stats.linregress(index, inte)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl = [i * slope for i in index]

    slope2, intercept, r, p, stderr = scipy.stats.linregress(index, inte2nd)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    t_sl2 = [i * slope2 for i in index]
    t_slope=[a-b for a,b in zip(t_sl,t_sl2)]

    inte2 = []
    for i in range(len(inte)):
        inte2.append(inte[i] - (intercept + t_slope[i]))
    inte3 = []
    for i in range(len(inte2)):
        inte3.append(inte2[i] - inte2[0])
    return inte3

def Bland_Altman_plot(Var1,Var2,title):
    Difference = [v - m for v, m in zip(Var1, Var2)]
    Mean = [(v + m) / 2 for v, m in zip(Var1, Var2)]
    Bias = Average(Difference)
    StanDev = statistics.stdev(Difference)
    LowerLOA = Bias - 1.96 * StanDev
    UpperLOA = Bias + 1.96 * StanDev
    inlims=0
    uplim=0
    downlim=0
    for d in Difference:
        if d>=UpperLOA:
            uplim+=1
        elif d<=LowerLOA:
            downlim+=1
        else:
            inlims+=1

    print('Total points: ', len(Difference),
          '\nInside points number: ', inlims,
          '\nUp points number: ', uplim,
          '\nDown points number: ', downlim,
          '\nUp Perc: ', (uplim/len(Difference))*100,
          '\nDown Perc: ', (downlim/len(Difference))*100,
          '\nOut Perc: ', ((downlim + uplim) / len(Difference)) * 100)

    plt.show()
    plt.title('Bland Altman Plot {name}'.format(name=title), fontsize=16)
    plt.xlabel('Average', fontsize=16)
    plt.ylabel('Difference', fontsize=16)
    plt.scatter(Mean, Difference, color='grey', linewidths=1.5)
    plt.axhline(y=Bias, color='black')
    plt.axhline(y=LowerLOA, color='black', ls=':')
    plt.axhline(y=UpperLOA, color='black', ls=':')
    plt.show()

    #OutPerc = ((downlim + uplim) / len(Difference)) * 100
    res_list = [len(Difference), inlims, uplim, downlim, (uplim / len(Difference)) * 100,
                (downlim / len(Difference)) * 100, ((downlim + uplim) / len(Difference)) * 100]
    return res_list

def intergral(span,dt):
    rects = []
    for i in range(len(span) - 1):
        rects.append(((span[i] + span[i + 1]) * dt) / 2)
    integral = [rects[0]]
    for i in range(len(rects) - 1):
        integral.append(integral[i] + rects[i + 1])

    return integral

def derivative(array,fs):
    dt = 1/fs
    der = []

    array = list(array)

    for i in range(len(array)-1):
        der.append((array[i+1]-array[i])/dt)
    return der

def Pink_noise_generator():
    pass

def residual_analysis(signal, sampling_freq, list_cutoff_freq):
    residuals_list = []
    signal = np.array(signal)
    for i in range(len(list_cutoff_freq)):
        filtered_signal = Butterworth(sampling_freq,list_cutoff_freq[i],signal)
        filtered_signal = np.array(filtered_signal)
        sum_squares = np.sum((signal - filtered_signal)**2)
        residual = np.sqrt(sum_squares/len(signal))
        residuals_list.append(residual)

    plt.plot(list_cutoff_freq, residuals_list)
    plt.show()

def correlation_analysis(x, y, method='pearson', plot=False, xlabel='X', ylabel='Y', title=None):
    """
    Perform correlation analysis between two data series.

    Parameters
    ----------
    x, y : array-like or pd.Series
        Input data
    method : str
        'pearson' or 'spearman'
    plot : bool
        If True, plot scatter with regression line
    xlabel, ylabel : str
        Axis labels for plot
    title : str
        Plot title

    Returns
    -------
    results : dict
        Dictionary with correlation coefficient, p-value, and N
    """

    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaNs pairwise
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Not enough data points for correlation.")

    # Compute correlation
    if method.lower() == 'pearson':
        r, p = pearsonr(x, y)
        corr_name = 'Pearson r'
    elif method.lower() == 'spearman':
        r, p = spearmanr(x, y)
        corr_name = 'Spearman ρ'
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    # Optional plot
    if plot:
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, color='k', alpha=0.7)

        # Regression line (for visualization)
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r', lw=2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is None:
            plt.title(f'{corr_name} = {r:.2f}, p = {p:.3g}')
        else:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    # Return results
    results = {
        'method': corr_name,
        'r': r,
        'p_value': p,
        'n': len(x)
    }

    return results

def Kinvent_convert_accel(signal):
    signal = np.asarray(signal)
    acceleration_range = np.max(signal) - np.min(signal)
    signal_converted = (signal - 32768) * (acceleration_range / 32768)

    return signal_converted

def Kinvent_convert_emg(signal):
    signal = np.asarray(signal)
    signal = (signal - 2 ** 23) / 10
    signal = signal/1000

    return signal

def prepare_emg_df(df):
    df = df.iloc[:, :-1]
    df.columns = ['Time', 'EMG']
    df['EMG'] = Kinvent_convert_emg(df['EMG'])

    return df

def prepare_ecg_df(df):
    df = df.iloc[:, :-1]
    df.columns = ['Time', 'ECG']
    df['ECG'] = Kinvent_convert_emg(df['ECG'])

    return df

def prepare_imu_df(df):
    df = df.iloc[:, [0, 5, 6, 7]]
    df.columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z']
    df['Acc_X'] = Kinvent_convert_accel(df['Acc_X'])
    df['Acc_Y'] = Kinvent_convert_accel(df['Acc_Y'])
    df['Acc_Z'] = Kinvent_convert_accel(df['Acc_Z'])

    return df