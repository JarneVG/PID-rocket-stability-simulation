# Use this code to test the TVC subsystem code and see if it matches your experimental data

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
fs = 1000.0  # Hz
dt = 1 / fs
T_total = 2  # seconds
t = np.arange(0, T_total, dt)
n = len(t)

# --- Input Signal (selectable) ---
input_type = 'step'  # 'sine' or 'step'
frequency = 0.5  # Hz
desired_TVC_angle_sine = np.radians(10 * np.sin(2 * np.pi * frequency * t))  # radians

step_start_time = 0.0  # seconds
step_start_index = int(step_start_time * fs)
step_amplitude_deg = 10  # degrees
desired_TVC_angle_step = np.zeros_like(t)
desired_TVC_angle_step[step_start_index:] = np.radians(step_amplitude_deg)

# Choose the input type here:
if input_type == 'sine':
    desired_TVC_angle = desired_TVC_angle_sine  # or `desired_TVC_angle_step`
elif input_type == 'step':
    desired_TVC_angle = desired_TVC_angle_step


# --- Delay Parameters ---
TVC_delay_s = 0.014  # seconds
TVC_delay_samples = int(TVC_delay_s * fs)  # convert to samples

# --- Second Order Discrete Transfer Function (example) ---
from scipy.signal import cont2discrete

# Continuous system
num_ct = [4102]
den_ct = [1.0, 92.91, 4167]

# Discretize system using Tustin (bilinear) method
system_d = cont2discrete((num_ct, den_ct), dt, method='tustin')
num_d, den_d, _ = system_d
num_d = num_d[0]  # remove extra brackets

print("Discretized numerator:", num_d)
print("Discretized denominator:", den_d)


# --- Backlash Parameters ---
TVC_backlash = np.radians(1.5)  # radians

def apply_backlash(input_signal, previous_real_output, backlash):
    if input_signal > previous_real_output + backlash:
        previous_real_output = input_signal - backlash
    elif input_signal < previous_real_output - backlash:
        previous_real_output = input_signal + backlash
    return previous_real_output   

# --- Preallocate Arrays ---
delayed_TVC = np.zeros(n)
filtered_TVC = np.zeros(n)
real_TVC = np.zeros(n)

max_order = max(len(num_d), len(den_d))
u_hist = np.zeros(max_order)
y_hist = np.zeros(max_order)

previous_real_TVC_angle = 0.0

# --- Simulation Loop ---
for i in range(n):
    # Delay
    if i >= TVC_delay_samples:
        delayed_angle = desired_TVC_angle[i - TVC_delay_samples]
    else:
        delayed_angle = 0.0
    delayed_TVC[i] = delayed_angle

    # Second-order filter
    u_hist = np.roll(u_hist, 1)
    y_hist = np.roll(y_hist, 1)
    u_hist[0] = delayed_angle

    # Compute output
    y = (np.dot(num_d, u_hist[:len(num_d)]) - np.dot(den_d[1:], y_hist[1:len(den_d)])) / den_d[0]
    y_hist[0] = y

    filtered_TVC[i] = y

    # Backlash
    real_TVC_angle = apply_backlash(y, previous_real_TVC_angle, TVC_backlash)
    real_TVC[i] = real_TVC_angle
    previous_real_TVC_angle = real_TVC_angle

# --- Plotting ---
# --- Combined Plot ---
plt.figure(figsize=(12, 6))
if input_type == 'step':
    #limit x axis to 0.15s
    plt.xlim(0, 0.15)

plt.plot(t, np.degrees(desired_TVC_angle), label='Desired TVC', linestyle='--')
plt.plot(t, np.degrees(delayed_TVC), label='Delayed TVC', linestyle=':')
plt.plot(t, np.degrees(filtered_TVC), label='Filtered (2nd Order)', linestyle='-')
plt.plot(t, np.degrees(real_TVC), label='With Backlash', linestyle='-.')
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("TVC Angle Evolution Through System")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
