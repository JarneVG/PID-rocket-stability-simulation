## Run with command: 'streamlit run PID_tuner.py'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import streamlit.components.v1 as components
import scipy.signal as signal
import control as ctrl
from collections import deque





st.title("PID tuner (rad2radss)")
col1, col2 = st.columns(2)
    
    
# === SIMULATION PARAMETERS ===
t_max = st.number_input("Total Simulation Time (s)", value=8.0)
dt = 0.001
n_steps = int(t_max / dt)


# === INPUT PARAMETERS ===
st.subheader("Select Rocket Parameters")
mass = st.number_input("Starting Rocket Mass (g)", value=595)
mass = mass/1000
start_MMOI = st.number_input("Starting Rocket MMOI (kgm²)", value=0.0165, format="%.4f" )
end_MMOI = st.number_input("Ending Rocket MMOI (kgm²)", value=0.0165, format="%.4f" )
start_lever = st.number_input("Starting TVC lever arm (m)", value=0.16, format="%.4f" )
end_lever = st.number_input("Ending TVC lever arm (m)", value=0.15, format="%.4f" )
max_tvc_deg = st.number_input("Maximum TVC Angle (degrees)", value=7.0, step=.1)



# === Thrust Curve Definitions ===
KlimaC2 = np.array([
    [0.0, 0.0],
    [0.04, 0.229],
    [0.12, 0.658],
    [0.211, 1.144],
    [0.291, 1.831],
    [0.385, 2.86],
    [0.447, 3.833],
    [0.505, 5.001],
    [0.567, 3.89],
    [0.615, 3.146],
    [0.665, 2.66],
    [0.735, 2.203],
    [0.815, 2.088],
    [0.93, 1.98],
    [4.589, 1.96],
    [4.729, 1.888],
    [4.815, 1.602],
    [4.873, 1.259],
    [4.969, 0.658],
    [5.083, 0.0]
])

KlimaC2_mass = np.array([
    [0.0, 11.3],
    [0.04, 11.2948],
    [0.12, 11.2544],
    [0.211, 11.1611],
    [0.291, 11.0257],
    [0.385, 10.7748],
    [0.447, 10.5387],
    [0.505, 10.2472],
    [0.567, 9.93361],
    [0.615, 9.74146],
    [0.665, 9.5763],
    [0.735, 9.38263],
    [0.815, 9.18732],
    [0.93, 8.92116],
    [4.589, 0.719051],
    [4.729, 0.412551],
    [4.815, 0.24179],
    [4.873, 0.147381],
    [4.969, 0.0426774],
    [5.083, 0.0]
])

KlimaC6 = np.array([    
    [0, 0],
    [0.046, 0.953],
    [0.168, 5.259],
    [0.235, 10.023],
    [0.291, 15.00],
    [0.418, 9.87],
    [0.505, 7.546],
    [0.582, 6.631],
    [0.679, 6.136],
    [0.786, 5.716],
    [1.26, 5.678],
    [1.357, 5.488],
    [1.423, 4.992],
    [1.469, 4.116],
    [1.618, 1.22],
    [1.701, 0.0],
])

KlimaC6_mass = np.array([
    [0.000, 9.60000],
    [0.046, 9.57895],
    [0.168, 9.21498],
    [0.235, 8.72326],
    [0.291, 8.05029],
    [0.418, 6.53342],
    [0.505, 5.80575],
    [0.582, 5.28150],
    [0.679, 4.68676],
    [0.786, 4.07772],
    [1.260, 1.48401],
    [1.357, 0.96385],
    [1.423, 0.63167],
    [1.469, 0.43046],
    [1.618, 0.04863],
    [1.701, 0.00000],
])


KlimaD3 = np.array([  
    [0, 0],
    [0.073, 0.229],
    [0.178, 0.686],
    [0.251, 1.287],
    [0.313, 2.203],
    [0.375, 3.633],
    [0.425, 5.006],
    [0.473, 6.465],
    [0.556, 8.181],
    [0.603, 9.01],
    [0.655, 6.922],
    [0.698, 5.463],
    [0.782, 4.291],
    [0.873, 3.576],
    [1.024, 3.146],
    [1.176, 2.946],
    [5.282, 2.918],
    [5.491, 2.832],
    [5.59, 2.517],
    [5.782, 1.859],
    [5.924, 1.287],
    [6.061, 0.715],
    [6.17, 0.286],
    [6.26, 0.0],
])

KlimaD3_mass = np.array([
    [0.000, 17.000],
    [0.073, 16.9921],
    [0.178, 16.947],
    [0.251, 16.8793],
    [0.313, 16.7777],
    [0.375, 16.6077],
    [0.425, 16.4047],
    [0.473, 16.146],
    [0.556, 15.5749],
    [0.603, 15.1953],
    [0.655, 14.8061],
    [0.698, 14.5559],
    [0.782, 14.1709],
    [0.873, 13.8346],
    [1.024, 13.3577],
    [1.176, 12.9226],
    [5.282, 1.61027],
    [5.491, 1.04565],
    [5.590, 0.796852],
    [5.782, 0.402106],
    [5.924, 0.192218],
    [6.061, 0.063356],
    [6.170, 0.0120934],
    [6.260, 0.000],
])


KlimaD9 = np.array([
    [0.000, 0.000],
    [0.040, 2.111],
    [0.116, 9.685],
    [0.213, 25.000],
    [0.286, 15.738],
    [0.329, 12.472],
    [0.369, 10.670],
    [0.420, 9.713],
    [0.495, 9.178],
    [0.597, 8.896],
    [1.711, 8.925],
    [1.826, 8.699],
    [1.917, 8.052],
    [1.975, 6.954],
    [2.206, 1.070],
    [2.242, 0.000],
])


KlimaD9_mass = np.array([
    [0.000, 16.1000],
    [0.040, 16.0659],
    [0.116, 15.7044],
    [0.213, 14.3477],
    [0.286, 13.1484],
    [0.329, 12.6592],
    [0.369, 12.2859],
    [0.420, 11.8667],
    [0.495, 11.2954],
    [0.597, 10.5519],
    [1.711, 2.54603],
    [1.826, 1.7287],
    [1.917, 1.11399],
    [1.975, 0.763006],
    [2.206, 0.0155338],
    [2.242, 0.0000],
])


EstesF15 = np.array([
    [0.0, 0.0],
    [0.148, 7.638],
    [0.228, 12.253],
    [0.294, 16.391],
    [0.353, 20.21],
    [0.382, 22.756],
    [0.419, 25.26],
    [0.477, 23.074],
    [0.52, 20.845],
    [0.593, 19.093],
    [0.688, 17.5],
    [0.855, 16.225],
    [1.037, 15.427],
    [1.205, 14.948],
    [1.423, 14.627],
    [1.452, 15.741],
    [1.503, 14.785],
    [1.736, 14.623],
    [1.955, 14.303],
    [2.21, 14.141],
    [2.494, 13.819],
    [2.763, 13.338],
    [3.12, 13.334],
    [3.382, 13.013],
    [3.404, 9.352],
    [3.418, 4.895],
    [3.45, 0.0]
])

EstesF15_mass = np.array([
    [0.0, 60.0],
    [0.148, 59.3164],
    [0.228, 58.3541],
    [0.294, 57.2108],
    [0.353, 55.905],
    [0.382, 55.1514],
    [0.419, 54.0771],
    [0.477, 52.3818],
    [0.52, 51.2397],
    [0.593, 49.4767],
    [0.688, 47.3744],
    [0.855, 43.9685],
    [1.037, 40.4849],
    [1.205, 37.3989],
    [1.423, 33.5],
    [1.452, 32.9674],
    [1.503, 32.026],
    [1.736, 27.8823],
    [1.955, 24.0514],
    [2.21, 19.6652],
    [2.494, 14.8632],
    [2.763, 10.4455],
    [3.12, 4.68731],
    [3.382, 0.51289],
    [3.404, 0.215344],
    [3.418, 0.0947253],
    [3.45, 0.0]
])


KlimaC2_mass_normalized = np.copy(KlimaC2_mass)
KlimaC2_mass_normalized[:, 1] = KlimaC2_mass[:, 1] / KlimaC2_mass[0, 1]

KlimaC6_mass_normalized = np.copy(KlimaC6_mass)
KlimaC6_mass_normalized[:, 1] = KlimaC6_mass[:, 1] / KlimaC6_mass[0, 1]

KlimaD3_mass_normalized = np.copy(KlimaD3_mass)
KlimaD3_mass_normalized[:, 1] = KlimaD3_mass[:, 1] / KlimaD3_mass[0, 1]

KlimaD9_mass_normalized = np.copy(KlimaD9_mass)
KlimaD9_mass_normalized[:, 1] = KlimaD9_mass[:, 1] / KlimaD9_mass[0, 1]

EstesF15_mass_normalized = np.copy(EstesF15_mass)
EstesF15_mass_normalized[:, 1] = EstesF15_mass[:, 1] / EstesF15_mass[0, 1]


# Change motor weights etc here, interpolation still happens according to the mass loss curve
motor_specs = {
    "Klima C2": {"propellant_mass": 0.0113, "burn_time": KlimaC2[-1, 0], "total_motor_mass": 0.0224},
    "Klima C6": {"propellant_mass": 0.0096, "burn_time": KlimaC6[-1, 0], "total_motor_mass": 0.0205},
    "Klima D3": {"propellant_mass": 0.017, "burn_time": KlimaD3[-1, 0], "total_motor_mass": 0.0279},
    "Klima D9": {"propellant_mass": 0.0161, "burn_time": KlimaD9[-1, 0], "total_motor_mass": 0.0271},
    "Estes F15": {"propellant_mass": 0.060, "burn_time": EstesF15[-1, 0], "total_motor_mass": 0.102},
    "None": {"propellant_mass": 0.0, "burn_time": 0, "total_motor_mass": 0.0},
}

# Interpolation functions for normalized mass
mass_profiles = {
    "Klima C2": interp1d(KlimaC2_mass_normalized[:, 0], KlimaC2_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima C6": interp1d(KlimaC6_mass_normalized[:, 0], KlimaC6_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima D3": interp1d(KlimaD3_mass_normalized[:, 0], KlimaD3_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Klima D9": interp1d(KlimaD9_mass_normalized[:, 0], KlimaD9_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
    "Estes F15": interp1d(EstesF15_mass_normalized[:, 0], EstesF15_mass_normalized[:, 1],
                         bounds_error=False, fill_value=(1.0, 0.0)),
}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Motor Selection")
    numOfMotors = st.slider("Number of Ascent Motors", 1, 6, value=2)
    motor_choice = st.selectbox("Select Motor for Ascent", ["Klima D3", "Klima D9", "Klima C2", "Klima C6", "Estes F15"])

    # Map motor names to thrust data arrays
    thrust_data_dict = {
        "Klima C2": KlimaC2,
        "Klima C6": KlimaC6,
        "Klima D3": KlimaD3,
        "Klima D9": KlimaD9,
        "Estes F15": EstesF15,
    }
    thrust_data = thrust_data_dict[motor_choice]
    
    st.subheader("PID Controller Parameters")
    Kp = st.number_input("Proportional Gain (Kp)", value=20.0, step=0.01)
    Ki = st.number_input("Integral Gain (Ki)", value=0.00, step=0.01)
    Kd = st.number_input("Derivative Gain (Kd)", value=6.0, step=0.01)
    setpoint = st.number_input("Setpoint (degrees)", value=0.0, step=1.0)
    
    st.subheader("Rocket Initial Conditions")
    theta0 = st.number_input("Initial Angle (degrees)", value=10.0, step=1.0)
    theta0 = np.radians(theta0)  # Convert to radians for calculations
    omega0 = st.number_input("Initial Angular rate (degrees/s)", value=0.0, step=1.0)
    omega0 = np.radians(omega0)  # Convert to radians for calculations

ascent_burn_time = motor_specs[motor_choice]["burn_time"]
ascent_propellant_mass = motor_specs[motor_choice]["propellant_mass"] * numOfMotors

with col2:
    st.subheader("TVC Non-idealities")
    TVC_backlash = st.number_input("TVC Backlash, total deadband (degrees)", value=0.0, step=0.01)
    TVC_backlash = np.radians(TVC_backlash)  # Convert to radians
    TVC_delay_s = st.number_input("TVC Delay (s)", value=0.014, step=0.001, format="%.3f")
    TVC_a0 = st.number_input("Transfer function a0 (numerator)", value=4102)
    TVC_b0 = st.number_input("Transfer function b0 (denominator)", value=4167)
    TVC_b1 = st.number_input("Transfer function b1 (denominator)", value=92.91)
    TVC_b2 = st.number_input("Transfer function b2 (denominator)", value=1)


    
    # Sampling and time parameters
    TVC_sampling_rate = 1000  # Hz
    TVC_Ts = 1.0 / TVC_sampling_rate
    TVC_duration = 0.15# seconds
    TVC_t = np.arange(0, TVC_duration, TVC_Ts)

    # Delay parameters
    TVC_delay = TVC_delay_s  # seconds
    TVC_delay_samples = int(TVC_delay / TVC_Ts)
    # Step input with delay
    TVC_u = np.zeros_like(TVC_t)
    TVC_u[TVC_delay_samples:] = 1  # Shift the step by delay

    # Transfer function (no delay included)
    num = [TVC_a0]
    den = [TVC_b2,TVC_b1,TVC_b0]
    G_ct = ctrl.tf(num, den)
    G_dt = ctrl.sample_system(G_ct, TVC_Ts, method='zoh')
    num_d = G_dt.num[0][0]  # numerator coefficients
    den_d = G_dt.den[0][0]  # denominator coefficients


    # Simulate response
    TVC_t_out, y = ctrl.forced_response(G_ct, TVC_t, TVC_u)

    # Plot
    st.subheader("Step Response")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(TVC_t, y, label="Output angle")
    ax.plot(TVC_t, TVC_u, '--', label="'Delayed' Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("TVC Step Response with Delay")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)




def apply_backlash(input_signal, previous_real_output, backlash):
    if input_signal > previous_real_output + backlash:
        previous_real_output = input_signal - backlash
    elif input_signal < previous_real_output - backlash:
        previous_real_output = input_signal + backlash
    return previous_real_output   

# Interpolate thrust curve
times = thrust_data[:, 0]
thrusts = thrust_data[:, 1]
thrust_func = interp1d(times, thrusts, bounds_error=False, fill_value=0.0)

# === SIMULATION SETUP ===
velocity = 0.0
altitude = 0.0

time_array = np.zeros(n_steps)
altitude_array = np.zeros(n_steps)
velocity_array = np.zeros(n_steps)
thrust_array = np.zeros(n_steps)
mass_array = np.zeros(n_steps)
MMOI_array = np.zeros(n_steps)
lever_array = np.zeros(n_steps)
theta_array = np.zeros(n_steps)
omega_array = np.zeros(n_steps)
alpha_array = np.zeros(n_steps)
desired_TVC_angle_array = np.zeros(n_steps)
delayed_desired_TVC_angle_array = np.zeros(n_steps)
delayed_2nd_order_TVC_angle_array = np.zeros(n_steps)
delayed_2nd_order_backlash_TVC_angle_array = np.zeros(n_steps)

delay_buffer = deque([0.0] * TVC_delay_samples, maxlen=TVC_delay_samples)

#TVC non idealities
max_order = max(len(num_d), len(den_d))
u_hist = np.zeros(max_order)  # input buffer
y_hist = np.zeros(max_order)  # output buffer

# Backlash
previous_real_TVC_angle = 0.0


theta = theta0
omega = omega0
alpha = 0.0
last_error = theta0 - np.radians(setpoint)
integral_error = 0.0
current_MMOI = start_MMOI
current_lever = start_lever

# === SIMULATION LOOP ===
for i in range(n_steps):
    t = i * dt
    
    # Interpolate thrust and MMOI based on time
    thrust = thrust_func(t)*numOfMotors
    if t < ascent_burn_time:
        current_MMOI = start_MMOI - (start_MMOI - end_MMOI) * (t / ascent_burn_time)
        current_lever = start_lever - (start_lever - end_lever) * (t / ascent_burn_time)
    else:
        current_MMOI = start_MMOI - (start_MMOI - end_MMOI) * (t / t_max)
        current_lever = start_lever - (start_lever - end_lever) * (t / t_max)
    
    # Convert setpoint to radians
    setpoint_rad = np.radians(setpoint)  # If setpoint is in degrees
    error = setpoint_rad - theta

    # PID control (assuming gains tuned for radians)
    integral_error += error * dt
    derivative_error = (error - last_error) / dt
    u_pid = Kp * error + Ki * integral_error + Kd * derivative_error
    last_error = error

    # Initialize
    desired_side_force = 0.0
    desired_TVC_angle = 0.0

    if current_MMOI != 0 and current_lever != 0 and thrust != 0:
        desired_torque = u_pid * current_MMOI
        desired_side_force = desired_torque / current_lever

        # Limit to max available thrust
        desired_side_force = np.clip(desired_side_force, -thrust, thrust)

        # Compute TVC angle
        desired_TVC_angle = np.arcsin(desired_side_force / thrust)

    # Limit TVC angle
    max_tvc_rad = np.radians(max_tvc_deg)
    desired_TVC_angle = np.clip(desired_TVC_angle, -max_tvc_rad, max_tvc_rad)

    # Delay buffer (use deque ideally)
    delay_buffer.append(desired_TVC_angle)
    if i >= TVC_delay_samples:
        delayed_desired_TVC_angle = delay_buffer[0]  # this is the delayed value
    else:
        delayed_desired_TVC_angle = 0.0

    
    # Apply second order transfer function to the delayed angle
    u_hist = np.roll(u_hist, 1)
    y_hist = np.roll(y_hist, 1)
    u_hist[0] = delayed_desired_TVC_angle
    output = np.dot(num_d, u_hist[:len(num_d)]) - np.dot(den_d[1:], y_hist[1:len(den_d)])
    output /= den_d[0]  # normalize
    y_hist[0] = output
    
    # Apply backlash
    real_TVC_angle = apply_backlash(output, previous_real_TVC_angle, TVC_backlash)
    real_side_thrust = thrust * np.sin(real_TVC_angle)
    previous_real_TVC_angle = real_TVC_angle
    
    
    # Calculate angular acceleration
    alpha = (real_side_thrust * current_lever) / current_MMOI
    # Update angular velocity and angle
    omega += alpha * dt
    theta += omega * dt
    
    
    # Update all arrays for plotting
    time_array[i] = t
    thrust_array[i] = thrust
    MMOI_array[i] = current_MMOI
    lever_array[i] = current_lever
    theta_array[i] = np.degrees(theta)  # Convert to degrees for plotting
    omega_array[i] = np.degrees(omega)  # Convert to degrees for plotting
    alpha_array[i] = np.degrees(alpha)  # Convert to degrees for plotting
    desired_TVC_angle_array[i] = np.degrees(desired_TVC_angle)  # Convert to degrees for plotting
    delayed_desired_TVC_angle_array[i] = np.degrees(delayed_desired_TVC_angle)  # Convert to degrees for plotting
    delayed_2nd_order_TVC_angle_array[i] = np.degrees(output)  # Convert to degrees for plotting
    delayed_2nd_order_backlash_TVC_angle_array[i] = np.degrees(real_TVC_angle)  # Convert to degrees for plotting
    
# === PLOTTING RESULTS ===
st.subheader("Simulation Results")

def plot_time_series(y, label, ylabel):
    ax.plot(time_array, y, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Plot 1: Thrust vs Time
st.subheader("Thrust Over Time")
plot_time_series(thrust_array, "Thrust [N]", "Thrust [N]")


# Plot 2: MMOI and Lever Arm
st.subheader("MMOI and Lever Arm Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_array, MMOI_array, label="MMOI [kg.m²]")
ax.plot(time_array, lever_array, label="Lever Arm [m]")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend()
st.pyplot(fig)


# Plot 3: Angular Kinematics
st.subheader("Angular Position, Velocity, and Acceleration")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_array, theta_array, label="Theta [deg]")
ax.plot(time_array, omega_array, label="Omega [deg/s]")
ax.plot(time_array, alpha_array, label="Alpha [deg/s²]")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle / Rate")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Plot 4: TVC Control Angles (only a small time frame)
st.subheader("TVC Command Angles Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
# limit x-axis
ax.set_xlim(0, 2)  # Adjust as needed for a small time frame
ax.plot(time_array, desired_TVC_angle_array, label="Desired")
ax.plot(time_array, delayed_desired_TVC_angle_array, label="Delayed")
ax.plot(time_array, delayed_2nd_order_TVC_angle_array, label="2nd Order Delayed")
ax.plot(time_array, delayed_2nd_order_backlash_TVC_angle_array, label="With Backlash")
ax.set_xlabel("Time [s]")
ax.set_ylabel("TVC Angle [deg]")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.sidebar.subheader("TVC angles")
st.sidebar.pyplot(fig)


# Plot 4: TVC Control Angles (only a small time frame)
st.subheader("Master plot")
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(time_array, theta_array, label="Theta [deg]")
ax.plot(time_array, thrust_array, label="Thrust [N]")
ax.plot(time_array, desired_TVC_angle_array, label="Desired TVC Output [deg]")
ax.plot(time_array, delayed_2nd_order_backlash_TVC_angle_array, label="Real TVC Output [deg]")
ax.set_xlabel("Time [s]")
ax.set_ylabel("")
ax.grid(True)
ax.legend()
st.pyplot(fig)


st.sidebar.subheader("Stability")
st.sidebar.pyplot(fig)