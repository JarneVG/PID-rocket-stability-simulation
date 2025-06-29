# About

This code allows you to simulate your rocket with changes in MMOI and thrust during the entire flight.

The PID controller has as input the angle the rocket makes (in rad) and outputs the desired angular acceleration (in rad/s²).
This angular acceleration can be converted to a TVC angle based on the current thrust and current MMOI.

This simulation also incorporates TVC backlash and dynamics which has a very significant effect on the stability of the rocket so be sure to measure this as well to get accurate gains.

# Further improvements
- Simulating IMU gyro readings and adding an observer with gaussian noise
- Adding disturbances like wind

# Other sims
- For altitude simulations, see the RocketLandingAltitudeEstimation.py simulations
- For the landing algorithm, this is in the works