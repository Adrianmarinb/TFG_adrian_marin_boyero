# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:42:06 2023

@author: adria
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import pandas as pd
import math
import os
os.chdir("D:/TFG/Archivos tilting/Tilting pab0")
print(os.getcwd())
print("------------------------------------------------------------")

# PLATEAU: linea1
linea1 = 8.5874

# ----------------------------------------------------------------------------------------
# PROGRAMA DE ANÁLISIS Y CÁLCULO DE ANGULOS PARA GOTAS EN SISTEMA TILTIN
# ----------------------------------------------------------------------------------------

# 1) Extraemos los datos del documento de nombre_file_omega para obtener la evolución de la posición del láser respecto al tiempo
# 2) Extraemos datos de posición de nombre_file_data para obtener la posición de los puntos a lo largo del tiempo
# 3) Derivamos  α(t)
# 4) Calculamos las distancias útiles de la gota para los diferentes t
# 5) Buscamos los puntos en los que encontramos plateaus de longitud de la gota, mostramos gráficas
# 6) Calculamos la fuerza gravitatoria para el platau que nos interesa


#-------------------------------------------------------------
# DATA READING
#-------------------------------------------------------------
#We extract omega from .txt and remove points with no values

Surface = 'pab0'
Volume = 100
numero = 5

nombre_file_angle = 'I-S' + str(Surface) +'-V' + str(Volume) + '-N'+ str(1) + '-angle.txt'
nombre_file_data = 'I-S' + str(Surface) + '-V' + str(Volume) + '-N'+ str(numero) + '-data.txt'
output_file = 'Video Analysis tilting - S'  + str(Surface) + ', V:' + str(Volume) + 'µL, Nº' + str(numero)

print('Video Analysis Tilting - S'  + str(Surface) + ', V:' + str(Volume) + 'µL, Nº' + str(numero))
print()

"Video Analysis tilting - Spab0, V:50µL, Nº2"

#-------------------------------------------------------------

#-------------------------------------------------------------
#Tomamos los datos de la evolución angular y arreglamos el formato

# Extraemos los datos de la columna AngleX(º)
df = pd.read_csv(nombre_file_angle, delimiter="\t")
time_column = df.iloc[:, 0]
angle_values = df.iloc[:, 8]

# Buscamos el lugar en el que empeza a aumentar el ángulo
significant_change_index = np.argmax(np.diff(angle_values) > 0.05)

# Convertimos el tiempo a formato convencional (HH:MM:SS.SSS)
time_values = pd.to_datetime(time_column, format='%H:%M:%S.%f')

# Calculamos diferencias en los valores temporales
time_diff = time_values - time_values[significant_change_index]

# Convertimos a segundos
time_diff_seconds_pd = time_diff.dt.total_seconds()
time_diff_seconds = time_diff_seconds_pd.tolist()

#-------------------------------------------------------------

#-------------------------------------------------------------
#Extraemos datos del .txt y eliminamos aquellos puntos sin información

with open(nombre_file_data) as fdata:
    lines = fdata.readlines()[3:]

t_data = []
x_front = []
y_front = []
x_behind = []
y_behind = []
x_center = []
y_center = []


for line in lines:
    values = line.strip().split('\t')
    t_val = float(values[0])
    t_data.append(t_val)
    
    x_front_val = float(values[1]) if len(values) > 1 else 0
    x_front.append(x_front_val)
    
    y_front_val = float(values[2]) if len(values) > 2 else 0
    y_front.append(y_front_val)
    
    x_behind_val = float(values[3]) if len(values) > 3 else 0
    x_behind.append(x_behind_val)
    
    y_behind_val = float(values[4]) if len(values) > 4 else 0
    y_behind.append(y_behind_val)
    
    x_center_val = float(values[5]) if len(values) > 5 else 0
    x_center.append(x_center_val)
    
    y_center_val = float(values[6]) if len(values) > 6 else 0
    y_center.append(y_center_val)
        
#-------------------------------------------------------------
#Pasamos de m a cm

for i in range(len(t_data)):
    x_front[i] = 100*x_front[i]
    y_front[i] = 100*y_front[i]
    x_behind[i] = 100*x_behind[i]
    y_behind[i] = 100*y_behind[i]
    x_center[i] = 100*x_center[i]
    y_center[i] = 100*y_center[i]
    
#-------------------------------------------------------------
# DATA FILTERING
#-------------------------------------------------------------

#-------------------------------------------------------------
#Arreglamos ceros raros de behind y center

t_data_numpy = np.array(t_data)
x_front_numpy = np.array(x_front)
y_front_numpy = np.array(y_front)
x_behind_numpy = np.array(x_behind)
y_behind_numpy = np.array(y_behind)
x_center_numpy = np.array(x_center)
y_center_numpy = np.array(y_center)

mask_x_front = x_front_numpy != 0
mask_y_front = y_front_numpy != 0
mask_x_behind = x_behind_numpy != 0
mask_y_behind = y_behind_numpy != 0
mask_x_center = x_center_numpy != 0
mask_y_center = y_center_numpy != 0

#-------------------------------------------------------------
#Definimos una lista para las diferencias de dist. entre front y behind

difference = [0.0]*len(x_front)

for i in range(len(x_front)):
    difference[i] = ( (x_front[i] - x_behind[i])**2 + (y_front[i] - y_behind[i])**2 )**0.5
    
difference_numpy = np.array(difference)
mask_difference = difference_numpy != x_front_numpy

#-------------------------------------------------------------
#Definimos una lista para radio front

radio_front = [0.0]*len(x_front)

for i in range(len(x_front)):
    radio_front[i] = ( (x_front[i])**2 + (y_front[i])**2 )**0.5
    
radio_front_numpy = np.array(radio_front)

mask_rb = radio_front_numpy != x_front_numpy

#-------------------------------------------------------------
#Definimos una lista para radio behind

radio_behind = [0.0]*len(x_front)

for i in range(len(x_front)):
    radio_behind[i] = ( (x_behind[i])**2 + (y_behind[i])**2 )**0.5
    
radio_behind_numpy = np.array(radio_behind)
    
mask_rf = radio_behind_numpy != x_front_numpy

#-------------------------------------------------------------
#Definimos una lista para radio center

radio_center = [0.0]*len(x_front)

for i in range(len(x_front)):
    radio_center[i] = ( (x_center[i])**2 + (y_center[i])**2 )**0.5
    
radio_center_numpy = np.array(radio_center)
    
mask_rc = radio_center_numpy != x_front_numpy

#-------------------------------------------------------------
# REGRESSION FOR ANGLE EVOLUTION
#-------------------------------------------------------------

def interpolate(x, y, x_new):
    """
    Linear interpolation of y values for new x values
    """
    y_interp = np.interp(x_new, x, y)
    return y_interp

#-------------------------------------------------------------
#Creamos un fit para el ángulo

t_new = np.linspace(time_diff_seconds[0], time_diff_seconds[-1], num=10000)
angle_interp = interpolate(time_diff_seconds, angle_values, t_new)

coeffs_angle = np.polyfit(t_new, angle_interp, 15)
f_angle = np.poly1d(coeffs_angle)
angle_t = f_angle(t_new) # Interpolación para θ(t)

angle_t_degrees = np.zeros(len(angle_t))
angle_values_degrees = np.zeros(len(angle_values))

for i in range(len(angle_t)):
    angle_t_degrees[i] = math.degrees(angle_t[i])
    
for i in range(len(angle_values)):
    angle_values_degrees[i] = math.degrees(angle_values[i])

#-------------------------------------------------------------
# RMSE

# Calculate the squared errors
squared_errors = (angle_values - f_angle(time_diff_seconds))**2

# Calculate RMSE
rmse = np.sqrt(np.mean(squared_errors))

# Print the RMSE
print(f"RMSE: {rmse:.4f}")

# ERROR associated to omega
error_angle = math.radians(abs(f_angle(linea1 - 0.01) - f_angle(linea1 + 0.01)))
print(f"• The error associated to θ is: {error_angle:.4f}")
print()

#-------------------------------------------------------------
# REGRESSION FOR OTHER MEASUREMENTS
#-------------------------------------------------------------

def get_closest_indices(t_data, linea1):

  # Find the index of the element closest to linea1 (including ties)
  idx = np.argmin(np.abs(t_data - linea1))

  # Check if there are elements before and after (considering potential ties)
  if idx > 0 and t_data[idx - 1] <= linea1:
    idx_before = idx - 1
  else:
    idx_before = idx

  if idx < len(t_data) - 1 and t_data[idx + 1] > linea1:
    idx_after = idx + 1
  else:
    idx_after = idx

  return idx_before, idx_after

# -----------------------------------------------------

def interpolate_values(t_data, radio_center, linea1):

  # Find the indices of the data points closest to linea1
  idx_before, idx_after = get_closest_indices(t_data, linea1)

  # Check if linea1 is within the range of the data
  if idx_before < 0 or idx_after >= len(t_data):
    print("WARNING: linea1 ({}) is outside the range of the data.".format(linea1))
    return None

  # Extract the time and radio_center values for interpolation
  t1, t2 = t_data[idx_before], t_data[idx_after]
  r1, r2 = radio_center[idx_before], radio_center[idx_after]

  # Perform linear interpolation
  slope = (r2 - r1) / (t2 - t1)
  radio_center_linea1 = r1 + slope * (linea1 - t1)

  return radio_center_linea1

# -----------------------------------------------------

print("\U0001F4C8 Approximate data for t =", linea1, "s:")

# Call the interpolation function to get the droplet length at linea1:
droplet_length = interpolate_values(t_data_numpy[mask_rc], difference_numpy[mask_rc], linea1)
print(f"   • Droplet length:  {droplet_length:.4f} cm")

#-------------------------------------------------------------
# OTHER CALCULATIONS
#-------------------------------------------------------------

gravity_granada = 9.796933 #m/s^2
density = 0.99823 #kg/l

number = 3
constant = 3

print( (f_angle(number) - f_angle(number + constant)) * 1/(number - (number + constant)))

angle_plateau = math.radians(f_angle(linea1))
tilting_force = (density * Volume * pow(10, -6) * math.sin(angle_plateau) ) / (droplet_length * 0.01)

#print("Angular velocity for the first plateau is", f(linea1))
print(f"   • θ(t) for plateau is: {angle_plateau:.4f} rads")
print()
print("\U0001F4A7 Calculated centrifugal force for t =", linea1, "s:")
print(f"   • For measured droplet: {tilting_force:.4e} N/m")


#-------------------------------------------------------------
# OTHER CALCULATIONS
#-------------------------------------------------------------

fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios':[1, 2]}, figsize=(22,12))
plt.style.use('default')

# plt.suptitle(output_file, fontsize=40)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

gs = gridspec.GridSpec(2, 1)

# Subplot 1 ---------------- (ANGULAR DATA) ----------------
ax1 = plt.subplot(gs[0, 0])
ax1.set_xlim(-0.1, t_data_numpy[-1] + 0.2)
plt.scatter(time_diff_seconds, angle_values, 100,)  #label='Raw absolute angle data from Tracker')
plt.scatter(t_new, angle_t, 10)
plt.axvline(x=linea1, color='black', linestyle='--', lw=2)
plt.xlabel('Time (s)', fontsize=35, labelpad=(30))
plt.ylabel('α (degrees)', fontsize=35, labelpad=(30))
# plt.title('Absolute angle vs Time', fontsize=40)  
plt.legend(loc=0, prop={'size': 20})
plt.grid()
plt.axvline(0, color='black', lw=3)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)

legend_markers = [
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072BD', markersize=20),
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D95319', markersize=20),
      ]

plt.legend(legend_markers, ["Inclination angle of the platform α",'Numerical interpolation'], loc=0, prop={'size': 35})

# Subplot 2 ---------------- (MEASUREMENTS) ----------------

ax = plt.subplot(gs[1, 0])
ax.set_xlim(-0.1, t_data_numpy[-1] + 0.2)
ax.set_ylim(0.61, max(difference_numpy) + 0.1)
plt.axvline(x=linea1, color='black', linestyle='--', lw=2)
# plt.scatter(t_data_numpy[mask_rf], radio_front_numpy[mask_rf], 30, color='blue', label='Front position')
# plt.scatter(t_data_numpy[mask_rb], radio_behind_numpy[mask_rb], 30, color='red', label='Back position')
# plt.scatter(t_data_numpy[mask_rc], radio_center_numpy[mask_rc], 30, color='green', label='Center position')
plt.scatter(t_data_numpy[mask_difference], difference_numpy[mask_difference], 30, color='#3CB371')
plt.xlabel('Time (s)', fontsize=35, labelpad=(30))
plt.ylabel('Lateral Length (cm)', fontsize=35, labelpad=(30))
plt.show()
plt.legend(loc=2, prop={'size': 20})
plt.grid()
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)

# Inset axes in subplot 2 

use_inset_axes = True

if use_inset_axes:
    
    axins = inset_axes(ax, 3, 3, loc=9, bbox_to_anchor=(0.5, 0.45), bbox_transform=ax.figure.transFigure)
    
    plt.axvline(x=linea1, color='black', linestyle='--', lw=2)
    axins.scatter(t_data_numpy[mask_difference], difference_numpy[mask_difference], 30, color='#3CB371')
    
    # Zoomedin part
    x1 = 8.3
    x2 = 8.8
    y1 = 1
    y2 = 1.2
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
plt.tight_layout()

fonts2 = 55
    
# Subplot 3 ---------------- (POLAR) ----------------

# Assuming x_front, x_center_numpy, y_center_numpy, mask_y_front, x_behind, mask_y_behind, y_front_numpy, y_behind_numpy, mask_y_center, radio_front_numpy, mask_rf, radio_behind_numpy, mask_rb, radio_center_numpy, mask_rc are already defined

theta_front = np.arctan2(x_front_numpy[mask_y_front], y_front_numpy[mask_y_front]) + np.radians(45)
theta_behind = np.arctan2(x_behind_numpy[mask_y_behind], y_behind_numpy[mask_y_behind]) + np.radians(45)
theta_center = np.arctan2(x_center_numpy[mask_y_center], y_center_numpy[mask_y_center]) + np.radians(45)

fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': 'polar'})

# Plotting with transparency
sc1 = ax.scatter(theta_front, radio_front_numpy[mask_rf], color='blue', label='Front position', alpha=0.2, lw=8)
sc2 = ax.scatter(theta_behind, radio_behind_numpy[mask_rb], color='red', label='Back position', alpha=0.2, lw=8)
sc3 = ax.scatter(theta_center, radio_center_numpy[mask_rc], color='green', label='Center position', alpha=0.2, lw=8)

ax.set_theta_zero_location('W')
ax.set_theta_direction(-1)

# Creating custom legend markers without transparency
legend_markers = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=20)
]

ax.legend(legend_markers, ['Front position', 'Rear position', 'Apex position'], loc='upper left', bbox_to_anchor=(0.72, 0.8), prop={'size': fonts2})

plt.xlabel('Radius (cm)', fontsize=fonts2, labelpad=(40))
plt.ylabel('Angle (Degrees)', fontsize=fonts2, labelpad=(70))
plt.tick_params(axis='y', labelsize=fonts2)
plt.tick_params(axis='x', labelsize=fonts2)

ax.set_thetamin(0)  # Example: start angle (in degrees)
ax.set_thetamax(60)  # Example: end angle (in degrees)
ax.set_ylim(1, None)

plt.tight_layout()
plt.show()

'''
# Subplot 4 ---------------- (CARTESIAN) ----------------

plot = plt.figure(figsize=(22, 15))

plt.scatter(x_front_numpy[mask_y_front], y_front_numpy[mask_y_front], color='blue', label='Front position', alpha=0.2, lw=8)
plt.scatter(x_behind_numpy[mask_y_behind], y_behind_numpy[mask_y_behind], color='red', label='Back position', alpha=0.2, lw=8)
plt.scatter(x_center_numpy[mask_y_center], y_center_numpy[mask_y_center], color='green', label='Center position', alpha=0.2, lw=8)
plt.xticks(np.arange(-1, 1, 0.5))
plt.yticks(np.arange(1.2, 2.5, 0.5))


plt.grid()
plt.tight_layout()
plt.show()
'''
# ---------------- (MARGINS) ----------------

#Márgenes
plt.subplots_adjust(
top=0.895,
bottom=0.15,
left=0.1,
right=0.9,
hspace=0.975,
wspace=0.51
)







