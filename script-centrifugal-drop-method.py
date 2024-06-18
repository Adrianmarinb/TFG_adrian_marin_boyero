import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import math
import os
os.chdir("D:\TFG\Archivos centrifugadora\Adri - 15 mar")
print("Location: " + os.getcwd())
print("------------------------------------------------------------")

# PLATEAU: linea1
linea1 = 4.1

# ----------------------------------------------------------------------------------------
# PROGRAMA DE ANÁLISIS Y CÁLCULO DE VELOCIDADES ANGULARES PARA GOTAS EN SISTEMA CENTRÍFUGO
# ----------------------------------------------------------------------------------------

# 1) Extraemos los datos del documento de nombre_file_omega para obtener la evolución de la posición del láser respecto al tiempo
# 2) Extraemos datos de posición de nombre_file_data para obtener la posición de los puntos a lo largo del tiempo
# 3) Derivamos θ(t) para conseguir ω(t) del láser (y el sistema)
# 4) Calculamos las distancias útiles de la gota para los diferentes t
# 5) Buscamos los puntos en los que encontramos plateaus de longitud de la gota, mostramos gráficas
# 6) Calculamos la velocidad angular para el platau que nos interesa y las fuerzas

#-------------------------------------------------------------
# DATA READING
#-------------------------------------------------------------
# We extract omega from .txt and remove points with no values

Surface = "pab0"
Volume = 100
numero = 1

nombre_file_omega = 'C-S' + str(Surface) + '-V' + str(Volume) + '-N'+ str(numero) + '-omega.txt'
nombre_file_data = 'C-S' + str(Surface) +'-V' + str(Volume) + '-N'+ str(numero) + '-data.txt'
output_file = 'Video Analysis - S'  + str(Surface) + ', V:' + str(Volume) + 'µL, Nº' + str(numero)

print('Video Analysis - S'  + "Tape" + ', V:' + str(Volume) + 'µL, Nº' + str(numero))
print()

"Video Analysis - R:3cm, V:50µL, Nº2"

with open(nombre_file_omega) as f:
    lines = f.readlines()[2:]

t = []
angle = []

for line in lines:
    values = line.strip().split('\t')
    t_val = float(values[0])
    t.append(t_val)
    if len(values) > 1:
        angle_val = float(values[1])
        angle_rad = math.radians(angle_val)  # Convert degrees to radians
        angle.append(angle_rad)
    else:
        angle.append(0)

for i in range(len(t)):
    if angle[i] == 0:
        t[i] = 0
    if t[i] == 0:
        angle[i] = 0

for i in range(len(angle)):
    if 0 in angle:
        angle.remove(0)
        t.remove(0)

for i in range(len(t)-1):
    if angle[i] == angle[i+1]:
        angle[i+1] = 0
        t[i+1] = 0

for i in range(len(angle)):
    if 0 in angle:
        angle.remove(0)
        t.remove(0)
        
#-------------------------------------------------------------
#We extract postions data from .txt and remove points with no values

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
    
    x_behind_val = float(values[1]) if len(values) > 1 else 0
    x_behind.append(x_behind_val)
    
    y_behind_val = float(values[2]) if len(values) > 2 else 0
    y_behind.append(y_behind_val)
    
    x_front_val = float(values[3]) if len(values) > 3 else 0
    x_front.append(x_front_val)
    
    y_front_val = float(values[4]) if len(values) > 4 else 0
    y_front.append(y_front_val)
    
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
# REGRESSION FOR ANGULAR VELOCITY
#-------------------------------------------------------------

def interpolate(x, y, x_new):
    """
    Linear interpolation of y values for new x values
    """
    y_interp = np.interp(x_new, x, y)
    return y_interp

def traditional_derivative(x, y):
    dy = [1.0]*len(y)
    for i in range(len(x)-1):
        dy[i] = (y[i+1]-y[i])/(abs(x[i+1]-x[i]))
    return dy

#-------------------------------------------------------------
#Interpolamos los valores iniciales

t_new = np.linspace(0, t[-1], num=10000)
angle_interp = interpolate(t, angle, t_new)

#-------------------------------------------------------------
# Hacemos regresión de θ(t), polinomio cúbico

coeffs_angle = np.polyfit(t_new, angle_interp, 2)
f_angle = np.poly1d(coeffs_angle)
angle_t = f_angle(t_new) # Interpolación para θ(t)
print("Angular velocity regression (cuadratic):")
print("• ω(t) = {:.3f}t^2 + {:.3f}t + {:.3f} rads/s".format(coeffs_angle[0], coeffs_angle[1], coeffs_angle[2]))

# La omega es la derivada

def f_omega(t, coeffs_angle):
    """
    Derivamos para obtener ω(t)
    """
    result = 2*coeffs_angle[0]*t + coeffs_angle[1]
    return result

omega_t = f_omega(t_new, coeffs_angle) # Obtenemos ω(t)

#-------------------------------------------------------------
# RMSE
squared_errors = (angle - f_angle(t))**2
rmse = np.sqrt(np.mean(squared_errors))

# RMSE for value t = linea1
RMSE_ω = abs(6*coeffs_angle[0]*linea1 + 2*coeffs_angle[1]) * rmse
print(f"• RMSE for ang. vel.: {RMSE_ω:.4f} rads/s")

# ERROR associated to omega
error_omega = f_omega(linea1 - 0.01, coeffs_angle) - f_omega(linea1 + 0.01, coeffs_angle)
print(f"• The error associated to ω is: {error_omega:.4f} rads/s")
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

# Call the interpolation function to get the measured center at linea1:
measured_center = interpolate_values(t_data_numpy[mask_rc], radio_center_numpy[mask_rc], linea1)
print(f"   • Apex center: {measured_center:.4f} cm")

# Call the interpolation function to get the averaged center at linea1:
average = np.array([0.0]*len(x_front))
average = (radio_front_numpy + radio_behind_numpy) / 2

averaged_center = interpolate_values(t_data_numpy[mask_rc], average[mask_rc], linea1)
print(f"   • Geometrical center: {averaged_center:.4f} cm")

# Call the interpolation function to get the droplet length at linea1:
droplet_length = interpolate_values(t_data_numpy[mask_rc], difference_numpy[mask_rc], linea1)
print(f"   • Droplet length:  {droplet_length:.4f} cm")

#-------------------------------------------------------------
# OTHER CALCULATIONS
#-------------------------------------------------------------

gravity_granada = 9.796933 #m/s^2
density = 0.99823 #kg/l

omega_plateau = f_omega(linea1, coeffs_angle)
centripetal_force_measured = (density * Volume * pow(10, -6) * measured_center * pow(omega_plateau, 2) ) / (droplet_length)
centripetal_force_averaged = (density * Volume * pow(10, -6) * averaged_center * pow(omega_plateau, 2) ) / (droplet_length)

#print("Angular velocity for the first plateau is", f(linea1))
print(f"   • ω(t) for plateau is: {omega_plateau:.4f} rads/s")
print()
print("\U0001F4A7 Calculated centrifugal force for t =", linea1, "s:")
print(f"   • For Apex center: {centripetal_force_measured:.4e} N/m")
print(f"   • For Geometrical center: {centripetal_force_averaged:.4e} N/m")

#-------------------------------------------------------------
# VISUALIZATION AND PLOTTING OF RESULTS
#-------------------------------------------------------------

#plt.figure(figsize=(22, 12))

fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios':[1, 2]}, figsize=(22,18))
plt.style.use('default')

#plt.suptitle(output_file, fontsize=30)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

gs = gridspec.GridSpec(2, 1)

fonts = 35


# Subplot 1 ---------------- (ANGULAR DATA) ----------------
ax1 = plt.subplot(gs[0, 0])
ax2 = ax1.twinx()

ax1.set_xlim(0, 6.5)
ax1.scatter(t, angle, 300, label='Raw absolute angle data from Tracker')
ax1.scatter(t_new, angle_t, 10, label='Cuadratic polynomial regression', color='#ADD8E6')
plt.xlim(1, 4.3)

ax1.set_xlabel('Time (s)', fontsize=fonts, labelpad=(30))
ax1.set_ylabel('θ (rads)', fontsize=fonts, labelpad=(30), color='#0072BD')
ax1.tick_params(axis="y", labelcolor='#0072BD', labelsize=fonts)
ax1.tick_params(axis="x", labelsize=fonts)

ax2.set_ylabel("ω (rads/s)", color='red', fontsize=fonts, labelpad=30)
ax2.scatter(t_new, omega_t, 50, color='red')
ax2.tick_params(axis="y", labelcolor='red', labelsize=fonts)

plt.axvline(x=linea1, color='black', linestyle='--', lw=2)
plt.xlabel('Time (s)', fontsize=fonts, labelpad=(30))

legend_markers = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072BD', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20),
]

plt.legend(legend_markers, [
    "Raw absolute angle data from Tracker",
    'Quadratic polynomial regression θ(t)',
    "ω(t) from derivating θ(t)"
], loc=3, prop={'size': fonts})

ax1.grid(axis='both')

plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)

plt.tight_layout()
plt.show()


# Subplot 2 ---------------- (MEASUREMENTS) ----------------

#plt.figure(figsize=(22, 12))

ax = plt.subplot(gs[1, 0])
ax.set_xlim(1, t_data_numpy[-1])
plt.xlim(1, 4.3)
plt.ylim(0.41, 1.6)
plt.axvline(x=linea1, color='black', linestyle='--', lw=2)

# plt.scatter(t_data_numpy[mask_rf], radio_front_numpy[mask_rf], 30, color='blue', label='Front position')
# plt.scatter(t_data_numpy[mask_rb], radio_behind_numpy[mask_rb], 30, color='red', label='Back position')
# plt.scatter(t_data_numpy[mask_rc], radio_center_numpy[mask_rc], 30, color='green', label='Center position')
plt.scatter(t_data_numpy[mask_difference], difference_numpy[mask_difference], 30, color='orange')
plt.xlabel('Time (s)', fontsize=fonts, labelpad=(30))
plt.ylabel('Lateral Length (cm)', fontsize=fonts, labelpad=(30))
plt.show()

# legend_markers = [
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=20)]

# plt.legend(legend_markers, ["Lateral length"], loc=0, prop={'size': fonts})

plt.grid()
plt.yticks(fontsize=fonts)
plt.xticks(fontsize=fonts)

plt.tight_layout()
plt.show()

use_inset_axes = False

if use_inset_axes:
    
    axins = inset_axes(ax, 3.5, 3.5, loc=9, bbox_to_anchor=(0.6, 0.44), bbox_transform=ax.figure.transFigure)
    
    plt.axvline(x=linea1, color='black', linestyle='--', lw=2)
    axins.scatter(t_data_numpy[mask_difference], difference_numpy[mask_difference], 30, color='orange')
    
    # Zoomedin part
    x1 = 3.96
    x2 = 4.22
    y1 = 1
    y2 = 1.3
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")


# Subplot 3 ---------------- (POLAR) ----------------

# Assuming x_front, x_center_numpy, y_center_numpy, mask_y_front, x_behind, mask_y_behind, y_front_numpy, y_behind_numpy, mask_y_center, radio_front_numpy, mask_rf, radio_behind_numpy, mask_rb, radio_center_numpy, mask_rc are already defined

theta_front = np.arctan2(x_front_numpy[mask_y_front], y_front_numpy[mask_y_front]) + np.degrees(-90)
theta_behind = np.arctan2(x_behind_numpy[mask_y_behind], y_behind_numpy[mask_y_behind]) + np.degrees(-90)
theta_center = np.arctan2(x_center_numpy[mask_y_center], y_center_numpy[mask_y_center]) + np.degrees(-90)

fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': 'polar'})

# Plotting with transparency
sc1 = ax.scatter(theta_front, radio_front_numpy[mask_rf], color='blue', label='Front position', alpha=0.4, lw=8)
sc2 = ax.scatter(theta_behind, radio_behind_numpy[mask_rb], color='red', label='Back position', alpha=0.4, lw=8)
sc3 = ax.scatter(theta_center, radio_center_numpy[mask_rc], color='green', label='Center position', alpha=0.2, lw=8)

ax.set_theta_zero_location('W')
ax.set_theta_direction(-1)

# Creating custom legend markers without transparency
legend_markers = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=20)
]

ax.legend(legend_markers, ['Front position', 'Rear position', 'Apex position'], loc='upper left', bbox_to_anchor=(0.5, 0.8), prop={'size': 40})

plt.xlabel('Radius (cm)', fontsize=40, labelpad=(-260))
plt.ylabel('Angle (Degrees)', fontsize=40, labelpad=(70))
plt.tick_params(axis='y', labelsize=40)
plt.tick_params(axis='x', labelsize=40)

ax.set_thetamin(3)  # Example: start angle (in degrees)
ax.set_thetamax(22)  # Example: end angle (in degrees)
ax.set_ylim(2.2, None)

plt.tight_layout()
plt.show()


'''
# ---------------- (MARGINS) ----------------

plt.subplots_adjust(
top=0.905,
bottom=0.115,
left=0.127,
right=0.945,
hspace=0.259,
wspace=0.5
)

'''










