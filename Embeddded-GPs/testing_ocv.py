from src.embedded_gp.create_OCV import create_OCV_full_cell, V_to_pos_half
import matplotlib.pyplot as plt
import pybamm as pb

param = pb.ParameterValues("Mohtat2020")

filename = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/Embeddded-GPs/src/Data/EPSCoR_Char_B4 - 024.csv'
i_D_start = 6419
i_D_end = 51115

SOC, Volt_ocv = create_OCV_full_cell(filename, i_D_start, i_D_end, 2.500*3600)
Volt_pos_half, p= V_to_pos_half(SOC, Volt_ocv)

param_pocp = param['Negative electrode OCP [V]'](SOC)
param_nocp = param['Positive electrode OCP [V]'](SOC)

param_ocp = param_nocp - param_pocp



plt.plot(SOC, Volt_ocv[::-1], label='Discharge Curve')
plt.plot(SOC, param_ocp, label='Potential Curve')
plt.legend()
plt.show()

plt.plot(SOC, p)
plt.plot()
plt.show()



t=1