from src.embedded_gp.create_OCV import create_OCV_full_cell, V_to_pos_half
import matplotlib.pyplot as plt
import pybamm as pb
import numpy as np

param = pb.ParameterValues("Mohtat2020")

filename = '/home/WVU-AD/ds0172/Desktop/PyBamm-Embedded-GP-main/src-non-jax/src/Data/EPSCoR_Char_B4 - 024.csv'
i_D_start = 6419
i_D_end = 51115

SOC, Volt_ocv = create_OCV_full_cell(filename, i_D_start, i_D_end, 2.500*3600)
Volt_pos_half = V_to_pos_half(SOC, Volt_ocv)

stos = np.linspace(0,1,1000)

param_pocp = param['Negative electrode OCP [V]'](stos)
param_nocp = param['Positive electrode OCP [V]'](stos[::-1])

param_ocp = param_nocp - param_pocp
vp = []
for sto in stos:
    vp.append(Volt_pos_half(sto).evaluate()[0][0])

data_ocp = param_nocp - vp

plt.plot(stos, vp, label='data pocp')
plt.plot(stos, param_pocp, label='actual pocp')
# plt.plot(stos, param_nocp, label = 'negative')
plt.legend()
plt.show()
#
# # plt.plot(SOC, p)
# plt.plot()
# plt.show()
h=1


t=1