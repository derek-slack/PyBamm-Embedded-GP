import pandas as pd
import numpy as np
import pybamm
import pybamm as pb
import matplotlib.pyplot as plt

param = pb.ParameterValues("Mohtat2020")

def create_OCV_full_cell(file_name, i_D_start, i_D_end, capacity):
    testing = pd.read_csv(file_name)

    Volt = testing['State'].to_numpy()[i_D_start:i_D_end]
    Amps = testing['Volts'].to_numpy()[i_D_start:i_D_end]
    t = testing['StepTime'].to_numpy()[i_D_start:i_D_end]

    tc = convert_time(t)


    SOC = np.ones(Volt.shape[0])
    Volt_ocv = np.zeros(Volt.shape[0])
    Volt_ocv[0] = float(Volt[0]) + 0.15
    mAh = capacity
    for i in range(1, len(Volt)):
        mAh = -Amps[i]*(tc[i] - tc[i-1]) + mAh
        SOC[i] = mAh/capacity
        Volt_ocv[i] = float(Volt[i]) + 0.15


    return SOC, Volt_ocv

def convert_time(t_vec):
    first_char = 11
    t_converted= np.zeros(len(t_vec))
    for i, t_i in enumerate(t_vec):
        t_hours = float(t_i[-first_char:-first_char+2])*3600
        t_minutes = float(t_i[-first_char+3:-first_char+5])*60
        t_seconds = float(t_i[-first_char+6:-first_char+8])
        t_milliseconds = float(t_i[-3:])
        t_converted[i] = t_hours + t_minutes + t_seconds + t_milliseconds
    return t_converted


def V_to_neg_half(SOC, V_ocv):
    pocp = param['Positive electrode OCP [V]'](np.linspace(0,1,len(V_ocv)))
    neg_half = V_ocv - pocp

    def OCP_neg(sto):
        return pybamm.Interpolant(SOC[::-1], neg_half[::-1], sto)

    return OCP_neg

def V_to_pos_half(SOC, V_ocv):
    nocp = param['Negative electrode OCP [V]'](np.linspace(1,0,len(V_ocv)))
    pos_half = V_ocv - nocp


    def OCP_pos(sto):
        return pybamm.Interpolant(SOC[::-1], pos_half, sto)

    return OCP_pos
