import torch
import math
import warnings
from torch import Tensor
import numpy as np
import json

import os

from dev_params import DeviceParams

PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ActionNormalizer():
    
    def __init__(self, action_space_low, action_space_high):
         
        self.action_space_low = action_space_low     
        self.action_space_high = action_space_high

    def action(self, action: np.ndarray) -> np.ndarray:
        
        low = self.action_space_low   
        high = self.action_space_high 

        scale_factor = (high - low) / 2     
        reloc_factor = high - scale_factor  

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high) 

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        
        low = self.action_space_low
        high = self.action_space_high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor  
        action = np.clip(action, -1.0, 1.0)

        return action


class OutputParser2(DeviceParams):  
    
    def __init__(self, CktGraph):
        self.ckt_hierarchy = CktGraph.ckt_hierarchy 
        self.op = CktGraph.op
        self.working_dir = None  
        super().__init__(self.ckt_hierarchy)       

    def set_working_dir(self, dir_path):
        
        self.working_dir = dir_path

    def ac(self, file_name):
        try:
            file_dir = self.working_dir
            AMP_NMCF_ac = open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r')
            lines_ac = AMP_NMCF_ac.readlines()     
            freq = []                       
            cmrrdc_ac = []
            PSRP_ac = []
            PSRN_ac = []
            dcgain_ac = []
            for line in lines_ac:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']    
                freq.append(float(Vac[0]))           
                cmrrdc_ac.append(float(Vac[1]))
                PSRP_ac.append(float(Vac[3]))
                PSRN_ac.append(float(Vac[5]))
                dcgain_ac.append(float(Vac[7]))
                
            return freq, cmrrdc_ac, PSRP_ac, PSRN_ac, dcgain_ac
        except:
            print("Simulation errors, no .AC simulation results.")

    def GBW_PM(self, file_name):
        try:
            file_dir = self.working_dir
            AMP_NMCF_GBW_PM = open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r') 
            lines_GBW_PM = AMP_NMCF_GBW_PM.readlines()     
            freq = []                       
            GBW_ac = []
            phase_margin_ac = []
            for line in lines_GBW_PM:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']     
                freq.append(float(Vac[0]))            
                GBW_ac.append(float(Vac[1]))
                phase_margin_ac.append(float(Vac[3]))
                
            return freq, GBW_ac, phase_margin_ac
        except:
            print("Simulation errors, no .GBW_PM simulation results.")
            
    def dc(self, file_name):
        try:
            file_dir = self.working_dir
            AMP_NMCF_dc = open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r')
            lines_dc = AMP_NMCF_dc.readlines()
            Temp_dc = []                     
            TC_dc = []
            Power_dc = []
            vos_dc = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                Temp_dc.append(float(Vdc[0]))
                TC_dc.append(float(Vdc[1]))
                Power_dc.append(float(Vdc[3])) 
                vos_dc.append(float(Vdc[5]))
          
            return Temp_dc, TC_dc, Power_dc, vos_dc
        except:
            print("Simulation errors, no .OP simulation results.")
      
    def tran(self, file_name):
        try:
            file_dir = self.working_dir
            AMP_NMCF_tran = open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r')
            lines_tran = AMP_NMCF_tran.readlines()
            time = []                         
            sr_rise = []
            sr_fall = []
            for line in lines_tran:
                line = line.split(' ')
                line = [i for i in line if i != '']
                time.append(float(line[0]))
                sr_rise.append(float(line[1]))
                sr_fall.append(float(line[3]))

            return time, sr_rise, sr_fall
        except:
                print("Simulation errors, no .TRAN simulation results.")

            
    def dcop(self, file_name):
        try:
            file_dir = self.working_dir
            AMP_NMCF_op = open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r')
            lines_op = AMP_NMCF_op.readlines()
            for index, line in enumerate(lines_op):
                if line == "Values:\n":       
                    start_idx = index
            _lines_op = lines_op[start_idx+2:-1]     
            lines_op = []
            for _line in _lines_op:           
                lines_op.append(float(_line.split('\n')[0].split('\t')[1]))   
            
            num_dev = len(self.ckt_hierarchy)
            num_dev_params_mos = len(self.params_mos)
            num_dev_params_r = len(self.params_r)
            num_dev_params_c = len(self.params_c)
            num_dev_params_i = len(self.params_i)
            num_dev_params_v = len(self.params_v)
            
            idx = 0        
            for i in range(num_dev):        
                dev_type = self.ckt_hierarchy[i][3]
                if dev_type == 'm' or dev_type == 'M':   
                    for j in range(num_dev_params_mos):
                        param = self.params_mos[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_mos
                elif dev_type == 'r' or dev_type == 'R':  
                    for j in range(num_dev_params_r):
                        param = self.params_r[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_r
                elif dev_type == 'c' or dev_type == 'C':
                    for j in range(num_dev_params_c):
                        param = self.params_c[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_c
                elif dev_type == 'i' or dev_type == 'I':
                    for j in range(num_dev_params_i):
                        param = self.params_i[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_i
                elif dev_type == 'v' or dev_type == 'V':
                    for j in range(num_dev_params_v):
                        param = self.params_v[j]
                        self.op[list(self.op)[i]][param] = lines_op[idx+j]
                    idx = idx + num_dev_params_v
                else:
                    None
            
            return self.op
        except:
            print("Simulation errors, no .OP simulation results.")

    def analyze_amplifier_performance(self, vinp, vout, time, d0):
        vinp = np.array(vinp)  
        vout = np.array(vout)
        time = np.array(time)
        def get_step_parameters(vinp, time):
            dv = np.diff(vinp)
            if len(np.where(dv > 0)[0]) == 0 or len(np.where(dv < 0)[0]) == 0:
                return None, None, None, None
            
            t0 = time[np.where(dv > 0)[0][0]]
            t1 = time[np.where(dv < 0)[0][0]]
            v0 = np.median(vinp[time < t0])
            v1 = np.median(vinp[(time > t0) & (time < t1)])
            return v0, v1, t0, t1
        v0, v1, t0, t1 = get_step_parameters(vinp, time)
        
        if v0 is None:  
            return np.nan, np.nan, np.nan, False, np.nan, np.nan, np.nan, np.nan
    
        pre_step_data = vout[time < t0]
        delta0 = (pre_step_data - v0) / v0
        d0_settle = np.mean(np.abs(delta0))
        stable = not np.any(np.abs(delta0) > d0)
    
        def find_settling_time_index(delta, d0):
            for i in range(len(delta)):
                if np.all(np.abs(delta[i:]) < d0):
                    return i
            return None
    
        def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
            idx = (time >= start_t) & (time <= end_t)
            vout_segment = vout[idx]
            time_segment = time[idx]
    
            target_value = v0 + (v1 - v0) / 2
            idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
            if idx_target is None:
                SR = np.nan
            else:
                SR = np.gradient(vout_segment, time_segment)[idx_target]
    
            if mode == 'positive':
                delta = (vout_segment - v1) / v1
            else:
                delta = (vout_segment - v0) / v0

            idx_settle = find_settling_time_index(delta, d0)
            if idx_settle is None:
                settling_time = np.nan
                d_settle = np.mean(np.abs(delta))
            else:
                settling_time = time_segment[idx_settle] - start_t
                d_settle = np.mean(np.abs(delta[idx_settle:]))
            return SR, settling_time, d_settle

        SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

        SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

        return d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n    

    def extract_tran_data(self,file_name):
        time_points = []
        raw_data = []
        vin_data = []
        vout_data = []
        time_data = []
        data_section = False
        file_dir = self.working_dir
        with open(f'{SPICE_NETLIST_DIR}/{file_dir}/{file_name}', 'r')as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    if line.startswith('Values:'):
                        data_section = True
                        continue
                    if data_section:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            time_points.append(int(parts[0]))
                            raw_data.append(float(parts[1]))
                        else:
                            raw_data.append(float(parts[0]))  
    
        if len(time_points) != len(raw_data)/3:
            print('Error in extracting transient data')
            return None, None
        for i in time_points:
            time_data.append(raw_data[3*i])
            vin_data.append(raw_data[3*i+2])
            vout_data.append(raw_data[3*i+1])
    
        return time_data, vin_data, vout_data

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / np.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std) 
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()  

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def trunc_normal(mean, std, a=-1, b=1):  
    
    
    output = np.zeros(len(mean))
    
    for i in range(len(mean)):
        output[i] = trunc_normal_(torch.empty(1), mean[i], std, a, b)[0]

    return output
    
def aggregate_rewards(results_dict, attention_weights):
    
    total_reward = 0
    for weight, (corner_idx, result) in zip(attention_weights, results_dict.items()):
        total_reward += weight * result['reward']
    return total_reward
    