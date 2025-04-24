import torch
import numpy as np
import os
import shutil

class PVTGraph:
    def __init__(self):
        self.pvt_corners = {

            'tt_027C_1v80': [1, 0, 0, 0, 0, 27 ,1.8],  

            'fs_-25C_1v62': [0, 1, 0, 0, 0, -25 ,1.62],
            'fs_-25C_1v98': [0, 1, 0, 0, 0, -25 ,1.98],
            'fs_125C_1v80': [0, 1, 0, 0, 0, 125 ,1.62],
            'fs_125C_1v98': [0, 1, 0, 0, 0, 125 ,1.98],

            'sf_-25C_1v62': [0, 0, 1, 0, 0, -25 ,1.62],
            'sf_-25C_1v98': [0, 0, 1, 0, 0, -25 ,1.98],
            'sf_125C_1v80': [0, 0, 1, 0, 0, 125 ,1.62],
            'sf_125C_1v98': [0, 0, 1, 0, 0, 125 ,1.98],

            'ff_-25C_1v62': [0, 0, 0, 1, 0, -25 ,1.62],
            'ff_-25C_1v98': [0, 0, 0, 1, 0, -25 ,1.98],
            'ff_125C_1v80': [0, 0, 0, 1, 0, 125 ,1.62],
            'ff_125C_1v98': [0, 0, 0, 1, 0, 125 ,1.98],

            'ss_-25C_1v62': [0, 0, 0, 0, 1, -25 ,1.62],
            'ss_-25C_1v98': [0, 0, 0, 0, 1, -25 ,1.98],
            'ss_125C_1v80': [0, 0, 0, 0, 1, 125 ,1.62],
            'ss_125C_1v98': [0, 0, 0, 0, 1, 125 ,1.98]
        }
        
        self.PWD = os.getcwd()
        self.SPICE_NETLIST_DIR = f'{self.PWD}/simulations'
        
        
        
        self.num_corners = len(self.pvt_corners)
        self.corner_dim = 22
        self.node_features = np.zeros((self.num_corners, self.corner_dim))  
        
        for i, (corner, pvt_code) in enumerate(self.pvt_corners.items()):
            self.node_features[i, :7] = pvt_code  
            self.node_features[i, 7:21] = -np.inf  
            self.node_features[i, 21] = -np.inf    
            
        edges = []
        for i in range(self.num_corners):
            for j in range(self.num_corners):
                if i != j:
                    edges.append([i, j])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        


    def _clean_pvt_dirs(self):
        corner_prefixes = ['ss', 'ff', 'tt', 'sf', 'fs']
        
        for corner in os.listdir(self.SPICE_NETLIST_DIR):
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            if os.path.isdir(corner_dir) and any(corner.startswith(prefix) for prefix in corner_prefixes):
                print(f"Removing existing corner directory: {corner_dir}")
                shutil.rmtree(corner_dir)

    def _create_pvt_dirs(self):
        
        for corner in self.pvt_corners.keys():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            os.makedirs(corner_dir)
            
            spiceinit_content ="""* ngspice initialization for sky130
* assert BSIM compatibility mode with "nf" vs. "W"
set ngbehavior=hsa
* "nomodcheck" speeds up loading time
set ng_nomodcheck
set num_threads=8"""
            
            spiceinit_path = os.path.join(corner_dir, '.spiceinit')
            with open(spiceinit_path, 'w') as f:
                f.write(spiceinit_content)
                
    def _create_pvt_netlists(self):
        
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_ACDC.cir', 'r') as f:
            netlist_content = f.readlines()
            
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            self._copy_support_files(corner_dir)
            
            corner_netlist = []

            process = corner.split('_')[0]  
            
            for line in netlist_content:
                if line.startswith('.temp'):
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_ACDC_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)
    
    def _create_pvt_netlists_tran(self):
        
        with open(f'{self.SPICE_NETLIST_DIR}/AMP_NMCF_Tran.cir', 'r') as f:
            netlist_content = f.readlines()
            
        for corner, params in self.pvt_corners.items():
            corner_dir = os.path.join(self.SPICE_NETLIST_DIR, corner)
            
            self._copy_support_files(corner_dir)
            
            corner_netlist = []

            process = corner.split('_')[0]  
            
            for line in netlist_content:
                if line.startswith('.temp'):
                    corner_netlist.append(f'.temp {params[5]}\n')
                elif line.startswith('.include') and 'tt.spice' in line:
                    corner_netlist.append(f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/{process}.spice\n')
                elif line.startswith('.PARAM supply_voltage'):
                    corner_netlist.append(f'.PARAM supply_voltage = {params[6]}\n')
                else:
                    corner_netlist.append(line)
            corner_netlist.insert(1, f'* PVT Corner: {corner}\n')
            netlist_path = os.path.join(corner_dir, f'AMP_NMCF_Tran_{corner}.cir')
            with open(netlist_path, 'w') as f:
                f.writelines(corner_netlist)

    def _copy_support_files(self, corner_dir):
        
        support_files = [
        ]
        
        for file in support_files:
            src = os.path.join(self.SPICE_NETLIST_DIR, file)
            dst = os.path.join(corner_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    def get_corner_netlist_path(self, corner_idx):
        
        corner_name = list(self.pvt_corners.keys())[corner_idx]
        return os.path.join(self.SPICE_NETLIST_DIR, corner_name, f'AMP_NMCF_ACDC_{corner_name}.cir')

    def update_performance_and_reward(self, corner_idx, new_performance, new_reward):
        
        current_reward = self.node_features[corner_idx, 21]
        if new_reward > current_reward:
            performance_array = np.array(list(new_performance.values()), dtype=np.float32)
            self.node_features[corner_idx, 7:21] = performance_array  
            self.node_features[corner_idx, 21] = new_reward        

    def update_performance_and_reward_r(self, corner_idx, info_dict, reward):
        
        performance_array = np.array(list(info_dict.values()), dtype=np.float32)
        self.node_features[corner_idx, 7:21] = performance_array  
        self.node_features[corner_idx, 21] = reward        

    def get_corner_name(self, idx):
        
        return list(self.pvt_corners.keys())[idx]
    
    def get_corner_idx(self, corner_name):
        
        return list(self.pvt_corners.keys()).index(corner_name)

    def get_best_corner(self):
        
        rewards = self.node_features[:, 21]
        best_idx = np.argmax(rewards)
        return best_idx, rewards[best_idx]

    def get_graph_features(self):
        
        return torch.tensor(self.node_features, dtype=torch.float32).to(self.device) 