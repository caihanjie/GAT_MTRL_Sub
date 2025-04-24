# GAT-Based Multi-Task RL for Robust Analog Sizing via PVT Variation-Aware Attention Sampling

This repository contains the implementation of the paper "GAT-Based Multi-Task RL for Robust Analog Sizing via PVT Variation-Aware Attention Sampling".

## Overview

This project presents a novel reinforcement learning approach to optimize analog circuit sizing across multiple Process, Voltage, and Temperature (PVT) variations. The key innovation is the use of Graph Attention Networks (GAT) with a specialized attention sampling mechanism that efficiently selects the most critical PVT corners during optimization.

## Features

- Multi-task reinforcement learning for analog circuit sizing
- PVT variation-aware attention sampling for critical corner selection
- Graph-based representation of circuit components and PVT corners
- Deep Deterministic Policy Gradient (DDPG) algorithm implementation
- Integration with ngspice for circuit simulation

## Requirements

The required dependencies are listed in the `environment.yml` file. You can create a conda environment using:

```bash
conda env create -f environment.yml
```

Note: You may need to install ngspice separately for circuit simulation.

## Project Structure

- `main_AMP.py`: Main execution script for the amplifier optimization
- `AMP_NMCF.py`: Definition of the amplifier environment
- `pvt_graph.py`: PVT corner graph representation and management
- `models.py`: Implementation of GAT-based actor-critic models
- `ddpg.py`: DDPG algorithm implementation
- `ckt_graphs.py`: Circuit graph representation
- `utils.py`: Utility functions for data processing and analysis
- `simulations/`: Directory for circuit simulation files
- `saved_results/`: Directory for saving optimization results
- `mosfet_model/`: MOSFET model files for circuit simulation

## Usage

To run the optimization:

```bash
python main_AMP.py
```
