# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:34:40 2020

@author: Daniel
"""


import numpy as np
import neo.io
import matplotlib.pyplot as plt
import scipy.signal
import os
import pandas as pd
from brian2 import *
from brian2modelfitting import *
import seaborn as sns
import pickle

plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

prefs.codegen.target = 'cython'

file_id = 'D14'

axon_files = r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
ephys_path = r"C:\Users\Daniel\repos\PatchSeq\ephys_df.csv"
out_path = r"E:\Dropbox\Dropbox\01_SST Project_daniel\spike_train_fitter\fitting_results\model_results_AeIF_round2"

ephys_file = pd.read_csv(ephys_path, index_col=0)

data_path = None

for root, dirs, files in os.walk(axon_files):
    #print(root)
    if file_id in root:
        found_root = root
        for f in files:
            if 'active' in f:
                found_file = f
                data_path = root + '\\' + found_file
                

if not data_path:
    raise ValueError("Did not find .abf file for file_id")

ljp = 16 * mV
ljp_mag = 16

def get_data_split(neo_block, dt=0.0001, seg_start = 0, seg_end = 1, seg_interval=1, trailing=0.1):
    """Extracts all necessary data from neo.Block
    Also forces sampling period dt on all arrays"""
    dt_original = neo_block.segments[0].analogsignals[0].sampling_period

    downsampling_factor = int((dt / dt_original).magnitude)

    vm = np.array([np.array(x.analogsignals[0])[::downsampling_factor,0] 
                   for x in neo_block.segments[seg_start:seg_end:seg_interval]])

    inp = np.array([np.array(x.analogsignals[1])[::downsampling_factor,0]
                    for x in neo_block.segments[seg_start:seg_end:seg_interval]])
    
    #inp = (inp / 1000) * 20
    
    inp_thrs = np.argwhere(np.diff(np.array(inp[1]) > 10) == 1)

    inp_onset = inp_thrs[0,0] - int(trailing / dt)

    inp_offset = inp_thrs[1,0] + int(trailing / dt)

    vm = vm[:,inp_onset:inp_offset] - ljp_mag

    inp = inp[:,inp_onset:inp_offset]

    ts = [np.argwhere(np.diff(np.array(x) > 0) == 1)[0::2][:,0] * dt for x in vm]

    t = np.arange(len(vm[0]))*dt
    
    #pdb.set_trace()

    return {'vm': vm,
            'ts': ts,
            'dt': dt,
            't': t,
            'ts': ts,
            'input': inp,
            'downsampling_factor': downsampling_factor}

block = neo.io.AxonIO(data_path).read_block()

rb_idx = int(ephys_file.loc[file_id]['Rheobase idx'])
data = get_data_split(block,
                      seg_start=rb_idx,
                      seg_end=rb_idx+(2*4),
                      seg_interval=2)

#data['vm'] = data['vm'][0:8:2] - ljp_mag
#data['input'] = data['input'][0:8:2]
#data['ts'] = data['ts'][0:8:2]

N = 1
C = ephys_file.loc[file_id]['Capacitance (pF)'] * pF  # Capacitance
gL = (1 / (ephys_file.loc[file_id]['Input R (MOhm)'] * 1000)) * nS  # Leak Conductance
EL = data['vm'][:,0:100].mean() * mV  # Leak Equilibrium Potential
VT = ephys_file.loc[file_id]['RS Threshold (mV)'] * mV  # Threshold Potential

for seed in range(0, 1):
    print("Current Seed: " + str(seed).zfill(2))
    # Seed is 100 + number in file ID
    seed = 100+int(file_id[-2:])+seed
    np.random.seed(seed)
    dt = 0.0001 * second
    defaultclock.dt = dt

    opt = NevergradOptimizer(method='PSO', num_workers=10000)
    #opt = SkoptOptimizer('RF')
    metric = GammaFactor(delta=10*ms, time = data['t'][-1]*second, rate_correction=True)

    model = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I)/C : volt
               dw/dt = (a*(vs-EL)-w)/tau_w : amp
               tau_w : second
               b : amp
               DeltaT : volt
               a : siemens'''

    fitter = SpikeFitter(model=model,
                         input_var='I',
                         dt=dt,
                         input=(data['input']) * pamp,
                         output=data['ts'],
                         n_samples=10000,
                         threshold='vs > 20*mV',
                         reset='vs=EL; w += b',
                         method='euler',
                         param_init={'vs': EL,
                                     'w': 0*amp})

    results, error = fitter.fit(n_rounds=50,
                                optimizer=opt,
                                metric=metric,
                                tau_w = [1*ms, 500*ms],
                                b = [0*nA, 0.15*nA],
                                DeltaT = [0.1*mV, 10*mV],
                                a = [0*nS, 10*nS])

    DeltaT = results['DeltaT']
    b = results['b']
    tau_w = results['tau_w']
    a = results['a']

    parameter_names = ['C', 'gL', 'EL', 'VT', 'DeltaT', 'b', 'tau_w', 'a', 'gamma', 'rb_idx', 'seed']
    parameters = [C, gL, EL, VT, DeltaT, b, tau_w, a, error, rb_idx, seed]
    outcome = dict(zip(parameter_names, parameters))

    # Save the result
    fname = out_path + os.sep + file_id + "_aeIF_" + str(seed) + '.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(outcome, f, pickle.HIGHEST_PROTOCOL)
    
    sweep = 0
    I = TimedArray((data['input'][sweep]) * pamp, dt = 0.0001 * second)
    
    eqs = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I(t))/C : volt
             dw/dt = (a*(vs-EL)-w)/tau_w : amp'''

    DeltaT = results['DeltaT']
    b = results['b']
    tau_w = results['tau_w']
    a = results['a']

    group = NeuronGroup(N, eqs,
                        threshold='vs > 20*mV',
                        reset='vs=EL; w += b',
                        method='euler')
    
    group.vs = EL
    #group.vd = EL
    group.w = 0*amp
    # 'rk2' method does not oscillate

    M = SpikeMonitor(group)
    S = StateMonitor(group, 'vs', record=True, when='before_thresholds')
    dt = 0.0001 * second
    defaultclock.dt = dt
    run(data['t'][-1]*second)

    # plt.figure()
    sim_v = np.array(S.vs[0])
    sim_v[sim_v > 0.02] = 0.02
    plt.plot(data['t'], data['vm'][sweep]/1000)
    plt.plot(data['t'][:-1], sim_v)
    # plt.legend(("data", "AeIF model"))
    plt.xlabel("time (ms)")
    plt.ylabel("voltage (V)")
    plt.legend(("Data", "Simulation"))
    plt.xlim(0,1.2)
    plt.show()

for idx, inp in enumerate(data['input']):
    I = TimedArray((inp) * pamp, dt = 0.0001 * second)
    
    eqs = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I(t))/C : volt
             dw/dt = (a*(vs-EL)-w)/tau_w : amp'''

    DeltaT = results['DeltaT']
    b = results['b']
    tau_w = results['tau_w']
    a = results['a']

    group = NeuronGroup(N, eqs,
                        threshold='vs > 20*mV',
                        reset='vs=EL; w += b',
                        method='euler')

    group.vs = EL
    #group.vd = EL
    group.w = 0*amp
    # 'rk2' method does not oscillate

    M = SpikeMonitor(group)
    S = StateMonitor(group, 'vs', record=True, when='before_thresholds')
    dt = 0.0001 * second
    defaultclock.dt = dt
    run(data['t'][-1]*second)

    plt.figure()
    sim_v = np.array(S.vs[0])
    sim_v[sim_v > 0.02] = 0.02
    plt.plot(data['t'], data['vm'][idx]/1000)
    plt.plot(data['t'][:-1], sim_v)
    # plt.legend(("data", "AeIF model"))
    plt.xlabel("time (ms)")
    plt.ylabel("voltage (V)")
    plt.legend(("Data", "Simulation"))
    plt.xlim(0,1.2)
    plt.show()
    plt.title()

