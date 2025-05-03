import h5py
import os
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 
from pylab import *
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'serif'

success_metric = 5e-3
theme = 'PuRd'

current_working_directory = os.getcwd()
parent_folder_path = os.path.dirname(current_working_directory)
target_folder_name = 'results'
target_file_name = 'resultFigure4.mat'
target_folder_path = os.path.join(parent_folder_path, target_folder_name, target_file_name)

RSE_data = np.array(scio.loadmat('%s' % (target_folder_path))['RSE1'])
RSE_data = np.transpose(RSE_data[:, :, :, :], [0, 2, 1, 3])

# creating figures
fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
data = np.squeeze(np.sum(np.squeeze(RSE_data) < success_metric, 3))

indices_linear = np.argwhere(data > -1)
data_linear = []
for index in list(indices_linear):
    data_linear.append(data[tuple(index)])
data_linear = np.array(data_linear)

# setting color bar 
color_map = matplotlib.colormaps.get_cmap(theme)

# creating the heatmap 
square = MarkerStyle(marker='s')

for x, y, z, v in zip(indices_linear[:, 0], indices_linear[:, 1], indices_linear[:, 2], data_linear):
    r, g, b, a = color_map(int((color_map.N - 1) * v / RSE_data.shape[3]))
    ax.scatter3D(x, y, z, c=[r, g, b], marker=square, s=180, alpha=0.01 + 0.99 * v / RSE_data.shape[3], linewidths=2, edgecolor='face') 

ax.view_init(elev=15, azim=136)
ax.grid(0)
ax.set_facecolor('white')

# adding title and labels 
ax.set_xlabel('SNR (dB)', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.xaxis.set_tick_params(pad=-4.0)
ax.set_ylabel('Sparsity', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.yaxis.set_tick_params(pad=-4.0)
ax.set_zlabel('Rank', fontsize=24, rotation='vertical', fontname='Times New Roman')
ax.zaxis.set_tick_params(pad=1.0)
pos_x_tick = np.arange(0, data.shape[0], 2)
ax.set_xticks(pos_x_tick)
ax.set_xticklabels([f'{val:d}' for val in np.arange(1, 12, 2)], fontsize=14)
pos_y_tick = np.arange(0, data.shape[1], 4)
ax.set_yticks(pos_y_tick)
ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0.01, 0.20, 0.0316)], fontsize=14)
pos_z_tick = np.arange(0, data.shape[2], 4)
ax.set_zticks(pos_z_tick)
ax.set_zticklabels([f'{val:d}' for val in np.arange(1, 34, 4)], fontsize=14)
# Normalizer 
norm = mpl.colors.Normalize(vmin=0, vmax=1) 

# creating ScalarMappable 
sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm) 
sm.set_array([]) 
  
cb = plt.colorbar(sm, shrink=1, ticks=np.linspace(0, 1, 5), cax=ax.inset_axes([-0.04, 0.18, 0.03, 0.57]))
cb.ax.tick_params(labelsize=14)
plt.tight_layout()

save_folder_path = os.path.join(parent_folder_path, target_folder_name, 'Figure4(a).jpg')
plt.savefig(save_folder_path, dpi=600)

RSE_data = np.array(scio.loadmat('%s' % (target_folder_path))['RSE2'])
RSE_data = np.transpose(RSE_data[:, :, :, :], [0, 2, 1, 3])

# creating figures
fig = plt.figure(2, figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

data = np.squeeze(np.sum(np.squeeze(RSE_data) < success_metric, 3))

indices_linear = np.argwhere(data > -1)
data_linear = []
for index in list(indices_linear):
    data_linear.append(data[tuple(index)])
data_linear = np.array(data_linear)

# setting color bar
color_map = matplotlib.cm.get_cmap(theme)

# creating the heatmap
square = MarkerStyle(marker='s')

for x, y, z, v in zip(indices_linear[:, 0], indices_linear[:, 1], indices_linear[:, 2], data_linear):
    r, g, b, a = color_map(int((color_map.N - 1) * v / RSE_data.shape[3]))
    ax.scatter3D(x, y, z, c=[r, g, b], marker=square, s=180, alpha=0.01 + 0.99 * v / RSE_data.shape[3], linewidths=2, edgecolor='face')

ax.view_init(elev=15, azim=136)
ax.grid(0)
ax.set_facecolor('white')

# adding title and labels
ax.set_xlabel('SNR (dB)', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.xaxis.set_tick_params(pad=-4.0)
ax.set_ylabel('Sparsity', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.yaxis.set_tick_params(pad=-4.0)
ax.set_zlabel('Rank', fontsize=24, rotation='vertical', fontname='Times New Roman')
ax.zaxis.set_tick_params(pad=1.0)
pos_x_tick = np.arange(0, data.shape[0], 2)
ax.set_xticks(pos_x_tick)
ax.set_xticklabels([f'{val:d}' for val in np.arange(1, 12, 2)], fontsize=14)
pos_y_tick = np.arange(0, data.shape[1], 4)
ax.set_yticks(pos_y_tick)
ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0.01, 0.2, 0.0316)], fontsize=14)
pos_z_tick = np.arange(0, data.shape[2], 4)
ax.set_zticks(pos_z_tick)
ax.set_zticklabels([f'{val:d}' for val in np.arange(1, 34, 4)], fontsize=14)
# Normalizer 
norm = mpl.colors.Normalize(vmin=0, vmax=1) 

sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
sm.set_array([])

cb = plt.colorbar(sm, shrink=1, ticks=np.linspace(0, 1, 5), cax=ax.inset_axes([-0.04, 0.18, 0.03, 0.57]))
cb.ax.tick_params(labelsize=14)
plt.tight_layout()

save_folder_path = os.path.join(parent_folder_path, target_folder_name, 'Figure4(b).jpg')
plt.savefig(save_folder_path, dpi=600)

RSE_data = np.array(scio.loadmat('%s' % (target_folder_path))['RSE3'])
RSE_data = np.transpose(RSE_data[:, :, :, :], [0, 2, 1, 3])

# creating figures
fig = plt.figure(3, figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

data = np.squeeze(np.sum(np.squeeze(RSE_data) < success_metric, 3))


indices_linear = np.argwhere(data > -1)
data_linear = []
for index in list(indices_linear):
    data_linear.append(data[tuple(index)])
data_linear = np.array(data_linear)

# setting color bar
color_map = matplotlib.cm.get_cmap(theme)

# creating the heatmap
square = MarkerStyle(marker='s')

for x, y, z, v in zip(indices_linear[:, 0], indices_linear[:, 1], indices_linear[:, 2], data_linear):
    r, g, b, a = color_map(int((color_map.N - 1) * v / RSE_data.shape[3]))
    ax.scatter3D(x, y, z, c=[r, g, b], marker=square, s=180, alpha=0.01 + 0.99 * v / RSE_data.shape[3], linewidths=2, edgecolor='face')

ax.view_init(elev=15, azim=136)
ax.grid(0)
ax.set_facecolor('white')

# adding title and labels
ax.set_xlabel('SNR (dB)', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.xaxis.set_tick_params(pad=-4.0)
ax.set_ylabel('Sparsity', fontsize=24, rotation='horizontal', fontname='Times New Roman')
ax.yaxis.set_tick_params(pad=-4.0)
ax.set_zlabel('Rank', fontsize=24, rotation='vertical', fontname='Times New Roman')
ax.zaxis.set_tick_params(pad=1.0)
pos_x_tick = np.arange(0, data.shape[0], 2)
ax.set_xticks(pos_x_tick)
ax.set_xticklabels([f'{val:d}' for val in np.arange(1, 12, 2)], fontsize=14)
pos_y_tick = np.arange(0, data.shape[1], 4)
ax.set_yticks(pos_y_tick)
ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0.01, 0.2, 0.0316)], fontsize=14)
pos_z_tick = np.arange(0, data.shape[2], 4)
ax.set_zticks(pos_z_tick)
ax.set_zticklabels([f'{val:d}' for val in np.arange(1, 34, 4)], fontsize=14)
# Normalizer 
norm = mpl.colors.Normalize(vmin=0, vmax=1) 

sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
sm.set_array([])

cb = plt.colorbar(sm, shrink=1, ticks=np.linspace(0, 1, 5), cax=ax.inset_axes([-0.04, 0.18, 0.03, 0.57]))
cb.ax.tick_params(labelsize=14)
plt.tight_layout()

save_folder_path = os.path.join(parent_folder_path, target_folder_name, 'Figure4(c).jpg')
plt.savefig(save_folder_path, dpi=600)