from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inisiasi variabel-variabel; 0 = max, 1 = median, 2 = mean
# dataSource = "Alba-12"
devices = ["C20", "Evan", "Alba"]
baseDirectory = r'D:\Universitas_Gadjah_Mada\Akademis--TIF_2018\Capstone\Data\Raw' + '\\' # + dataSource + '\\'
co = np.full([6,7,2], -102, dtype=int)
bo = np.full([6,7,2], -102, dtype=int)
lt = np.full([6,7,2], -102, dtype=int)
c3 = np.full([6,7,2], -102, dtype=int)
b3 = np.full([6,7,2], -102, dtype=int)
l3 = np.full([6,7,2], -102, dtype=int)
# Looping
for device in range(3):
    for axis in range(7):
        for ordinate in range(6):
            csv4 = np.genfromtxt(baseDirectory + devices[device] + '\\' + str(axis) + ',' + str(ordinate) + "_" + devices[device] + ".csv",delimiter=',', case_sensitive=False)
            csv12 = np.genfromtxt(baseDirectory + devices[device] + '-12\\' + str(axis) + ',' + str(ordinate) + "_" + devices[device] + "-12.csv",delimiter=',', case_sensitive=False)
            co[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * co[ordinate, axis, 0], csv4[0,5:210]])
            co[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * co[ordinate, axis, 1], csv12[0,5:210]])
            bo[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * bo[ordinate, axis, 0], csv4[1,5:210]])
            bo[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * bo[ordinate, axis, 1], csv12[1,5:210]])
            lt[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * lt[ordinate, axis, 0], csv4[2,5:210]])
            lt[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * lt[ordinate, axis, 1], csv12[2,5:210]])
            c3[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * c3[ordinate, axis, 0], csv4[3,5:210]])
            c3[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * c3[ordinate, axis, 1], csv12[3,5:210]])
            b3[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * b3[ordinate, axis, 0], csv4[4,5:210]])
            b3[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * b3[ordinate, axis, 1], csv12[4,5:210]])
            l3[ordinate,axis,0] = np.max([np.ones(np.shape(csv4[0,5:210])) * l3[ordinate, axis, 0], csv4[5,5:210]])
            l3[ordinate,axis,1] = np.max([np.ones(np.shape(csv4[0,5:210])) * l3[ordinate, axis, 1], csv12[5,5:210]])
# Variabel-variabel untuk figure
fig, axs = plt.subplots(nrows=2, ncols=6)
titles = ['Candy Once', 'Beetroot Once', 'Lemon Twice', 'Candy Thrice', 'Beetroot Thrice', 'Lemon Thrice']
ylabels = ['Max -4 dBm', 'Max -12 dBm']
vmin = np.min([co[:,:,:], bo[:,:,:], lt[:,:,:], c3[:,:,:], b3[:,:,:], l3[:,:,:]])
vmax = np.max([co[:,:,:], bo[:,:,:], lt[:,:,:], b3[:,:,:], l3[:,:,:]])
im = np.empty(3, dtype=AxesImage)
# Menampilkan figure
for i in range(2):
    # Menampilkan heatmap Candy Once
    axs[i,0].imshow(co[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan heatmap Beetroot Once
    axs[i,1].imshow(bo[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan heatmap Lemon Twice
    axs[i,2].imshow(lt[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan heatmap Candy Thrice
    axs[i,3].imshow(c3[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan heatmap Beetroot Thrice
    axs[i,4].imshow(b3[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan heatmap Lemon Thrice
    im[i] = axs[i,5].imshow(l3[:,:,i], vmin=vmin, vmax=vmax, origin='lower')
    # Menampilkan ylabels dan colorbars
    axs[i,0].set_ylabel(ylabels[i], size='large')
    divider = make_axes_locatable(axs[i,5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
for i in range(6):
    axs[0,i].set_title(titles[i])
# Menampilkan figure
plt.show()