from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inisiasi variabel-variabel; 0 = max, 1 = median, 2 = mean
dataSource = "Alba-12"
baseDirectory = r'D:\Universitas_Gadjah_Mada\Akademis--TIF_2018\Capstone\Data\Raw' + '\\' + dataSource + '\\'
co = np.empty([6,7,3], dtype=int)
bo = np.empty([6,7,3], dtype=int)
lt = np.empty([6,7,3], dtype=int)
c3 = np.empty([6,7,3], dtype=int)
b3 = np.empty([6,7,3], dtype=int)
l3 = np.empty([6,7,3], dtype=int)
# Looping
for axis in range(7):
    for ordinate in range(6):
        csv = np.genfromtxt(baseDirectory + str(axis) + ',' + str(ordinate) + "_" + dataSource + ".csv",delimiter=',', case_sensitive=False)
        co[ordinate,axis,0] = np.max(csv[0,5:209])
        co[ordinate,axis,1] = np.median(csv[0,5:209])
        co[ordinate,axis,2] = np.mean(csv[0,5:209])
        bo[ordinate,axis,0] = np.max(csv[1,5:209])
        bo[ordinate,axis,1] = np.median(csv[1,5:209])
        bo[ordinate,axis,2] = np.mean(csv[1,5:209])
        lt[ordinate,axis,0] = np.max(csv[2,5:209])
        lt[ordinate,axis,1] = np.median(csv[2,5:209])
        lt[ordinate,axis,2] = np.mean(csv[2,5:209])
        c3[ordinate,axis,0] = np.max(csv[3,5:209])
        c3[ordinate,axis,1] = np.median(csv[3,5:209])
        c3[ordinate,axis,2] = np.mean(csv[3,5:209])
        b3[ordinate,axis,0] = np.max(csv[4,5:209])
        b3[ordinate,axis,1] = np.median(csv[4,5:209])
        b3[ordinate,axis,2] = np.mean(csv[4,5:209])
        l3[ordinate,axis,0] = np.max(csv[5,5:209])
        l3[ordinate,axis,1] = np.median(csv[5,5:209])
        l3[ordinate,axis,2] = np.mean(csv[5,5:209])
# Variabel-variabel untuk figure
fig, axs = plt.subplots(nrows=3, ncols=6)
titles = ['Candy Once', 'Beetroot Once', 'Lemon Twice', 'Candy Thrice', 'Beetroot Thrice', 'Lemon Thrice']
ylabels = ['Max', 'Median', 'Mean']
vmin = np.min(np.array([np.min(co[:,:,:1]), np.min(bo[:,:,:1]), np.min(lt[:,:,:1]), np.min(c3[:,:,:1]), np.min(b3[:,:,:1]), np.min(l3[:,:,:1])]))
vmax = np.max(np.array([np.max(co[:,:,2]), np.max(bo[:,:,2]), np.max(lt[:,:,2]), np.max(c3[:,:,2]), np.max(b3[:,:,2]), np.max(l3[:,:,2])]))
im = np.empty(3, dtype=AxesImage)
# Menampilkan figure
for i in range(3):
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