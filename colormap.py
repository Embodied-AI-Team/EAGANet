import numpy as np
import proplot as pplt
import matplotlib as mpl

# x = np.linspace(0, 2*np.pi, endpoint=True, num=20)
# y = 50*np.sin(x)
# cmap = mpl.cm.get_cmap('jet')
# norm = mpl.colors.Normalize(vmin=-50, vmax=100)
# colors = cmap(norm(y))

# fig, ax = pplt.subplots()
# ax.scatter(x, y, c=colors)
# fig.colorbar(cmap, values=np.linspace(-80, 100, endpoint=False, num=80),align='top')
# fig.save('colorbar.png',bbox_inches='tight')

import numpy as np
import proplot as pplt
import matplotlib as mpl
import matplotlib.pyplot as plt # import matplotlib

# x = np.linspace(0, 2*np.pi, endpoint=True, num=20)
# y = 50*np.sin(x)
# cmap = mpl.cm.get_cmap('jet')
# norm = mpl.colors.Normalize(vmin=-50, vmax=100)
# colors = cmap(norm(y))

# #fig, ax = pplt.subplots()
# fig = plt.figure(figsize = (8,4))
# ax = fig.add_axes([0.1, 0.3, 0.8, 0.6]) # [left, bottom, width, height]
# ax.scatter(x, y, c=colors)

# # fig.colorbar(cmap, values=np.linspace(-80, 100, endpoint=False, num=80),align='top',orientation='horizontal') # wrong way
# cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.1]) #  [left, bottom, width, height] for horizontal colorbar
# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                     cax = cbar_ax,
#                      values=np.linspace(-80, 100, endpoint=False, num=80),
#                      orientation='horizontal') # colorbar is placed at [0.1, 0.1, 0.8, 0.1]

# # cbar.ax.tick_params(axis='x', labelsize=8) # change the colorbar tick size.
# fig.savefig('colorbar_horizontal.png', bbox_inches='tight')

import numpy as np
import proplot as pplt
import matplotlib as mpl
import matplotlib.pyplot as plt # import matplotlib

x = np.linspace(0, 2*np.pi, endpoint=True, num=20)
y = 50*np.sin(x)
cmap = mpl.cm.get_cmap('jet')
# norm = mpl.colors.Normalize(vmin=-50, vmax=100) # old norm
norm = mpl.colors.Normalize(vmin=0, vmax=1) # new norm
# norm = mpl.colors.Normalize(vmin=-1, vmax=1) # try to use vmin/max as -1 to 1
colors = cmap(norm(y/100 + 0.5))  # rescale data to [0, 1]

fig = plt.figure(figsize = (8,4))
ax = fig.add_axes([0.1, 0.3, 0.8, 0.6]) # [left, bottom, width, height]
ax.scatter(x, y, c=colors)

cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.1]) #  [left, bottom, width, height] for horizontal colorbar
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax = cbar_ax,
                     values=np.linspace(0, 1, endpoint=False, num=100), # change the value.
                     orientation='horizontal')
fig.savefig('colorbar_horizontal_0_to_1.png', bbox_inches='tight')