from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    fig2 = plt.figure()
    ax2 = Axes3D(fig2, auto_add_to_figure=False)
    fig2.add_axes(ax2)
    ax2.set_xlim3d(0, 30)
    ax2.set_ylim3d(-15, 15)
    ax2.set_zlim3d(-15, 15)
    # ax.plot_surface(xx, yy, zz, cmap='Greys_r')
    ax2.set_xlabel('X(m)')
    ax2.set_ylabel('Y(m)')
    ax2.set_zlabel('Z(m)')
    plt.show()
