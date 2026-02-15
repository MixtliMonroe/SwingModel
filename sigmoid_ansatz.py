import numpy as np
import warnings
import matplotlib.pyplot as plt
from main import SwingModel
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import LineCollection

def generate_r(l, delta, k):

  sigmoid = lambda x: 1/(1 + np.exp(-x))

  def r(u, v):
    return l - delta*sigmoid(k*u*v)

  return r

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

if __name__ == "__main__":
  # Set params
  g = 9.8
  l = 1
  m = 1
  delta = 0.2
  k = 10

  # Initialise model and define r function
  Model = SwingModel(g=g, m=m)
  
  r = generate_r(l=l, delta=delta, k=k)
  Model.set_r(r=r)

  # Solve using solve_ivp
  sol = Model.solve(t_eval = np.arange(0, 30, 1e-3), q0 = [0.2, 0.2])
  theta = sol.y[0, :]
  theta_dot = sol.y[1, :]
  t = sol.t

  # Cut off solution when |theta| > pi
  key = np.where(np.abs(theta) <= np.pi)
  theta = theta[key]
  theta_dot = theta_dot[key]
  t = t[key]

  if False: # Animation
    ani = Model.animate_sol()
    plt.show()
  
  if False: # Plot solution on phase plane
    R = r(theta, theta_dot)

    fig, ax = plt.subplots()
    fig.tight_layout()

    # Plot solution
    line = colored_line(x=theta, y=theta_dot, c=-R, ax=ax, cmap="jet")
    cbar = fig.colorbar(line, ticks=[-l, delta-l])
    cbar.ax.set_yticklabels(["crouching", "standing"])
    ax.annotate('', xy=(theta[-1], theta_dot[-1]),
            xytext=(theta[-1]-.1, theta_dot[-1]),
            arrowprops=dict(arrowstyle='->', linewidth=1.2, color='k'),
            clip_on=False)

    # Plot separatrices
    thetadot_crit_crouch = 2*np.sqrt(g/l)
    thetadot_crit_stand =  2*np.sqrt(g/(l-delta))
    plt.plot(np.linspace(-np.pi, 0, 500), thetadot_crit_crouch*np.cos(np.linspace(-np.pi, 0, 500)/2), color="black", ls="--", label="Fixed-length separatrix")
    plt.plot(np.linspace(-np.pi, 0, 500), -thetadot_crit_stand*np.cos(np.linspace(-np.pi, 0, 500)/2), color="black", ls="--")
    plt.plot(np.linspace(0, np.pi, 500),  -thetadot_crit_crouch*np.cos(np.linspace(0, np.pi, 500)/2), color="black", ls="--")
    plt.plot(np.linspace(0, np.pi, 500),  thetadot_crit_stand*np.cos(np.linspace(0, np.pi, 500)/2),   color="black", ls="--")

    # Plot vector plot
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-1.1*3*thetadot_crit_crouch/2, 1.1*3*thetadot_crit_crouch/2, 50)
    X, Y = np.meshgrid(x, y)
    DQDT = Model._dqdt(t=None, q=[X, Y])
    ax.quiver(x, y, *DQDT, color="black", alpha=0.3, label=r"$d\mathbf{q}/dt$")

    # Set axis ticks and labels
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, np.pi], [r"$-\pi$", r"$\pi$"])
    ax.set_xlabel(r"$\theta$", fontsize=11)
    ax.xaxis.set_label_coords(1, 0.55)

    ax.set_ylim(-1.1*3*thetadot_crit_crouch/2, 1.1*3*thetadot_crit_crouch/2)
    ax.set_yticks(np.array([-1, 1])*3*thetadot_crit_crouch/2, [r"$-\frac{3\sqrt{g/\text{l}}}{2}$", r"$\frac{3\sqrt{g/\text{l}}}{2}$"])
    ax.set_ylabel(r"$\dot{\theta}$", fontsize=11)
    ax.yaxis.set_label_coords(0.56, 1)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set axis arrows
    trans_x = blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate('', xy=(1.02, 0), xycoords=trans_x,
              xytext=(0.98, 0), textcoords=trans_x,
              arrowprops=dict(arrowstyle='->', linewidth=1.2, color='k'),
              clip_on=False)
    
    trans_y = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate('', xy=(0, 1.02), xycoords=trans_y,
              xytext=(0, 0.98), textcoords=trans_y,
              arrowprops=dict(arrowstyle='->', linewidth=1.2, color='k'),
              clip_on=False)

    plt.savefig("img/phaseplane_ansatz1.png", dpi=1500, bbox_inches="tight")
    plt.show()

  if True: # Plot drdt
    fig, ax = plt.subplots()
    fig.tight_layout()

    DRDT = Model._drdt(np.array([theta, theta_dot]))

    # Set axis limits, ticks and labels
    ax.set_ylim(-1.1*max(np.abs(DRDT)), 1.1*max(np.abs(DRDT)))
    ax.set_yticks(list(range(-30,40,10)), [str(i) for i in range(-30,40,10)])
    ax.set_ylabel(r"$\dot{r}$")

    ax.set_xlim(0, max(t))
    ax.set_xticks([2,4,6,8,10], [str(i) for i in range(2,12,2)])
    ax.set_xlabel("t")
    ax.xaxis.set_label_coords(1, 0.45)

    # Move bottom x-axis to centre, passing through (0,0)
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.plot(t, DRDT, color="black")
    plt.savefig("img/drdt_ansatz1.png", dpi=1500, bbox_inches="tight")
    plt.show()

       