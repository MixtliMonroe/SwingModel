import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

def dqdt(q, g=9.8, l=1):
  theta, thetadot = q
  return np.array([thetadot, -(g/l)*np.sin(theta)])

def solve(q0, t_eval, g=9.8, l=1):

  solve_dqdt = lambda t, q: dqdt(q, g=g, l=l)
  t_span = [min(t_eval), max(t_eval)]

  sol = solve_ivp(solve_dqdt, t_span=t_span, t_eval=t_eval, y0=q0)

  return sol.y

if __name__ == "__main__":
  g = 9.8
  l = 1
  thetadot_crit = 2*np.sqrt(g/l)

  x = np.linspace(-5*np.pi/2, 5*np.pi/2, 30)
  y = np.linspace(-2*thetadot_crit, 2*thetadot_crit, 30)
  X, Y = np.meshgrid(x, y)
  DQDT = dqdt([X, Y], g=g, l=l)

  fig, ax = plt.subplots()
  fig.tight_layout()
  ax.quiver(x, y, *DQDT, color="black", alpha=0.3, label=r"$d\mathbf{q}/dt$")

  # Closed oscillation solution
  closed_sol = solve(q0=[0, 0.5*thetadot_crit], t_eval=np.linspace(0, (2.5)*np.pi*np.sqrt(l/g), 100), g=g, l=l)
  plt.plot(*closed_sol, color="C0", label="Closed oscillatory solution")

  # Critical solution
  crit_sol = solve(q0=[1e-5-np.pi, 0], t_eval=np.linspace(0, (5.5)*np.pi*np.sqrt(l/g), 100), g=g, l=l)
  plt.plot(np.linspace(-5*np.pi/2, 5*np.pi/2, 500), thetadot_crit*np.cos(np.linspace(-5*np.pi/2, 5*np.pi/2, 500)/2), color="red", label="Separatrix")
  plt.plot(np.linspace(-5*np.pi/2, 5*np.pi/2, 500), -thetadot_crit*np.cos(np.linspace(-5*np.pi/2, 5*np.pi/2, 500)/2), color="red")

  # Open solution
  open_sol = solve(q0=[-5*np.pi/2, 1.4*thetadot_crit], t_eval=np.linspace(0, (3)*np.pi*np.sqrt(l/g), 100), g=g, l=l)
  plt.plot(open_sol[0][np.where(open_sol[0]<5*np.pi/2)], open_sol[1][np.where(open_sol[0]<5*np.pi/2)], color="C2", label="Open anti-clockwise rotating solution")

  # Plot null lines
  ax.vlines(np.array([-2*np.pi, -np.pi, np.pi, 2*np.pi]), 0, 2*thetadot_crit, color="black", ls="--", lw=1)
  ax.vlines(np.array([-2*np.pi, -np.pi, np.pi, 2*np.pi]), -2*thetadot_crit, -0.3*thetadot_crit, color="black", ls="--", lw=1)

  # Set axis ticks and labels
  ax.set_xlim(-0.1-5*np.pi/2, 0.1+5*np.pi/2)
  ax.set_xticks([-2*np.pi, -np.pi, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", r"$\pi$", r"$2\pi$"])
  ax.set_xlabel(r"$\theta$", fontsize=10)
  ax.xaxis.set_label_coords(0.99, 0.48)

  ax.set_ylim(-2*thetadot_crit, 2*thetadot_crit)
  ax.set_yticks(np.array([-2, -1, 1, 2])*thetadot_crit, ["$-2$", "$-1$", "$1$", "$2$"])
  ax.set_ylabel(r"$\frac{\dot{\theta}}{2}\sqrt{\frac{L}{g}}$", fontsize=14)
  ax.yaxis.set_label_coords(0.59, 0.96)

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

  plt.legend(loc=3, prop={'size': 9}, framealpha=0.9)
  plt.savefig("output.png", dpi=1500, bbox_inches="tight")
  plt.show()