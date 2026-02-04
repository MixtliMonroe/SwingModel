import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SwingModel():
  '''
  SwingModel handles the construction and solving of the variable-length pendulum
  Methods:
    - set_g: Set the value of the gravitational constant g
    - set_r: Set how the length of the pendulum r(t) varies in time
    - solve: Use scipy's solve_ivp to solve the ode for the chosen g and r(theta, theta_dot)
  '''
  def __init__(self, g: float = 9.81):
    assert isinstance(g, (float, np.floating)), "g should  be a positive float"
    assert g > 0, "g should be a positive float"

    self.g = g
    self.r = None
    self.partialr_partialu = None
    self.partialr_partialu = None
    self.sol = None

  def set_g(self, g):
    '''
    Set the value of the gravitational constant g for the problem
    
    Arguments:
      - g: Value of g
    '''
    assert isinstance(g, (float, np.floating)), "g should  be a positive float"
    assert g > 0, "g should  be a positive float"

    self.g = g
  
  def set_r(self, r, h=1e-10):
    '''
    Set the pendulum length function r(theta, theta_dot) as a function of theta, theta_dot.
    Also evaluates partial derivatives using finite difference methods.
    
    Arguments:
      - t: Function r(theta, theta_dot)
      - h: Step size for finite difference evaluation
    '''
    assert callable(r), f"Expected function, got {type(r).__name__}"

    self.r = r

    # Finite difference (central diff)
    self.partialr_partialu = lambda u, v: (r(u + h, v) - r(u - h,v)) / (2*h)
    self.partialr_partialv = lambda u, v: (r(u, v + h) - r(u,v - h)) / (2*h)
  
  def _dqdt(self, t, q):
    '''
    Internal function for solving the ODE.
    q = [theta, theta_dot]
    
    Arguments:
      - t: Time at which to evaluate dqdt
      - q: Angle and angular velocity at which to evaluate dqdt
    '''
    theta, theta_dot = q
    return np.array([theta_dot, 
                     -(self.g*np.sin(theta) + 2*self.partialr_partialu(theta, theta_dot)*theta_dot**2)/(self.r(theta, theta_dot) + 2*self.partialr_partialv(theta, theta_dot)*theta_dot)])
  
  def solve(self, t_eval, q0, **kwargs):
    '''
    Solve the variable-length pendulum ODE using scipy's solve_ivp
    
    Arguments:
      - t_eval: Values of time for which to solve the ODE for
      - q0: Initial angle and angular velocity
      - kwargs: Any keyword arguments to pass to solve_ivp

    Returns:
      - The solve_ivp output
    '''
    assert self.r is not None, "No pendulum length function set"
    assert isinstance(q0, (list, np.ndarray)), f"Expected array, got {type(q0).__name__}"
    assert np.array(q0).shape == (2,), f"Expected shape (2,) for theta and theta_dot, got {q0.shape}"
    assert isinstance(t_eval, (list, np.ndarray)), f"Expected array, got {type(t_eval).__name__}"

    t_span = [min(t_eval), max(t_eval)]

    self.sol = solve_ivp(self._dqdt, t_span=t_span, t_eval=t_eval, y0=q0, **kwargs)

    return self.sol
  
  def animate_sol(self):
    '''
    Generates an animation of the pendulum once the solution has been generated

    Returns:
      - ani: the matplotlib animation.FuncAnimation
    '''
    fig, ax = plt.subplots()

    theta = self.sol.y[0, :]
    theta_dot = self.sol.y[1, :]
    r = np.array([self.r(theta[i], theta_dot[i]) for i in range(len(theta))])

    x = r[0] * np.sin(theta[0])
    y = r[0] * -np.cos(theta[0])

    mass, = ax.plot([x], [y], 'o', markersize=8, color="black")
    rod = ax.plot([0, x], [0, y], color="black")[0]
    ax.set(xlim=[-1.1*max(r), 1.1*max(r)], ylim=[-1.1*max(r), 1.1*max(r)])
    ax.set_aspect("equal")

    def update(frame):
      x = r[frame] * np.sin(theta[frame])
      y = r[frame] * -np.cos(theta[frame])
      mass.set_data([x], [y])
      rod.set_data([0, x], [0, y])

      return mass, rod
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(theta), interval=30)
    return ani



if __name__ == "__main__":
  # Initialise model
  Swing = SwingModel()

  # Define pendulum length as a function of time
  def r(theta, theta_dot):
    # The sign variable constrols whether the mass is up or down
    sign = np.tanh(2*theta*theta_dot)
    return 1 - sign*0.1

  Swing.set_r(r)

  # Solve using solve_ivp
  sol = Swing.solve(t_eval = np.linspace(0, 10, 1000), q0 = [0.5, 0])

  ani = Swing.animate_sol()
  plt.show()