import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SwingModel():
  '''
  SwingModel handles the construction and solving of the variable-length pendulum
  Methods:
    - set_g: Set the value of the gravitational constant g
    - set_r: Set how the length of the pendulum r(t) varies in time
    - solve: Use scipy's solve_ivp to solve the ode for the chosen g and r(t)
  '''
  def __init__(self, g: float = 9.81):
    assert isinstance(g, (float, np.floating)), "g should  be a positive float"
    assert g > 0, "g should  be a positive float"

    self.g = g
    self.r = None

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
    Set the pendulum length function r(t) as a function of time.
    Also evaluates r dot using finite difference methods.
    
    Arguments:
      - t: Function r(t)
      - h: Step size for finite differenc r dot evaluation
    '''
    assert callable(r), f"Expected function, got {type(r).__name__}"

    self.r = r
    self.r_dot = lambda t: (r(t+h) - r(t)) / h
  
  def _dqdt(self, t, q):
    '''
    Internal function for solving the ODE.
    q = [theta, theta_dot]
    
    Arguments:
      - t: Time at which to evaluate dqdt
      - q: Angle and angular velocity at which to evaluate dqdt
    '''
    theta, theta_dot = q
    return np.array([theta_dot, -(2*self.r_dot(t)*theta_dot + self.g*np.sin(theta))/self.r(t)])
  
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
    assert r is not None, "No pendulum length function set"
    assert isinstance(q0, (list, np.ndarray)), f"Expected array, got {type(q0).__name__}"
    assert np.array(q0).shape == (2,), f"Expected shape (2,) for theta and theta_dot, got {q0.shape}"
    assert isinstance(t_eval, (list, np.ndarray)), f"Expected array, got {type(t_eval).__name__}"

    t_span = [min(t_eval), max(t_eval)]

    return solve_ivp(self._dqdt, t_span=t_span, t_eval=t_eval, y0=q0, **kwargs)
  

if __name__ == "__main__":
  # Initialise model
  Swing = SwingModel()
  
  # Define pendulum length as a function of time
  r = lambda t: 1 + .5*np.sin(np.sqrt(Swing.g)*t)
  Swing.set_r(r)

  # Solve using solve_ivp
  sol = Swing.solve(t_eval = np.linspace(0, 10, 1000), q0 = [.1, 0])

  # Plot solution
  plt.plot(sol.t, sol.y[0, :])
  plt.xlabel("time")
  plt.ylabel("theta")
  plt.show()