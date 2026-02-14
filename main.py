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
  def __init__(self, g: float = 9.81, m: float = 1):
    assert isinstance(g, (float, np.floating)), "g should  be a positive float"
    assert g > 0, "g should be a positive float"

    self.g = g
    self.m = m
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

  def set_m(self, m):
    '''
    Set the value of the mass m for the problem
    
    Arguments:
      - m: Value of m
    '''
    assert isinstance(m, (float, np.floating)), "m should  be a positive float"
    assert m > 0, "m should  be a positive float"

    self.m = m

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
      - t: Time at which to evaluate dq/dt
      - q: Angle and angular velocity at which to evaluate dqdt
    '''
    theta, theta_dot = q
    return np.array([theta_dot, 
                     -(self.g*np.sin(theta) + 2*self.partialr_partialu(theta, theta_dot)*theta_dot**2)/(self.r(theta, theta_dot) + 2*self.partialr_partialv(theta, theta_dot)*theta_dot)])
  
  def _drdt(self, q):
    '''
    Internal function for computing dr/dt.
    Can accept either scalar values [u, v] or array inputs.
    '''
    u, v = q
    
    # Handle array inputs by vectorizing
    if isinstance(u, np.ndarray):
      # Flatten arrays for computation
      u_flat = u.flatten()
      v_flat = v.flatten()
      result = np.zeros_like(u_flat, dtype=float)
      
      for i in range(len(u_flat)):
        vdot = self._dqdt(None, q=np.array([u_flat[i], v_flat[i]]))[1]
        udot = v_flat[i]
        result[i] = self.partialr_partialu(u_flat[i], v_flat[i])*udot + self.partialr_partialv(u_flat[i], v_flat[i])*vdot
      
      return result.reshape(u.shape)
    else:
      # Original scalar behavior
      vdot = self._dqdt(None, q=q)[1]
      udot = q[1]
      return self.partialr_partialu(u, v)*udot + self.partialr_partialv(u, v)*vdot

  def _d2rdt2(self, q, h=1e-4):
    '''
    Internal function for computing d2r/dt2.
    Can accept either scalar values [u, v] or array inputs.
    '''
    u, v = q
    
    # Finite difference (central diff)
    partialrdot_partialu = lambda u, v: (self._drdt(np.array([u + h, v])) - self._drdt(np.array([u - h,v]))) / (2*h)
    partialrdot_partialv = lambda u, v: (self._drdt(np.array([u, v + h])) - self._drdt(np.array([u,v - h]))) / (2*h)

    # Handle array inputs by vectorizing
    if isinstance(u, np.ndarray):
      # Flatten arrays for computation
      u_flat = u.flatten()
      v_flat = v.flatten()
      result = np.zeros_like(u_flat, dtype=float)
      
      for i in range(len(u_flat)):
        vdot = self._dqdt(None, q=np.array([u_flat[i], v_flat[i]]))[1]
        udot = v_flat[i]
        result[i] = partialrdot_partialu(u_flat[i], v_flat[i])*udot + partialrdot_partialv(u_flat[i], v_flat[i])*vdot
      
      return result.reshape(u.shape)
    else:
      # Original scalar behavior
      vdot = self._dqdt(None, q=q)[1]
      udot = q[1]
      return partialrdot_partialu(u, v)*udot + partialrdot_partialv(u, v)*vdot

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
  
  def energies(self):
    if self.sol is None:
      raise RuntimeError("No solution available. Call 'solve' before requesting energies.")
    
    theta = self.sol.y[0, :]
    theta_dot = self.sol.y[1, :]
    r = np.array([self.r(theta[i], theta_dot[i]) for i in range(len(theta))])
    rdot = self._drdt([theta, theta_dot])

    KE = (1/2)*self.m*((r*theta_dot)**2 + rdot**2)
    PE = -self.m*self.g*r*np.cos(theta)

    return KE + PE, KE, PE

  def tension(self, q=None, h=1e-4):

    if q is None: # Unspecified state, compute T for the given solution
      if self.sol is None:
        raise RuntimeError("No solution available. Call 'solve' before requesting tension.")
      
      theta = self.sol.y[0, :]
      theta_dot = self.sol.y[1, :]
      r = np.array([self.r(theta[i], theta_dot[i]) for i in range(len(theta))])
      r_ddot = self._d2rdt2([theta, theta_dot], h=h)

      return self.m*(self.g*np.cos(theta) + r*theta_dot**2 - r_ddot)
    
    else: # Compute T for the specified state(s)
      u, v = q

      # Handle array inputs by vectorizing
      if isinstance(u, np.ndarray):
        # Flatten arrays for computation
        u_flat = u.flatten()
        v_flat = v.flatten()
        result = np.zeros_like(u_flat, dtype=float)
        
        for i in range(len(u_flat)):
          result[i] = self.m*(self.g*np.cos(u_flat[i]) + self.r(u_flat[i], v_flat[i])*v_flat[i]**2 - self._d2rdt2(np.array([u_flat[i], v_flat[i]])))
        
        return result.reshape(u.shape)
      
      else: # Original scalar behavior
        return self.m*(self.g*np.cos(u) + self.r(u, v)*v**2 - self._d2rdt2(q))

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

def generate_r(L, delta, k):

  sigmoid = lambda x: 1/(1 + np.exp(-x))

  def r(u, v):
    return L - delta*sigmoid(k*u*v)

  return r

if __name__ == "__main__":
  # Initialise model
  Swing = SwingModel()
  
  # Define pendulum length as a function of time
  r = generate_r(L=1, delta=0.3, k=2)
  Swing.set_r(r)

  # Solve using solve_ivp
  sol = Swing.solve(t_eval = np.linspace(0, 18, 2000), q0 = [0.2, 0])

  if False: # Animation
    ani = Swing.animate_sol()
    plt.show()

  if False: # Phase plane plot
    theta = sol.y[0, :]
    thetadot = sol.y[1, :]
    theta_lims = (-np.max(np.abs(theta))*1.1, np.max(np.abs(theta)*1.1))
    thetadot_lims = (-np.max(np.abs(thetadot))*1.1, np.max(np.abs(thetadot)*1.1))

    N = 300
    R = [[r(U, V) for U in np.linspace(*theta_lims, N)] for V in np.linspace(*thetadot_lims, N)]
    R = np.array(R)

    plt.plot(theta, thetadot, color="black")
    plt.xlabel(r"$\theta$")
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    plt.xlim(*theta_lims)
    plt.ylabel(r"$\dot{\theta}$")
    plt.ylim(*thetadot_lims)
    plt.imshow(R, extent=[*theta_lims, *thetadot_lims], cmap="jet", aspect=.5)
    plt.show()

  if False: # Energy plot
    E, KE, PE = Swing.energies()
    T = Swing.tension()
    plt.plot(np.linspace(0, 10, 1000), E, label="Total Energy")
    plt.plot(np.linspace(0, 10, 1000), KE, label="Kinetic Energy")
    plt.plot(np.linspace(0, 10, 1000), PE, label="Gravitational Potential Energy")
    plt.plot(np.linspace(0, 10, 1000), T*1e-1, label="Rod Tension * 1e-1")
    plt.legend()
    plt.show()

  if True: # Power
    N = 300
    u = np.linspace(-np.pi, np.pi, N)
    v = np.linspace(-np.pi, np.pi, N)
    U, V = np.meshgrid(u, v)
    DRDT = Swing._drdt([U, V])
    T = Swing.tension([U, V])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(U, V, np.abs(DRDT)*T, antialiased=False)
    plt.show()