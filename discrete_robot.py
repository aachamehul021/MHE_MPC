import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def create_discrete_model(Ts):
    model_name = "discrete_robot"
    
    # States
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    
    # Controls
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    
    # Discrete state equations using Euler integration
    x_next = x + Ts * (v * ca.cos(theta))
    y_next = y + Ts * (v * ca.sin(theta))
    theta_next = theta + Ts * omega
    
    model = AcadosModel()
    model.x = states
    model.u = controls
    model.p = ca.SX.sym('p', 3)  # Reference (x, y, theta)
    model.disc_dyn_expr = ca.vertcat(x_next, y_next, theta_next)
    model.name = model_name
    
    return model

def create_ocp_solver(model, xref, x_0, N=20, T=6.0):
    ocp = AcadosOcp()
    ocp.model = model
    
    Ts = T / N  # Sampling time
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = T
    
    # Cost function
    Q = np.diag([10, 10, 1])   # State cost weights
    R = np.diag([1, 1])        # Control cost weights
    Q_terminal = np.diag([100, 100, 10])
    
    x = model.x
    u = model.u
    x_ref = model.p
    
    x_err = x - x_ref
    u_err = u
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = x_err.T @ Q @ x_err + u_err.T @ R @ u_err
    ocp.model.cost_expr_ext_cost_e = x_err.T @ Q_terminal @ x_err
    
    # State constraints
    ocp.constraints.lbx = np.array([-2.5, -2.5, -np.pi])
    ocp.constraints.ubx = np.array([2.5, 2.5, np.pi])
    ocp.constraints.idxbx = np.array([0, 1, 2])
    
    # Control constraints
    ocp.constraints.lbu = np.array([-2, -np.pi/2])
    ocp.constraints.ubu = np.array([2, np.pi/2])
    ocp.constraints.idxbu = np.array([0, 1])
    
    ocp.constraints.x0 = x_0  # Initial state
    ocp.parameter_values = xref  # Reference state
    
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.tol = 1e-1  
    ocp.solver_options.regularize_method = 'MIRROR'
    
    return AcadosOcpSolver(ocp, json_file='acados_ocp.json')

def visualize_trajectory_realtime(ocp_solver, x_ref, N):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    x_traj_history = []
    
    for i in range(N+1):
        x_traj = np.array([ocp_solver.get(j, "x") for j in range(N+1)])
        
        if i == 0:
            x_traj_history = [x_traj[0, :2]]
        else:
            x_traj_history.append(x_traj[min(i, N), :2])
        
        ax.clear()
        ax.plot(*zip(*x_traj_history), 'b-', label='Trajectory')
        ax.scatter(x_ref[0], x_ref[1], color='red', marker='x', s=100, label='Target')
        
        # Plot the robot as a triangle
        x, y, theta = x_traj[min(i, N)]
        triangle = np.array([[0.1, 0], [-0.1, 0.05], [-0.1, -0.05], [0.1, 0]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_triangle = triangle @ R.T + np.array([x, y])
        ax.fill(rotated_triangle[:, 0], rotated_triangle[:, 1], 'g', alpha=0.6)
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Robot Path')
        ax.legend()
        ax.grid()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    Ts = 0.3  # Sampling time
    model = create_discrete_model(Ts)
    
    x0 = np.array([0, 0, 0])  # Initial state
    xref = np.array([2, 2, np.pi/2])  # Reference state
    
    ocp_solver = create_ocp_solver(model, xref, x_0=x0, N=20, T=6.0)
    
    status = ocp_solver.solve()
    
    if status == 0:
        print("Solver converged!")
        visualize_trajectory_realtime(ocp_solver, xref, 20)
    else:
        print(f"Solver failed with status {status}")
