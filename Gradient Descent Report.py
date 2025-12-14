import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    """The objective function: x^2 + y^2"""
    return x ** 2 + y ** 2


def grad_f(x, y):
    """The gradient vector: [2x, 2y]"""
    return 2 * x, 2 * y


def run_gradient_descent():
    # Initial Parameters
    x = 3.0
    y = 4.0
    learning_rate = 0.1
    epsilon = 0.01  # Convergence threshold

    # History to store the input points for plotting
    history_x = [x]
    history_y = [y]
    history_cost = [f(x, y)]

    print(f"{'Iter':<5} | {'Point (x, y)':<20} | {'Cost':<10} | {'Grad Norm':<10}")
    print("-" * 60)

    iteration = 0
    while True:
        # Calculate the gradient and norm
        gx, gy = grad_f(x, y)
        grad_norm = np.sqrt(gx ** 2 + gy ** 2)
        cost = f(x, y)

        # Print the current state
        print(f"{iteration:<5} | ({x:.4f}, {y:.4f})      | {cost:<10.4f} | {grad_norm:<10.4f}")

        # Check for convergence
        if grad_norm < epsilon:
            print("-" * 60)
            print(f"Converged at iteration {iteration}")
            break

        # Update the position
        x = x - learning_rate * gx
        y = y - learning_rate * gy

        # Save the history for plotting
        history_x.append(x)
        history_y.append(y)
        history_cost.append(f(x, y))

        iteration += 1

    return history_x, history_y, history_cost


def plot_results(history_x, history_y, history_cost):
    # Create a grid of points for the background plots
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Initialize the figure
    fig = plt.figure(figsize=(14, 6))

    # --- Plot 1: 3D "Bowl" Surface ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    # Overlay the descent path
    ax1.plot(history_x, history_y, history_cost, color='r', marker='o',
             markersize=3, label='Gradient Descent Path')
    ax1.set_title('3D View: The "Bowl"')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Cost')
    ax1.legend()

    # --- Plot 2: 2D Geometry (Contour) ---
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=1, fontsize=8)
    # Overlay the path
    ax2.plot(history_x, history_y, color='red', marker='o', markersize=3, label='Path')

    # Add arrows to show the direction of steps
    for i in range(len(history_x) - 1):
        ax2.arrow(history_x[i], history_y[i],
                  history_x[i + 1] - history_x[i], history_y[i + 1] - history_y[i],
                  head_width=0.15, head_length=0.1, fc='r', ec='r')

    ax2.set_title('2D View: Geometry & Descent Path')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hx, hy, hc = run_gradient_descent()
    plot_results(hx, hy, hc)