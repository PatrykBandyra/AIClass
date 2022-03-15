import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import time


def create_graph(a, b, x_size, y_size, pos, show_surface=False):
    x_ax = np.linspace(-x_size, x_size)
    y_ax = np.linspace(-y_size, y_size)
    x, y = np.meshgrid(x_ax, y_ax)
    banana_fkt = a**2 - 2*a*x + x**2 + b*y**2 - 2*b*y*x**2 + b*x**4

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, banana_fkt, locator=plt.LogLocator())
    fmt = ticker.LogFormatterMathtext()
    fmt.create_dummy_axis()
    ax.clabel(cs, cs.levels, fmt=fmt)
    ax.set_title(f'Banana Function Minimum at [1,1]')
    for a in range(pos[0].size - 1):
        if pos[0][a] != 0 or pos[1][a] != 0:
            plt.plot(pos[0][a], pos[1][a], 'ro')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.plot(1, 1, 'rx')
    if show_surface:
        x = np.outer(np.linspace(-5, 5, 30), np.ones(30))
        y = x.copy().T  # transpose
        z = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # 3D function to display

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(x, y, z, edgecolor='none')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.set_title('Banana function surface')
        print('Close both graphs to continue...')
    plt.show()


def banana_function(a, b, x, y):
    return a**2 - 2*a*x + x**2 + b*y**2 - 2*b*y*x**2 + b*x**4


def gradient(a, b, x, y):
    banana_f = np.zeros((2, 1))
    banana_f[0] = -2*a + 2*x - 4*b*y*x + 4*b*x**3   # df/dx
    banana_f[1] = 2*b*y - 2*b*x**2                  # df/dy
    return banana_f


def inverted_hessian(a, b, x, y):
    h = np.zeros((2, 2))
    determinant = -8*b**2*y+24*b**2*x**2+4*b-16*b**2*x**2
    h[0][0] = 2*b
    h[1][0] = 4*b*x
    h[0][1] = 4*b*x
    h[1][1] = 12*b*x**2 - 4*b*y + 2
    h[0][0] = h[0][0] / determinant
    h[1][0] = h[1][0] / determinant
    h[0][1] = h[0][1] / determinant
    h[1][1] = h[1][1] / determinant
    return h


def steepest_descent_method(epsilon, max_iteration, start_pos_x=None, start_pos_y=None, debug=False, display_results=False):
    a = 1
    b = 100
    x_size = 10
    y_size = 10

    return_pos = [0, 0]

    if start_pos_x is None and start_pos_y is None:
        # Starting point
        pos = [np.Infinity, np.Infinity]

        while pos[0] < -5 or pos[0] > 5:
            pos[0] = int(input("Enter starting point X [-5,5]: "))
            if pos[0] < -5 or pos[0] > 5:
                print("Wrong value. X should be in range [-5,5]")
        while pos[1] < -5 or pos[1] > 5:
            pos[1] = int(input("Enter starting point Y [-5,5]: "))
            if pos[1] < -5 or pos[1] > 5:
                print("Wrong value. Y should be in range [-5,5]")
    else:
        pos = [start_pos_x, start_pos_y]

    pos_history = np.zeros((2, 50))
    pos_history[0][0] = pos[0]
    pos_history[1][0] = pos[1]

    grad = gradient(a, b, pos[0], pos[1])

    if debug:
        print('Close graph to continue...')
        create_graph(a, b, x_size, y_size, pos_history)
        print(f'Starting position: [{pos[0]}, {pos[1]}] = {banana_function(a, b, pos[0], pos[1])}')
        print(f'First gradient: [{grad[0]}, {grad[1]}]')

    old_grad = [0, 0]

    current_iteration = 0

    start_time = time.time()
    while (np.abs(grad[0]) > epsilon or np.abs(grad[1]) > epsilon) and current_iteration <= max_iteration:

        # Calculate direction
        grad = gradient(a, b, pos[0], pos[1])
        d = -1 * gradient(a, b, pos[0], pos[1])

        if debug:
            print(f'd = [{d[0]}, {d[1]}]')

        # Perform 1 dimensional search
        a = 1   # Starting step length
        a_min = 10e-9   # Minimal step length
        decrease_factor = 0.75
        stop = False

        new_pos = [0, 0]
        new_pos[0] = pos[0]
        new_pos[1] = pos[1]

        old_value = banana_function(a, b, new_pos[0], new_pos[1])

        i = 0
        while not stop:
            i += 1
            new_pos[0] = pos[0] + a * d[0]
            new_pos[1] = pos[1] + a * d[1]
            new_value = banana_function(a, b, new_pos[0], new_pos[1])

            if new_value < 0.3*old_value:   # How low must be the value at certain point to be chosen
                stop = True
            elif a < a_min:
                stop = True
            else:
                a *= decrease_factor

        if debug:
            print(f'New position: [{new_pos[0]}, {new_pos[1]}] = {new_value}')

        # Memory allocation for history
        if current_iteration >= pos_history[0].size:
            buffer = np.zeros((2, (pos_history[0].size + 100)))
            for a in range(pos_history[0].size - 1):
                buffer[0][a] = pos_history[0][a]
                buffer[1][a] = pos_history[1][a]
            pos_history = buffer
            buffer = None

        pos_history[0][current_iteration] = pos[0]
        pos_history[1][current_iteration] = pos[1]

        pos[0] = new_pos[0]
        pos[1] = new_pos[1]

        return_pos[0] = new_pos[0]
        return_pos[1] = new_pos[1]

        old_grad[0] = grad[0]
        old_grad[1] = grad[1]
        grad = gradient(a, b, pos[0], pos[1])

        # Check if the gradient is sufficiently small
        if debug and (np.abs(grad[0]) < epsilon or np.abs(grad[1]) < epsilon):
            print(f'Gradient is sufficiently small. Exiting...')

        if debug and current_iteration >= max_iteration:
            print(f'Maximum iteration {max_iteration} exceeded. Exiting...')

        current_iteration += 1
    end_time = time.time()

    end_z = banana_function(a, b, return_pos[0], return_pos[1])
    if display_results:
        create_graph(a, b, x_size, y_size, pos_history, show_surface=True)
        print(f'Ended up at: [{return_pos[0]}, {return_pos[1]}] = {end_z}')
        print('Global minimum at: [1,1] = 0')
        print(f'Execution time: {round(end_time-start_time, 3)}s')
        print("End of Steepest Descent Method")
    else:
        return end_time-start_time, end_z


def newton_method(epsilon, max_iteration, start_pos_x=None, start_pos_y=None, debug=False, display_results=False):
    a = 1
    b = 100
    x_size = 10
    y_size = 10

    return_pos = [0, 0]

    if start_pos_x is None and start_pos_y is None:
        # Starting point
        pos = [np.Infinity, np.Infinity]

        while pos[0] < -5 or pos[0] > 5:
            pos[0] = int(input("Enter starting point X [-5,5]: "))
            if pos[0] < -5 or pos[0] > 5:
                print("Wrong value. X should be in range [-5,5]")
        while pos[1] < -5 or pos[1] > 5:
            pos[1] = int(input("Enter starting point Y [-5,5]: "))
            if pos[1] < -5 or pos[1] > 5:
                print("Wrong value. Y should be in range [-5,5]")
    else:
        pos = [start_pos_x, start_pos_y]

    pos_history = np.zeros((2, 50))
    pos_history[0][0] = pos[0]
    pos_history[1][0] = pos[1]

    grad = gradient(a, b, pos[0], pos[1])

    if debug:
        print('Close graph to continue...')
        create_graph(a, b, x_size, y_size, pos_history)
        print(f'Starting position: [{pos[0]}, {pos[1]}] = {banana_function(a, b, pos[0], pos[1])}')
        print(f'First gradient: [{grad[0]}, {grad[1]}]')

    old_grad = [0, 0]

    current_iteration = 0

    start_time = time.time()
    while (np.abs(grad[0]) > epsilon or np.abs(grad[1]) > epsilon) and current_iteration <= max_iteration:

        # Calculate direction
        neg_hess = inverted_hessian(a, b, pos[0], pos[1])
        grad = gradient(a, b, pos[0], pos[1])
        d = [0, 0]
        d[0] = -1 * (neg_hess[0][0]*grad[0] + neg_hess[0][1]*grad[1])
        d[1] = -1 * (neg_hess[1][0]*grad[0] + neg_hess[1][1]*grad[1])

        if debug:
            print(f'd = [{d[0]}, {d[1]}]')

        # Perform 1 dimensional search
        a = 1   # Starting step length
        a_min = 10e-9   # Minimal step length
        decrease_factor = 0.75
        stop = False

        new_pos = [0, 0]
        new_pos[0] = pos[0]
        new_pos[1] = pos[1]

        old_value = banana_function(a, b, new_pos[0], new_pos[1])

        i = 0
        while not stop:
            i += 1
            new_pos[0] = pos[0] + a * d[0]
            new_pos[1] = pos[1] + a * d[1]
            new_value = banana_function(a, b, new_pos[0], new_pos[1])

            if new_value < 0.3*old_value:   # How low must be the value at certain point to be chosen
                stop = True
            elif a < a_min:
                stop = True
            else:
                a *= decrease_factor

        if debug:
            print(f'New position: [{new_pos[0]}, {new_pos[1]}] = {new_value}')

        # Memory allocation for history
        if current_iteration >= pos_history[0].size:
            buffer = np.zeros((2, (pos_history[0].size + 100)))
            for a in range(pos_history[0].size - 1):
                buffer[0][a] = pos_history[0][a]
                buffer[1][a] = pos_history[1][a]
            pos_history = buffer
            buffer = None

        pos_history[0][current_iteration] = pos[0]
        pos_history[1][current_iteration] = pos[1]

        pos[0] = new_pos[0]
        pos[1] = new_pos[1]

        return_pos[0] = new_pos[0]
        return_pos[1] = new_pos[1]

        old_grad[0] = grad[0]
        old_grad[1] = grad[1]
        grad = gradient(a, b, pos[0], pos[1])

        # Check if the gradient is sufficiently small
        if debug and (np.abs(grad[0]) < epsilon or np.abs(grad[1]) < epsilon):
            print(f'Gradient is sufficiently small. Exiting...')

        if debug and current_iteration >= max_iteration:
            print(f'Maximum iteration {max_iteration} exceeded. Exiting...')

        current_iteration += 1
    end_time = time.time()

    end_z = banana_function(a, b, return_pos[0], return_pos[1])
    if display_results:
        create_graph(a, b, x_size, y_size, pos_history, show_surface=True)
        print(f'Ended up at: [{return_pos[0]}, {return_pos[1]}] = {end_z}')
        print('Global minimum at: [1,1] = 0')
        print(f'Execution time: {round(end_time-start_time, 3)}s')
        print("End of Newton Method")
    else:
        return end_time-start_time, end_z


def time_iteration_impact(x, y):
    epsilon = 10e-3
    start_pos = [x, y]
    iterations = [500, 1000, 2000, 5000, 10000]
    times_steepest = []
    e_steepest = []
    times_newton = []
    e_newton = []
    print('Timing execution time and efficiency (z coordinate; closer to 0 = better) of search depending on number of iterations...')
    for i in iterations:
        t1, e1 = steepest_descent_method(epsilon, i, start_pos[0], start_pos[1], debug=False, display_results=False)
        times_steepest.append(t1)
        e_steepest.append(e1)
        print(f'Timed steepest descent for iteration {i}')
        t2, e2 = newton_method(epsilon, i, start_pos[0], start_pos[1], debug=False, display_results=False)
        times_newton.append(t2)
        e_newton.append(e2)
        print(f'Timed Newton method for iteration {i}')
    print('Creating graphs...')

    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle('Execution time and efficiency of search depending on number of iterations\nSteepest Descent Method')
    ax1.plot(iterations, times_steepest)
    ax2.plot(iterations, e_steepest)
    plt.xlabel("Iterations number")
    ax1.set(ylabel='Time [s]')
    ax2.set(ylabel='Efficiency [z coord.]')

    fig2, (ax3, ax4) = plt.subplots(2)
    fig2.suptitle('Execution time and efficiency of search depending on number of iterations\nNewton Method')
    ax3.plot(iterations, times_newton)
    ax4.plot(iterations, e_steepest)
    plt.xlabel("Iterations number")
    ax3.set(ylabel='Time [s]')
    ax4.set(ylabel='Efficiency [z coord.]')
    plt.show()


# def time_epsilon_impact(x, y):
#     start_pos = [x, y]
#     iterations = 10000
#     epsilons = [10e-3, 50e-3, 10e-4, 50e-4, 10e-5]
#     times_steepest = []
#     e_steepest = []
#     times_newton = []
#     e_newton = []
#     print('Timing execution time and efficiency (z coordinate; closer to 0 = better) of search depending on epsilon value...')
#     for e in epsilons:
#         t1, e1 = steepest_descent_method(e, iterations, start_pos[0], start_pos[1], debug=False, display_results=False)
#         times_steepest.append(t1)
#         e_steepest.append(e1)
#         print(f'Timed steepest descent for epsilon {e}')
#         t2, e2 = newton_method(e, iterations, start_pos[0], start_pos[1], debug=False, display_results=False)
#         times_newton.append(t2)
#         e_newton.append(e2)
#         print(f'Timed Newton method for epsilon {e}')
#     print('Creating graphs...')
#
#     fig1, (ax1, ax2) = plt.subplots(2)
#     fig1.suptitle('Execution time and efficiency of search depending on epsilon value\nSteepest Descent Method')
#     ax1.plot(epsilons, times_steepest)
#     ax2.plot(epsilons, e_steepest)
#     plt.xlabel("Epsilon value")
#     ax1.set(ylabel='Time [s]')
#     ax2.set(ylabel='Efficiency [z coord.]')
#
#     fig2, (ax3, ax4) = plt.subplots(2)
#     fig2.suptitle('Execution time and efficiency of search depending on epsilon value\nNewton Method')
#     ax3.plot(epsilons, times_newton)
#     ax4.plot(epsilons, e_steepest)
#     plt.xlabel("Iterations number")
#     ax3.set(ylabel='Time [s]')
#     ax4.set(ylabel='Efficiency [z coord.]')
#     plt.show()


if __name__ == '__main__':
    #newton_method(10e-3, 1000, debug=True, display_results=True)
    steepest_descent_method(10e-3, 1000, debug=True, display_results=True)
    # time_iteration_impact(3, 3)
