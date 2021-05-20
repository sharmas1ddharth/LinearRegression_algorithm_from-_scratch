import numpy as np


def compute_error_for_line_given_points(b, m, points):
    # initialize error at 0
    totalError = 0

    # for every point
    for i in range(len(points)):
        # get x and y value
        x = points[i, 0]
        y = points[i, 1]

        # get the difference and square it
        totalError += (y - (m * x + b)) ** 2

    # get the average
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_of_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_of_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)

    return [b, m]


def step_gradient(current_b, current_m, points, learning_rates):
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))

    new_b = current_b - (learning_rates * b_gradient)
    new_m = current_m - (learning_rates * m_gradient)

    return [new_b, new_m]


def run():
    # collect our data
    points = np.genfromtxt('data.csv', delimiter=',')

    # define our hyperparameters
    learning_rate = 0.0001

    # slope : y = mx+b
    initial_b = 0
    initial_m = 0
    num_of_iterations = 1000
    print(f"starting gradient descent at b = {initial_b}, m = {initial_m}, "
          f"error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations)

    print(f"After {num_of_iterations } b = {b}, m = {m}, error = {compute_error_for_line_given_points(b, m, points)}")


if __name__ == "__main__":
    run()
