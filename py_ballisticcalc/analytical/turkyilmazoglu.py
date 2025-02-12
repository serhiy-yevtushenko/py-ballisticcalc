import cmath
import math
from functools import partial

from py_ballisticcalc.analytical.common_analytic_functions import (
    compute_x_from_time,
    f_from_angle,
    flight_distance_from_apex_height,
    flight_time_from_apex_height,
    x_from_apex_height_distance_angle_and_y,
)
#from scipy.optimize import brenth, minimize_scalar

from py_ballisticcalc.analytical.drag_free_ballistics import EARTH_GRAVITY_CONSTANT


def get_dimensionless_velocity(velocity, initial_velocity):
    return velocity / initial_velocity

def get_dimensionless_time(time, initial_velocity, g=EARTH_GRAVITY_CONSTANT):
    return time * g / initial_velocity

def from_dimensionless_tau_to_time(tau, initial_velocity, g=EARTH_GRAVITY_CONSTANT):
    return tau * initial_velocity / g

def get_dimensionless_X(x, initial_velocity, g=EARTH_GRAVITY_CONSTANT):
    return x * g / (initial_velocity**2)


def from_X_to_x(X, initial_velocity, g=EARTH_GRAVITY_CONSTANT):
    return (initial_velocity**2) * X / g


def get_dimensionless_Y(y, initial_velocity, g=EARTH_GRAVITY_CONSTANT):
    return y * g / (initial_velocity**2)


def from_Y_to_y(Y, initial_velocity, g):
    return (initial_velocity**2) * Y / g


def get_dimensionless_U(velocity, initial_velocity):
    return velocity / initial_velocity


def from_U_to_velocity(U, initial_velocity):
    return U * initial_velocity

def compute_apex_height_dimensionless(p, theta_0_in_degrees):
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    print(f"{cos_theta_squared=}")
    k_turk = p * cos_theta_squared

    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)
    print(f"{alpha=} {beta=}")

    print(f"C={k_turk=}")
    D_1 = -alpha * k_turk
    D_2 = -beta * k_turk
    print(f"{f_from_angle(theta_0_in_degrees)=}")
    D_3 = 1 + k_turk * f_from_angle(theta_0_in_degrees)
    disrc = (D_2**2) - (4 * D_1 * D_3)
    print(f"{D_2**2=} {(4*D_1*D_3)=}")
    print(f"D_1={D_1} D_2={D_2} D_3 ={D_3} D_2**2-4*D_1*D_3={disrc=}")

    A_1_denominator = (
        -k_turk * (alpha**2)
        - 4 * beta
        - 4
        * k_turk
        * beta
        * f_from_angle_approx_turkyilmazoglu(theta_0_in_degrees, theta_0_in_degrees)
    )
    A_1 = k_turk / (A_1_denominator)
    A_2 = alpha + 2 * beta * math.tan(theta_0_in_radians)
    A_L = 1 + k_turk * f_from_angle_approx_turkyilmazoglu(
        theta_0_in_degrees, theta_0_in_degrees
    )
    denominator = 2 * k_turk * beta
    print(f"{A_1=} {A_1_denominator=} {A_2=} {A_L=}")
    print(
        f"{f_from_angle_approx_turkyilmazoglu(theta_0_in_degrees, theta_0_in_degrees)=} {denominator=}"
    )
    # math.log(A_3/(A_3+A_4))
    nominator = (cmath.cos(theta_0_in_radians) ** 2) * (
        2
        * cmath.sqrt(A_1)
        * alpha
        * cmath.atan(
            cmath.sqrt(A_1) * (2 * beta * cmath.tan(theta_0_in_radians))/
            (1 + A_1 * A_2 * alpha)
        )
        + cmath.log(A_L)
    )
    return (nominator / denominator).real


def compute_apex_height(
    initial_velocity, k, theta_0_in_degrees, g=EARTH_GRAVITY_CONSTANT
):
    # return from_Y_to_y(
    #     compute_apex_height_dimensionless(compute_p(initial_velocity, k),
    #                                       theta_0_in_degrees),
    #     initial_velocity, g
    # )
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    D = (initial_velocity**2) * (cos_theta_squared)

    print(f"{D=} {theta_0_in_degrees=} {theta_0_in_radians=}")
    A, B, C, delta = compute_A_B_C_delta(k, D, theta_0_in_degrees)
    print(
        f"compute_apex_height B={B} A={A} C ={C} {(B**2)-(4*A*C)=} {delta=} {4*A*C=} {B**2=}"
    )
    if delta > 0:
        D_1 = -D / g
        # print(f'{D_1=}')
        height_theta_0 = compute_Y_from_angle(theta_0_in_radians, A, B, C, delta, D_1)
        height_zero = compute_Y_from_angle(0, A, B, C, delta, D_1)
        print(f"{height_zero=} {height_theta_0=}")
        result = height_zero - height_theta_0
        print(f"compute_apex_height RETURN {theta_0_in_degrees=} {result=}\n")
        return result
    else:
        raise ValueError("Not yet implemented")


def compute_A_B_C_delta(k, D, theta_0_in_degrees):
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)
    B = -alpha * k * D
    A = -beta * k * D
    C = 1 + k * D * f_from_angle(theta_0_in_degrees)
    delta = (B ** 2) - (4 * A * C)
    return A, B, C, delta


def compute_Y_from_angle(theta_0_in_radians, A, B, C, delta, D_1):
    tan_theta_0 = math.tan(theta_0_in_radians)
    delta_sqrt = math.sqrt(delta)
    log_first_term = math.log(abs(A * (tan_theta_0**2) + B * tan_theta_0 + C))
    print(f'{A=} {B=} {C=} {delta=}')
    print(f'{2*A=} {B+delta_sqrt=} {B-delta_sqrt=}')
    global_multiplier = D_1 / (2 * A)
    print(f'{global_multiplier=} {D_1/(math.cos(theta_0_in_radians)**2)=}')
    two_A_tan_plus_B = 2 * A * tan_theta_0 + B
    log_two_abs_nominator = abs(two_A_tan_plus_B + delta_sqrt)
    log_two_abs_denominator = abs(two_A_tan_plus_B - delta_sqrt)
    second_log_term = math.log(log_two_abs_nominator / log_two_abs_denominator)/delta_sqrt
    second_log_term_multiplier = B
    # print(f'{log_first_term=} {global_multiplier=} {log_two_abs_nominator=} {log_two_abs_denominator=}')
    # print(f"{log_two_abs_nominator/log_two_abs_denominator=} {second_log_term=} {second_log_term_multiplier=} {global_multiplier*second_log_term_multiplier=}")
    return global_multiplier * (
        log_first_term + second_log_term_multiplier * second_log_term
    )


def compute_apex_distance_dimensionless(p, theta_0_in_degrees):
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    k_turk = p * cos_theta_squared
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)
    B = -alpha * k_turk
    A = -beta * k_turk
    C = 1 + k_turk * f_from_angle(theta_0_in_degrees)
    delta = (B**2) - (4 * A * C)
    print(
        f"compute_apex_distance_dimensionless B={B} A={A} C ={C} {(B**2)-(4*A*C)=} {delta=}"
    )

    A_1_denominator = (
        -k_turk * (alpha**2)
        - 4 * beta
        - 4 * k_turk * beta * f_from_angle(theta_0_in_degrees)
    )
    A_1 = k_turk / (A_1_denominator)
    A_2 = alpha + 2 * beta * math.tan(theta_0_in_radians)
    print(f"{A_1=} {A_2=}")

    atan_argument = cmath.sqrt(A_1) * (alpha - A_2) / (1 + A_1 * A_2 * alpha)
    atan_multiplier = 2 * cmath.sqrt(A_1) * (cos_theta_squared) / k_turk
    X_a_c = atan_multiplier * cmath.atan(atan_argument)
    return X_a_c.real


def compute_apex_distance(
    initial_velocity, k, theta_0_in_degrees, g=EARTH_GRAVITY_CONSTANT
):
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    D = (initial_velocity**2) * (cos_theta_squared)
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)

    B = -alpha * k * D
    A = -beta * k * D
    C = 1 + k * D * f_from_angle(theta_0_in_degrees)
    delta = (B**2) - (4 * A * C)

    print(
        f"compute_apex_distance B={B} A={A} C ={C} {(B**2)-(4*A*C)=} {delta=} {4*A*C=} {B**2=}"
    )

    B_minus_root_delta = B - math.sqrt(delta)
    log_numerator = 2 * A * math.tan(theta_0_in_radians) + B_minus_root_delta
    B_plus_root_delta = B + math.sqrt(delta)
    log_denumerator = 2 * A * math.tan(theta_0_in_radians) + B_plus_root_delta
    log_term_theta = math.log(abs(log_numerator) / abs(log_denumerator))

    multiplier = -D / (g * math.sqrt(delta))
    X_a_c_theta = multiplier * log_term_theta

    log_term_zero = math.log(
        abs(2 * A * math.tan(0) + B_minus_root_delta)
        / abs(2 * A * math.tan(0) + B_plus_root_delta)
    )
    X_a_c_0 = multiplier * log_term_zero

    return X_a_c_0 - X_a_c_theta


def compute_time_of_ascent_dimensionless(p, theta_0_in_degrees: float):
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    k_turk = p * cos_theta_squared
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)
    C = 1 + k_turk * f_from_angle(theta_0_in_degrees)

    A_1_denominator = (
        -k_turk * (alpha**2)
        - 4 * beta
        - 4
        * k_turk
        * beta
        * f_from_angle(theta_0_in_degrees)
    )
    A_1 = k_turk / (A_1_denominator)
    A_2 = alpha + 2 * beta * cmath.tan(theta_0_in_radians)
    print(f"{A_1=} {A_1_denominator=} {A_2=}")

    tau_a = cmath.cos(theta_0_in_radians)*cmath.log((A_2*cmath.sqrt(k_turk)-2*cmath.sqrt(-beta))/
                                                    (alpha*cmath.sqrt(k_turk)-2*cmath.sqrt(-C*beta)))/(cmath.sqrt(-k_turk*beta))
    return tau_a.real



def compute_time_of_ascent(
    initial_velocity, k, theta_0_in_degrees, g=EARTH_GRAVITY_CONSTANT
):
    theta_0_in_radians = math.radians(theta_0_in_degrees)

    cos_theta_squared = math.cos(theta_0_in_radians) ** 2
    D = (initial_velocity**2) * (cos_theta_squared)
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)

    print(f"{D=}")

    B = -alpha * k * D
    A = -beta * k * D
    C = 1 + k * D * f_from_angle(theta_0_in_degrees)
    delta = (B**2) - (4 * A * C)
    print(
        f"compute_time_of_ascent B={B} A={A} C ={C} {(B**2)-(4*A*C)=} {delta=} {4*A*C=} {B**2=}"
    )

    # D_1 = -math.sqrt(D)/g
    D_1 = -initial_velocity * math.cos(theta_0_in_radians) / g
    print(f"{D_1=}")
    # D_1 = D

    sqrt_delta = math.sqrt(delta)
    print(f"{sqrt_delta=}")

    tan_theta_0 = math.tan(theta_0_in_radians)

    T_theta_0 = (
        D_1 / (math.sqrt(-A)) * math.asin((-2 * A * tan_theta_0 - B) / sqrt_delta)
    )
    T_zero = D_1 / (math.sqrt(-A)) * math.asin((-2 * A * math.tan(0) - B) / sqrt_delta)
    print(f"{T_theta_0=} {T_zero=}")

    return T_zero - T_theta_0

def f_from_angle_approx_turkyilmazoglu(angle_in_degrees, theta_0_in_degrees):
    alpha, beta = compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees)

    angle_in_radians = math.radians(angle_in_degrees)
    return alpha * math.tan(angle_in_radians) + beta * (math.tan(angle_in_radians) ** 2)

def compute_alpha_beta_turkyilmazoglu(theta_0_in_degrees):
    theta_0_in_radians = math.radians(theta_0_in_degrees)
    log_term = math.log(1 / math.tan(theta_0_in_radians / 2 + math.pi / 4))
    # print(f'compute_alpha_beta_turkyilmazoglu {log_term=}')
    tan_theta_0 = math.tan(theta_0_in_radians)
    alpha = -2 * log_term / tan_theta_0
    beta = 1 / math.sin(theta_0_in_radians) + log_term / (tan_theta_0**2)
    return alpha, beta


def determine_flight_time(V_0, k, theta_0_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    return flight_time_from_apex_height(compute_apex_height(V_0, k, theta_0_in_degrees, g))

def compute_flight_distance(V_0, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    apex_height = compute_apex_height(V_0, k, angle_in_degrees, g)
    return flight_distance_from_apex_height(
        V_0,
        k,
        angle_in_degrees,
        apex_height,
        g,
    )

def get_x_from_t(initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    L = compute_flight_distance(initial_velocity, k, angle_in_degrees, g)
    x_apex = compute_apex_distance(initial_velocity, k, angle_in_degrees, g)
    t_apex = compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    time_of_flight = determine_flight_time(initial_velocity, k, angle_in_degrees, g)
    return partial(
        compute_x_from_time,
        L=L,
        x_apex=x_apex,
        t_apex=t_apex,
        time_of_flight=time_of_flight,
    )


def compute_y_from_time(time, H, time_apex, time_of_flight):
    return (H * time * (time_of_flight - time)) / (
        time_apex**2 + (time_of_flight - 2 * time_apex) * time
    )


def get_y_from_t(initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    H = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    time_apex = compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    time_flight = determine_flight_time(initial_velocity, k, angle_in_degrees, g)
    y_from_t = partial(
        compute_y_from_time, H=H, time_apex=time_apex, time_of_flight=time_flight
    )
    return y_from_t


def compute_trajectory(
    initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT, time_step=0.01
):
    time_of_flight = determine_flight_time(initial_velocity, k, angle_in_degrees, g)
    x_from_t = get_x_from_t(initial_velocity, k, angle_in_degrees, g)
    y_from_t = get_y_from_t(initial_velocity, k, angle_in_degrees, g)
    t = 0
    points = []
    while t <= time_of_flight:
        x = x_from_t(t)
        y = y_from_t(t)
        #    points.append(((x, y), t))
        points.append((x, y))
        t += time_step
    if t != time_of_flight:
        points.append((x_from_t(time_of_flight), y_from_t(time_of_flight)))
    return points


def x_from_velocity_angle_and_y(initial_velocity, k, angle_in_degrees, y, g):
    H = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    x_apex = compute_apex_distance(initial_velocity, k, angle_in_degrees)
    return x_from_apex_height_distance_angle_and_y(initial_velocity, k, angle_in_degrees, y, H, x_apex, g)

#
# def compute_max_range_launch_angle(initial_velocity, k, g=EARTH_GRAVITY_CONSTANT):
#     p = k * (initial_velocity**2)
#
#     def lambda_from_angle(angle_in_radians):
#         return math.log(math.tan(angle_in_radians / 2 + math.pi / 4))
#
#     def alpha_from_angle(angle_in_radians, p):
#         equation_left_part = math.tan(angle_in_radians) ** 2 + p * math.sin(
#             angle_in_radians
#         ) / (4 + 4 * p * math.sin(angle_in_radians))
#         equation_right_part = (1 + p * lambda_from_angle(angle_in_radians)) / (
#             1
#             + p * math.sin(angle_in_radians)
#             + lambda_from_angle(angle_in_radians) * (math.cos(angle_in_radians) ** 2)
#         )
#         result = equation_left_part - equation_right_part
#         # print(f'alpha_from_angle {angle_in_radians=} {equation_left_part=} {equation_right_part=} {result=}')
#         return result
#
#     res = brenth(partial(alpha_from_angle, p=p), 0, math.pi / 4)
#     return math.degrees(res)
#
#
# def compute_launch_angle_for_point(initial_velocity, k, x, y, g=EARTH_GRAVITY_CONSTANT):
#     x_from_angle = partial(x_from_velocity_angle_and_y, initial_velocity=initial_velocity, y=y, k=k, g=g)
#     def f(angle_in_degrees):
#         x_for_angle = x_from_angle(angle_in_degrees=angle_in_degrees)
#         result = x_for_angle - x
#         print(f'{angle_in_degrees=} {x_for_angle=} {x=} {result=}')
#         return result
#     launch_angles = drag_free_ballistics.calculate_drag_free_launch_angles_in_degrees(x, y, initial_velocity, g)
#     if launch_angles is not None:
#         return internal_find_one_angle(f, launch_angles, ANGLE_EPSILON_IN_DEGREES, EPSILON_IN_METERS)
#         #return levenberg_marquandt(f, launch_angles, ANGLE_EPSILON, EPSILON_IN_METER)
#
# def compute_launch_angle_for_point_minimizer(initial_velocity, k, x, y, g=EARTH_GRAVITY_CONSTANT):
#     x_from_angle = partial(x_from_velocity_angle_and_y, initial_velocity=initial_velocity, y=y, k=k, g=g)
#     def f(angle_in_degrees):
#         x_for_angle = x_from_angle(angle_in_degrees=angle_in_degrees)
#         result = abs(x_for_angle - x)
#         print(f'{angle_in_degrees=} {x_for_angle=} {x=} {result=}')
#         return result
#
#     launch_angles = drag_free_ballistics.calculate_drag_free_launch_angles_in_degrees(x, y, initial_velocity, g)
#     if launch_angles is None:
#         launch_angles = (0, 90)
#     res = minimize_scalar(f, bounds=launch_angles, method="bounded")
#     if res.success:
#         angle = res.x
#         solution_exists = res.fun<0.1
#         return angle, solution_exists
#     raise ValueError("No solution found")