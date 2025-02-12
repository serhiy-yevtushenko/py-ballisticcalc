import math
from functools import partial

from py_ballisticcalc.analytical.common_analytic_functions import (
    EARTH_GRAVITY_CONSTANT,
    compute_x_from_time,
    estimate_apex_distance_from_apex_height,
    flight_distance_from_apex_height,
    flight_time_from_apex_height,
    time_of_ascent_from_apex_height,
    velocity_from_angle,
    x_from_apex_height_distance_angle_and_y,
)

# from scipy.optimize import brenth, minimize_scalar

STEP_SIZE = 0.5


# the formulas in this implementation correspond to the article:
# Approximate analytical description of the projectile motion with
# drag force by Peter Chudinov (Athens Journal Of Sciences, Volume 1, Issue 2)
# Pages 97-106 https://doi.org/10.30958/ajs.1-2-2
# Area of applicability, according to article, is
# 0 <= launch_angle <=70 degree
# 0 <= initial_velocity <=50 MPS
# 0 <= p <= 1.5, where p = k*(initial_velocity**2)
# k = 1/(terminal_velocity**2) = (\rho_a*C_d*S)/(2*m*g)
# where \rho_a is air density, C_d is the drag factor for sphere,
# S is the cross-sectional area of object
# and terminal_velocity is the terminal velocity of the projectile
# It could be as well, that area of applicability extends to
# 0 <= launch_angle <=90 degree
# 0 <= initial_velocity <=80 MPS
# 0 <= p <= 4
# (In the article it's stated, that the formula were extended in another articles
# but do not state clearly, whether these results applicable to formulas in the
# article.
# the example for baseball for article applies for p=1

def compute_apex_height(V_0, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    theta_0_radians = math.radians(angle_in_degrees)

    nominator = (V_0**2) * (math.sin(theta_0_radians) ** 2)
    # print(f"{k=} {V_0 ** 2=} {math.sin(theta_0_radians)=}")
    denominator = g * (2 + k * (V_0**2) * math.sin(theta_0_radians))
    # print(f"{nominator=} {denominator=}")
    return nominator / denominator


def determine_flight_time(V_0, k, theta_0_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    return flight_time_from_apex_height(
        compute_apex_height(V_0, k, theta_0_in_degrees, g), g
    )

def compute_flight_distance(V_0, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    return flight_distance_from_apex_height(
        V_0, k, angle_in_degrees, compute_apex_height(V_0, k, angle_in_degrees, g), g
    )


def compute_time_of_ascent(V_0, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    apex_height = compute_apex_height(V_0, k, angle_in_degrees, g)
    return time_of_ascent_from_apex_height(V_0, k, angle_in_degrees, apex_height, g)


def compute_apex_distance(V_0, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    apex_height = compute_apex_height(V_0, k, angle_in_degrees, g)
    return estimate_apex_distance_from_apex_height(V_0, k, angle_in_degrees, apex_height, g)


def compute_impact_angle(
    initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT
):
    L = compute_flight_distance(initial_velocity, k, angle_in_degrees)
    H = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    return math.degrees(
        -math.atan2(
            L * H,
            (L - compute_apex_distance(initial_velocity, k, angle_in_degrees, g)) ** 2,
        )
    )


def compute_impact_velocity(
    initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT
):
    impact_angle = compute_impact_angle(initial_velocity, k, angle_in_degrees, g)

    return velocity_from_angle(initial_velocity, k, angle_in_degrees, impact_angle)


def compute_y_from_time(time, H, time_apex, time_of_flight):
    return (H*time*(time_of_flight-time))/(time_apex ** 2 + (time_of_flight - 2 * time_apex) * time)

def get_y_from_t(initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    apex_height = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    time_apex = time_of_ascent_from_apex_height(initial_velocity, k, angle_in_degrees, apex_height, g)
    time_flight = flight_time_from_apex_height(apex_height, g)
    y_from_t = partial(
        compute_y_from_time,
        H=apex_height,
        time_apex=time_apex,
        time_of_flight=time_flight,
    )
    return y_from_t


def get_x_from_t(initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT):
    H = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    L = flight_distance_from_apex_height(
        initial_velocity, k, angle_in_degrees, H, g
    )
    x_apex = compute_apex_distance(initial_velocity, k, angle_in_degrees, g)
    t_apex = compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    time_of_flight = flight_time_from_apex_height(
        compute_apex_height(initial_velocity, k, angle_in_degrees, g), g
    )
    return partial(compute_x_from_time, L=L, x_apex=x_apex, t_apex = t_apex, time_of_flight=time_of_flight)

# def compute_max_range_launch_angle(initial_velocity, k, g=EARTH_GRAVITY_CONSTANT):
#     p = k*(initial_velocity**2)
#     def lambda_from_angle(angle_in_radians):
#         return math.log(math.tan(angle_in_radians/2+math.pi/4))
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
#         #print(f'alpha_from_angle {angle_in_radians=} {equation_left_part=} {equation_right_part=} {result=}')
#         return result
#     res = brenth(partial(alpha_from_angle, p=p), 0, math.pi/4)
#     return math.degrees(res)

def x_from_velocity_angle_and_y(initial_velocity, k, angle_in_degrees, y, g):
    H = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    # This one is approximate and related to chudinov_2014
    x_apex = estimate_apex_distance_from_apex_height(initial_velocity, k, angle_in_degrees, H,

                                                     EARTH_GRAVITY_CONSTANT)
    return x_from_apex_height_distance_angle_and_y(initial_velocity, k, angle_in_degrees, y, H, x_apex, g)


# def compute_launch_angle_for_point(initial_velocity, k, x, y, g=EARTH_GRAVITY_CONSTANT):
#     x_from_angle = partial(x_from_velocity_angle_and_y, initial_velocity=initial_velocity, y=y, k=k, g=g)
#     def f(angle_in_degrees):
#         x_for_angle = x_from_angle(angle_in_degrees=angle_in_degrees)
#         result = abs(x_for_angle - x)
#         print(f'{angle_in_degrees=} {x_for_angle=} {x=} {result=}')
#         return result
#     launch_angles = drag_free_ballistics.calculate_drag_free_launch_angles_in_degrees(x, y, initial_velocity, g)
#     if launch_angles is not None:
#         res = minimize_scalar(f, bounds=launch_angles, method="bounded")
#         if res.success:
#             angle = res.x
#             solution_exists = res.fun < 0.1
#             return angle, solution_exists
#     raise ValueError("No solution found")


def compute_trajectory(initial_velocity, k, angle_in_degrees, g=EARTH_GRAVITY_CONSTANT,
                       time_step = 0.01):

    time_of_flight = determine_flight_time(initial_velocity, k, angle_in_degrees, g)
    x_from_t = get_x_from_t(initial_velocity, k, angle_in_degrees, g)
    y_from_t = get_y_from_t(initial_velocity, k, angle_in_degrees, g)
    t = 0
    points = []
    while t<=time_of_flight:
        x = x_from_t(t)
        y=y_from_t(t)
    #    points.append(((x, y), t))
        points.append((x, y))
        t += time_step
    if t!=time_of_flight:
        points.append((x_from_t(time_of_flight), y_from_t(time_of_flight)))
    return points

