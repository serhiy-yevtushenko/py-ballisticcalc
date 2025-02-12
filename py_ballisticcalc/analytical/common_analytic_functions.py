import math

EARTH_GRAVITY_CONSTANT = 9.81


def flight_time_from_apex_height(apex_height, g=EARTH_GRAVITY_CONSTANT):
    return 2 * math.sqrt(2 * apex_height / g)


def compute_p(initial_velocity, k):
    """Compute coefficient p from k and initial velocity"""
    return k*(initial_velocity**2)


def compute_k_from_p(p, initial_velocity):
    """Determine coefficient k from p and initial velocity."""
    return p/(initial_velocity**2)


def f_from_angle(angle_in_degrees):
    angle_in_radians = math.radians(angle_in_degrees)
    return math.sin(angle_in_radians) / (math.cos(angle_in_radians) ** 2) + math.log(
        math.tan(angle_in_radians / 2 + math.pi / 4)
    )


def compute_velocity_at_apex(V_0, k, angle_in_degrees):
    # this formula is from article:
    # approximate analytical description of the projectile motion with a quadratic
    # drag force (2014)
    theta_0_radians = math.radians(angle_in_degrees)
    sqrt_term = 1 + k * (V_0**2) * (
        math.sin(theta_0_radians)
        + (math.cos(theta_0_radians) ** 2)
        * math.log(math.tan(theta_0_radians / 2 + math.pi / 4))
    )
    return V_0 * math.cos(theta_0_radians) / math.sqrt(sqrt_term)

    # return V_0*math.cos(theta_0_radians)/math.sqrt(1+k*(V_0**2)*(math.cos(theta_0_radians)**2)*f_from_theta(theta_0_in_degrees))


def flight_distance_from_apex_height(V_0, k, angle_in_degrees, apex_height, g=EARTH_GRAVITY_CONSTANT):
    return compute_velocity_at_apex(V_0, k, angle_in_degrees) * flight_time_from_apex_height(apex_height, g)


# TODO: clarify, whether this is a general formula or
# heuristic estimate (then it should not belong here)
def time_of_ascent_from_apex_height(V_0, k, angle_in_degrees, apex_height, g):
    return (
            flight_time_from_apex_height(
                apex_height, g
            )
            - k
            * apex_height
            * compute_velocity_at_apex(V_0, k, angle_in_degrees)
    ) / 2


def estimate_apex_distance_from_apex_height(V_0, k, angle_in_degrees, apex_height, g):
    return math.sqrt(
        flight_distance_from_apex_height(
            V_0, k, angle_in_degrees, apex_height, g
        )
        * apex_height
        / math.tan(math.radians(angle_in_degrees))
    )

def velocity_from_angle(initial_velocity, k, launch_angle_in_degree, angle_in_degree):
    launch_angle_in_radians = math.radians(launch_angle_in_degree)
    angle_in_radians = math.radians(angle_in_degree)
    return (initial_velocity * math.cos(launch_angle_in_radians)) / (
        math.cos(angle_in_radians)
        * math.sqrt(
            1
            + k
            * (initial_velocity**2)
            * (math.cos(launch_angle_in_radians) ** 2)
            * (f_from_angle(launch_angle_in_degree) - f_from_angle(angle_in_degree))
        )
    )

def compute_y_from_x(x, H, L, x_apex):
    return (H*x*(L-x))/(x_apex**2+(L-2*x_apex)*x)


def x_from_apex_height_distance_angle_and_y(initial_velocity, k, angle_in_degrees, y, H, x_apex, g):
    L = flight_distance_from_apex_height(
        initial_velocity, k, angle_in_degrees, H, g
    )
    delta = L / 2 + y / H * (x_apex - L / 2)
    print(
        f"{angle_in_degrees=} {y=} {L=} {H=} {x_apex=} {delta=} {delta**2=} {L*y/math.tan(math.radians(angle_in_degrees))=}")
    return delta + math.sqrt(delta ** 2 - L * y / math.tan(math.radians(angle_in_degrees)))


def compute_x_from_time(time, L, x_apex, t_apex, time_of_flight):
    a_1 = L/x_apex
    w_1 = time-t_apex
    w_2 = 2*time*(time_of_flight-time)/a_1
    c = 2*(a_1-1)/a_1

    return L*(w_1**2+w_2+w_1*math.sqrt(w_1**2 + c*w_2))/(2*(w_1**2)+a_1*w_2)
