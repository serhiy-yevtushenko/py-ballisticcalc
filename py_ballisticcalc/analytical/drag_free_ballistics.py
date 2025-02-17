"""
This file provide implementation of the analytical functions
for drag-free ballistic
"""
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Final

EARTH_GRAVITY_CONSTANT: Final[float] = 9.81 # Acceleration due to gravity (m/s^2)

# Default integration step size for BallisticCalculatorInterface
STEP_SIZE = 0.5


def is_target_reachable(
    distance: float, height: float, velocity: float, g: float = EARTH_GRAVITY_CONSTANT
) -> bool:
    """
    Check if target is reachable with given parameters.

    Args:
        distance (float): Horizontal distance to target in meters
        height (float): Vertical height difference to target in meters
        velocity (float): Initial launch velocity in m/s

    Returns:
        bool: True if target is reachable, False otherwise
    """
    return (
        calculate_drag_free_launch_angles_in_degrees(distance, height, velocity, g)
        is not None
    )


def calculate_drag_free_total_time_of_flight(
    velocity: float, angle_in_degrees: float, gravity: float = EARTH_GRAVITY_CONSTANT
):
    angle_rad = math.radians(angle_in_degrees)
    return (2 * velocity * math.sin(angle_rad)) / gravity


def calculate_drag_free_time_to_point_using_y(
    slant_distance_to_target,
    launch_angle_in_degrees,
    muzzle_velocity,
    gravity=EARTH_GRAVITY_CONSTANT,
):
    """"""
    angle_in_rad = math.radians(launch_angle_in_degrees)

    vertical_velocity = muzzle_velocity * math.sin(angle_in_rad)
    target_height = slant_distance_to_target * math.sin(angle_in_rad)
    inner_term = vertical_velocity**2 - 2 * gravity * target_height
    term = math.sqrt(inner_term)
    time_1 = (vertical_velocity - term) / gravity
    time_2 = (vertical_velocity + term) / gravity
    return time_1, time_2


def calculate_drag_free_total_time_of_flight_to_point(
    distance_x_in_meters,
    height_in_meters,
    velocity: float,
    gravity: float = EARTH_GRAVITY_CONSTANT,
):
    launch_angles = calculate_drag_free_launch_angles_in_degrees(
        distance_x_in_meters, height_in_meters, velocity, gravity
    )
    if launch_angles is not None:
        print(f'{launch_angles=}')
        unlofted_launch_angle, lofted_launch_angle = launch_angles
        print(f"{unlofted_launch_angle=} {lofted_launch_angle=}")

        t_x_unlofted = distance_x_in_meters / (
            math.cos(math.radians(unlofted_launch_angle)) * velocity
        )
        t_x_lofted = distance_x_in_meters / (
            math.cos(math.radians(lofted_launch_angle)) * velocity
        )
        return t_x_unlofted, t_x_lofted


def compute_drag_free_position_at_time(
    initial_velocity_mps: float,
    angle_in_degree: float,
    time: float,
    gravity: float = EARTH_GRAVITY_CONSTANT,
) -> tuple[float, float]:
    angle_in_rad = math.radians(angle_in_degree)
    x = initial_velocity_mps * time * math.cos(angle_in_rad)
    y = initial_velocity_mps * time * math.sin(angle_in_rad) - 0.5 * gravity * (time**2)
    return (x, y)


def calculate_drag_free_apex_height(
    velocity: float, angle_in_degrees: float, gravity: float = EARTH_GRAVITY_CONSTANT
):
    angle_rad = math.radians(angle_in_degrees)
    return (velocity**2 * math.sin(angle_rad) ** 2) / (2 * gravity)


def calculate_drag_free_range(
    velocity: float, angle_in_degrees: float, gravity: float = EARTH_GRAVITY_CONSTANT
):
    angle_rad = math.radians(angle_in_degrees)
    return (velocity**2 * math.sin(2 * angle_rad)) / gravity


def calculate_drag_free_range_for_time(
    velocity: float,
    angle_in_degrees: float,
    time: float,
    gravity: float = EARTH_GRAVITY_CONSTANT,
):
    max_possible_range = calculate_drag_free_range(velocity, angle_in_degrees, gravity)
    angle_in_radians = math.radians(angle_in_degrees)
    covered_distance = velocity * time * math.cos(angle_in_radians)
    return min(covered_distance, max_possible_range)


def calculate_drag_free_launch_angles_in_degrees(
    distance: float, height: float, velocity: float, g: float = EARTH_GRAVITY_CONSTANT
) -> Optional[Tuple[float, float]]:
    """
    Calculate both possible launch angles for hitting a target.

    Args:
        distance (float): Horizontal distance to target in meters
        height (float): Vertical height difference to target in meters
        velocity (float): Initial launch velocity in m/s

    Returns:
        Optional[Tuple[float, float]]: Tuple of (low angle, high angle) in degrees,
                                     or None if target is unreachable
                                     Special cases:
                                        if distance is 0 and height > 0:
                                        returns (90, 90)
                                        if distance is 0 and height < 0:
                                        returns (-90, -90)
                                        if both distance and height are 0:
                                        return None (as no angle could be determined)
    """
    # Check if target is reachable with given velocity
    velocity_squared = velocity**2
    discriminant = velocity_squared**2 - g * (
        g * distance**2 + 2 * height * velocity_squared
    )
    # print(f"{discriminant=}")

    if discriminant < 0:
        return None
    if distance == 0:
        if height>0:
            return (90, 90)
        elif height<0:
            return (-90, -90)
        else:
            # There is no angle between point (0, 0) and (0, 0)
            return None
    else:
        # Calculate the two possible angles using quadratic formula
        term1 = velocity_squared
        term2 = math.sqrt(discriminant)
        term3 = g * distance
        assert term3 != 0
        angle1 = math.atan2(term1 - term2, term3)
        angle2 = math.atan2(term1 + term2, term3)

        # Return angles in ascending order (low, high)
        return (math.degrees(min(angle1, angle2)), math.degrees(max(angle1, angle2)))


def ballistic_calculator_no_drag(
    v0_in_mps: float,
    angle_in_degrees: float,
    g: float = EARTH_GRAVITY_CONSTANT,
    step_size: float = STEP_SIZE,
    max_time_in_seconds: Optional[float] = None,
):
    """
    Computes trajectory information for a given launch velocity and angle (no drag).

    Parameters:
        v0_in_mps (float): Initial velocity (m/s).
        angle_in_degrees (float): Launch angle (degrees).

    Returns:
        dict: Trajectory information including:
            - H_max: Maximum height (m).
            - zero_points: [0, range] (m).
            - trajectory_points: List of (x, y) tuples for the trajectory.
    """
    angle_in_degrees = math.radians(angle_in_degrees)

    # Maximum height
    H_max = (v0_in_mps**2 * math.sin(angle_in_degrees) ** 2) / (2 * g)

    # Range (horizontal distance where projectile lands)
    if max_time_in_seconds is None:
        range_ = (v0_in_mps**2 * math.sin(2 * angle_in_degrees)) / g
    else:
        max_range = (v0_in_mps**2 * math.sin(2 * angle_in_degrees)) / g
        time_limited_range = v0_in_mps * max_time_in_seconds * math.cos(angle_in_degrees)
        range_ = min(max_range, time_limited_range)

    # Generate trajectory points
    x = 0
    x_points = []
    y_points = []
    while x<range_:
        x_points.append(x)
        y_points.append(x*math.tan(angle_in_degrees) - (g * x**2) / (2 * v0_in_mps**2 * math.cos(angle_in_degrees) ** 2))
        x+=step_size

    trajectory_points = list(zip(x_points, y_points))
    return {
        "H_max": H_max,
        "D_H_max": range_/2,
        "zero_points": [0, range_],
        "trajectory_points": trajectory_points,
    }


@dataclass
class BallisticCalculatorInterface:
    muzzle_velocity_in_mps: float = 50
    step_size: float = 0.1

    def get_muzzle_velocity(self):
        return self.muzzle_velocity_in_mps

    def compute_trajectory(
        self,
        angle_in_degrees: float,
        step_size: Optional[float] = None,
        max_time_in_seconds: Optional[float] = None,
    ) -> list[tuple[float, float]]:
        trajectory_full_data = self.compute_trajectory_with_height_and_zeros(
            angle_in_degrees,
            step_size=self.step_size if step_size is None else step_size,
            max_time_in_seconds=max_time_in_seconds,
        )
        return trajectory_full_data["trajectory_points"]

    def compute_trajectory_with_height_and_zeros(
        self,
        angle_in_degrees: float,
        step_size: Optional[float] = None,
        max_time_in_seconds: Optional[float] = None,
    ):
        return ballistic_calculator_no_drag(
            self.muzzle_velocity_in_mps,
            angle_in_degrees,
            step_size=self.step_size if step_size is None else step_size,
            max_time_in_seconds=max_time_in_seconds,
        )


def max_height_and_range_for_velocity(v0, g:float=EARTH_GRAVITY_CONSTANT):
    theta = math.radians(45)
    H_max = (v0**2 * math.sin(theta) ** 2) / (2 * g)
    # Range (horizontal distance where projectile lands)
    range_ = (v0**2 * math.sin(2 * theta)) / g
    return {"H_max": H_max, "D_max": range_}

