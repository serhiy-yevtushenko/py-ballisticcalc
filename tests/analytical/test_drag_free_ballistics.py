import math

import pytest

from py_ballisticcalc.analytical.drag_free_ballistics import (
    calculate_drag_free_launch_angles_in_degrees,
    is_target_reachable,
    EARTH_GRAVITY_CONSTANT,
    calculate_drag_free_total_time_of_flight,
    calculate_drag_free_apex_height,
    calculate_drag_free_range,
    calculate_drag_free_total_time_of_flight_to_point,
    compute_drag_free_position_at_time,
    calculate_drag_free_range_for_time,
)


def test_flat_ground_shot():
    """Test shooting on flat ground (height = 0)"""
    angles = calculate_drag_free_launch_angles_in_degrees(100, 0, 50)
    assert angles is not None
    low, high = angles

    # For flat ground, angles should be complementary
    # assert pytest.approx(low + high, abs=1e-5) == math.pi/2
    print(f"{low=} {high=}")

    assert pytest.approx(low + high, abs=1e-5) == 90

    # Check both angles are positive
    assert low > 0
    assert high > 0

    # Low angle should be less than 45 degrees
    # assert low < math.pi/4
    assert low < 45
    # High angle should be more than 45 degrees
    # assert high > math.pi/4
    assert high > 45


def test_unreachable_target():
    """Test that impossible shots return None"""
    # Try to hit target 1000m away with only 10 m/s velocity
    result = calculate_drag_free_launch_angles_in_degrees(1000, 0, 10)
    assert result is None


def test_uphill_shot():
    """Test shooting uphill"""
    angles = calculate_drag_free_launch_angles_in_degrees(100, 50, 50)
    assert angles is not None
    low, high = angles

    # Both angles should be positive
    assert low > 0
    assert high > 0

    # Low angle should be less than high angle
    assert low < high


def test_downhill_shot():
    """Test shooting downhill"""
    angles = calculate_drag_free_launch_angles_in_degrees(100, -20, 50)
    assert angles is not None
    low, high = angles

    # Both angles should be real and different
    assert not pytest.approx(low) == high

    # For downhill shots, one angle might be negative
    # But high angle should always be positive
    assert high > 0


def test_maximum_range():
    """Test shots at the limit of reachability"""
    v = 50  # m/s
    # Maximum range on flat ground is v²/g
    max_range = v**2 / EARTH_GRAVITY_CONSTANT

    # Test just within maximum range
    angles = calculate_drag_free_launch_angles_in_degrees(max_range * 0.99, 0, v)
    assert angles is not None

    # Test just beyond maximum range
    angles = calculate_drag_free_launch_angles_in_degrees(max_range * 1.01, 0, v)
    assert angles is None


@pytest.mark.parametrize(
    "distance,height,velocity,expected",
    [
        (100, 0, 50, True),  # Normal shot
        (1000, 0, 10, False),  # Too far
        (100, 50, 50, True),  # Uphill shot
        (100, -20, 50, True),  # Downhill shot
    ],
)
def test_target_reachability(distance, height, velocity, expected):
    """Test the is_target_reachable method with various scenarios"""
    assert is_target_reachable(distance, height, velocity) == expected


@pytest.mark.parametrize("g", [EARTH_GRAVITY_CONSTANT, 1.62, 3.72])  # Earth, Moon, Mars
def test_different_gravity(g):
    """Test calculator works with different gravitational constants"""
    angles = calculate_drag_free_launch_angles_in_degrees(100, 0, 50, g)
    assert angles is not None
    low, high = angles
    assert low > 0
    assert high > 0
    assert low < high


@pytest.mark.parametrize(
    "velocity,angle,expected_time_of_flight,expected_max_height, expected_range",
    [
        (10, 45, 1.44, 2.55, 10.20),
        (20, 30, 2.04, 5.10, 35.31),
        (50, 60, 8.84, 95.77, 220.97),
    ],
)
def test_compute_projectile_motion(
    velocity, angle, expected_time_of_flight, expected_max_height, expected_range
):
    time_of_flight = calculate_drag_free_total_time_of_flight(velocity, angle)
    assert pytest.approx(time_of_flight, 0.01) == expected_time_of_flight
    max_height = calculate_drag_free_apex_height(velocity, angle)
    assert pytest.approx(max_height, 0.01) == expected_max_height
    range = calculate_drag_free_range(velocity, angle)
    assert pytest.approx(range, 0.01) == expected_range



@pytest.mark.parametrize(
    "velocity,angle,max_time, expected_range",
    [
        (10, 45, 1.44, 10.20),
        (10, 45, 1.00, 7.07),
        (20, 30, 2.04, 35.31),
        (50, 60, 8.84, 220.97),
    ],
)
def test_compute_drag_free_range_for_time(velocity, angle, max_time, expected_range):
    range = calculate_drag_free_range_for_time(velocity, angle, max_time)
    assert pytest.approx(expected_range, 0.01) == range


def test_compute_projectile_time_to_position():
    distance_in_meter = 400
    height_in_meter = 300
    velocity_meter_per_second = 930

    launch_angles = calculate_drag_free_launch_angles_in_degrees(
        distance_in_meter, height_in_meter, velocity_meter_per_second
    )
    assert launch_angles is not None
    unlofted_angle, lofted_angle = launch_angles
    times_x = calculate_drag_free_total_time_of_flight_to_point(
        distance_in_meter, height_in_meter, velocity_meter_per_second
    )
    pos_unlofted = compute_drag_free_position_at_time(velocity_meter_per_second, unlofted_angle, times_x[0])
    pos_lofted = compute_drag_free_position_at_time(velocity_meter_per_second, lofted_angle, times_x[1])

    assert pos_unlofted[0]==pytest.approx(pos_lofted[0], abs=1e-8)
    assert pos_unlofted[1]==pytest.approx(pos_lofted[1], abs=1e-8)


def test_launch_angles_vertical_shot():
    angles = calculate_drag_free_launch_angles_in_degrees(0, 100, 50, EARTH_GRAVITY_CONSTANT)
    assert (90, 90)==angles

    angles = calculate_drag_free_launch_angles_in_degrees(0, -100, 50, EARTH_GRAVITY_CONSTANT)
    assert (-90, -90)==angles

    angles = calculate_drag_free_launch_angles_in_degrees(0, 0, 50, EARTH_GRAVITY_CONSTANT)
    assert angles is None
