
import time

import pytest

import py_ballisticcalc.analytical.chudinov_2014 as chudinov_2014
import py_ballisticcalc.analytical.drag_free_ballistics as drag_free_ballistics
from py_ballisticcalc.analytical.common_analytic_functions import compute_velocity_at_apex, compute_p, compute_k_from_p
from py_ballisticcalc.analytical.drag_free_ballistics import EARTH_GRAVITY_CONSTANT


def test_maximal_height():
    initial_velocity = 930
    angle_in_degrees = 38.58299087491584
    drag_free_max_range_height = drag_free_ballistics.calculate_drag_free_apex_height(initial_velocity, angle_in_degrees)
    print(f'Drag-Free max height for launch angle {angle_in_degrees} is {drag_free_max_range_height}')
    drag_free_max_range = drag_free_ballistics.calculate_drag_free_range(initial_velocity, angle_in_degrees)
    assert drag_free_max_range_height == pytest.approx(17145.31)
    assert drag_free_max_range == pytest.approx(85962.56)

@pytest.fixture()
def baseball_example():
    # the example of the baseball from article
    # corresponds to having
    # radius_in_cm = 6
    # mass_in_grams = 145
    # C_d 0.38 and
    # terminal_velocity_mps = 40
    # maximal_initial_velocity_mps = 55
    # initial launch angle corresponds to one, used in Chudinov 2014

    initial_velocity = 40
    angle_in_degrees = 45
    k = 0.000625

    return initial_velocity, angle_in_degrees, k


def badminton_shuttlecock_example():
    # radius_in_cm = 6
    # mass_in_grams = 5
    # C_d 0.60 and
    # terminal_velocity_mps = 6.7
    initial_velocity = 117
    angle_in_degrees = 45
    k = 0.000625

    return initial_velocity, angle_in_degrees, k


def test_maximal_height_applicable_area(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example

    drag_free_max_range_apex_height = drag_free_ballistics.calculate_drag_free_apex_height(initial_velocity, angle_in_degrees)
    print(f'Drag-Free apex height for launch angle {angle_in_degrees} is {drag_free_max_range_apex_height}')
    assert drag_free_max_range_apex_height == pytest.approx(40.7, abs=0.1)
    drag_max_height = chudinov_2014.compute_apex_height(initial_velocity, k, angle_in_degrees, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert drag_max_height==pytest.approx(30.1, abs=0.1)

def test_time_of_flight_applicable_area(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example


    drag_free_max_range = drag_free_ballistics.calculate_drag_free_range(initial_velocity, angle_in_degrees, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert drag_free_max_range == pytest.approx(163.09887869520895)
    launch_angles = drag_free_ballistics.calculate_drag_free_launch_angles_in_degrees(drag_free_max_range, 0, initial_velocity, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert launch_angles[0] == pytest.approx(45.0)
    assert launch_angles[1] == pytest.approx(45.0)

    drag_free_time_of_flight_unlofted, drag_free_time_of_flight_lofted = drag_free_ballistics.calculate_drag_free_total_time_of_flight_to_point(drag_free_max_range, 0,
                                                                                                      initial_velocity, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert drag_free_time_of_flight_unlofted == pytest.approx(5.766, abs=0.001)
    assert drag_free_time_of_flight_lofted == pytest.approx(5.766, abs=0.001)

    time_of_flight = chudinov_2014.determine_flight_time(initial_velocity, k, angle_in_degrees, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert time_of_flight == pytest.approx(4.956, abs=0.001)

def test_apex_velocity_applicable_area(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    velocity_max_height  = compute_velocity_at_apex(initial_velocity, k, angle_in_degrees)
    assert velocity_max_height==pytest.approx(19.3, abs=0.1)

def test_flight_range(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    flight_distance = chudinov_2014.compute_flight_distance(initial_velocity, k, angle_in_degrees)
    assert flight_distance==pytest.approx(95.65, abs=0.01)

def test_apex_time(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    time_to_apex=chudinov_2014.compute_time_of_ascent(initial_velocity, k, angle_in_degrees)
    print(f"{time_to_apex=}")
    assert time_to_apex==pytest.approx(2.30, abs=0.01)

def test_apex_distance(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    x_apex=chudinov_2014.compute_apex_distance(initial_velocity, k, angle_in_degrees)
    print(f"{x_apex=}")
    assert x_apex==pytest.approx(53.68, abs=0.01)

def test_impact_angle(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    impact_angle = chudinov_2014.compute_impact_angle(initial_velocity, k, angle_in_degrees, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert impact_angle==pytest.approx(-58.6, abs=0.1)

def test_impact_velocity(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    impact_velocity = chudinov_2014.compute_impact_velocity(initial_velocity, k, angle_in_degrees, drag_free_ballistics.EARTH_GRAVITY_CONSTANT)
    assert impact_velocity == pytest.approx(26.0, abs=0.1)

def test_y_from_t(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g = drag_free_ballistics.EARTH_GRAVITY_CONSTANT

    H = chudinov_2014.compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    time_apex = chudinov_2014.compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    time_flight = chudinov_2014.determine_flight_time(initial_velocity, k, angle_in_degrees, g)

    y_from_t = chudinov_2014.get_y_from_t(initial_velocity, k, angle_in_degrees, g)

    assert y_from_t(0) == 0
    assert y_from_t(time_flight)== 0
    assert y_from_t(time_apex) == pytest.approx(H)

def test_compute_trajectory(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g=EARTH_GRAVITY_CONSTANT
    trajectory = chudinov_2014.compute_trajectory(initial_velocity, k, angle_in_degrees, g)
    expected_distance = chudinov_2014.compute_flight_distance(initial_velocity, k, angle_in_degrees, g)
    assert len(trajectory)==497
    assert trajectory[0] == (0, 0)
    assert trajectory[-1][0] == pytest.approx(expected_distance)
    assert trajectory[-1][1] == pytest.approx(0)

# def test_find_optimal_launch_angle(baseball_example):
#     initial_velocity, angle_in_degrees, k = baseball_example
#     start_time = time.time_ns()
#     optimal_angle_in_degrees = chudinov_2014.compute_max_range_launch_angle(initial_velocity, k)
#     end_time = time.time_ns()
#
#     print(f'{optimal_angle_in_degrees=} Elapsed time: {(end_time-start_time)/1e9: 0.9f}')
#     assert optimal_angle_in_degrees==pytest.approx(40.8808370318302)
#
#     max_range = chudinov_2014.compute_flight_distance(initial_velocity, k, optimal_angle_in_degrees)
#     range_of_flight = chudinov_2014.compute_flight_distance(initial_velocity, k, angle_in_degrees)
#     print(f'{max_range=} {range_of_flight=}')
#     assert range_of_flight<max_range
#
#     assert range_of_flight==pytest.approx(95.6571)
#     assert max_range==pytest.approx(96.6360)
#
#
# def test_launch_angle_for_point(baseball_example):
#     initial_velocity, _, k = baseball_example
#     y = 20
#     x = 10
#     launch_angle, solution_exist = chudinov_2014.compute_launch_angle_for_point(initial_velocity, k, x, y)
#     print(f'{launch_angle=}')
#     assert launch_angle == pytest.approx(86.644, 1e-3)
#     assert solution_exist

def test_x_from_t(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g = drag_free_ballistics.EARTH_GRAVITY_CONSTANT

    x_from_t = chudinov_2014.get_x_from_t(initial_velocity, k, angle_in_degrees, g)
    time_flight = chudinov_2014.determine_flight_time(initial_velocity, k, angle_in_degrees, g)
    flight_distance = chudinov_2014.compute_flight_distance(initial_velocity, k, angle_in_degrees)
    time_apex = chudinov_2014.compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    x_apex = chudinov_2014.compute_apex_distance(initial_velocity, k, angle_in_degrees, g)

    assert x_from_t(0) == 0
    assert x_from_t(time_flight) == pytest.approx(flight_distance)
    assert x_from_t(time_flight) == pytest.approx(flight_distance)
    assert x_from_t(time_apex) == pytest.approx(x_apex)

def test_compute_p(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    p = compute_p(initial_velocity, k)
    assert p==pytest.approx(1.0)
    k_computed = compute_k_from_p(p, initial_velocity)
    assert k_computed==pytest.approx(k)
