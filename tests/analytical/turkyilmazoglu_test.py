import math

import pytest

import py_ballisticcalc.analytical.chudinov_2014 as chudinov_2014
from py_ballisticcalc.analytical import common_analytic_functions
from py_ballisticcalc.analytical.common_analytic_functions import compute_p
from py_ballisticcalc.analytical.drag_free_ballistics import EARTH_GRAVITY_CONSTANT
from py_ballisticcalc.analytical.turkyilmazoglu import (
    get_dimensionless_Y,
    get_dimensionless_X,
    compute_apex_height_dimensionless,
    from_X_to_x,
    from_Y_to_y,
    from_dimensionless_tau_to_time,
    from_U_to_velocity,
    get_dimensionless_time,
    get_dimensionless_U,
    compute_apex_distance_dimensionless,
    compute_apex_distance,
    compute_apex_height,
    compute_time_of_ascent,
    compute_time_of_ascent_dimensionless,
    # compute_max_range_launch_angle,
    # compute_flight_distance
    # compute_launch_angle_for_point,
)
from .chudinov_2014_test import baseball_example

ANGLES_BETWEEN_5_AND_85 = list(range(5, 90, 5))

def test_get_dimensionless_Y(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    assert get_dimensionless_Y(30.1242048944448, initial_velocity, g) == pytest.approx(
        0.1846990312590647
    )
    assert from_Y_to_y(0.1846990312590647, initial_velocity, g) == pytest.approx(
        30.1242048944448
    )

def test_get_dimensionless_X(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    assert get_dimensionless_X(53.68, initial_velocity, g) == pytest.approx(0.3291255)
    assert from_X_to_x(0.3291255, initial_velocity, g) == pytest.approx(53.68)


def test_get_dimensionless_time(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    assert get_dimensionless_time(2.30, initial_velocity, g) == pytest.approx(0.564075)
    assert from_dimensionless_tau_to_time(
        0.564075, initial_velocity, g
    ) == pytest.approx(2.30)


EXPECTED_F_VALUES = {
    5: 0.17520029196951092,
    10: 0.35447293825665804,
    15: 0.5422436642451047,
    20: 0.743707537873393,
    25: 0.9653889034030505,
    30: 1.215972811000721,
    35: 1.507632149225428,
    40: 1.8582764270814582,
    45: 2.295587149392638,
    50: 2.86472264842833,
    55: 3.6441346196360636,
    60: 4.781059512062569,
    65: 6.580789907723151,
    70: 9.768501731019036,
    75: 16.447126462015145,
    80: 35.09585639811943,
    85: 134.2764438363638
}

@pytest.mark.parametrize("angle_in_degrees", ANGLES_BETWEEN_5_AND_85)
def test_f_from_angle(angle_in_degrees):
    assert common_analytic_functions.f_from_angle(angle_in_degrees)==pytest.approx(EXPECTED_F_VALUES[angle_in_degrees])


def test_get_dimensionless_velocity(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    assert get_dimensionless_U(26, initial_velocity) == pytest.approx(0.65)
    assert from_U_to_velocity(0.65, initial_velocity) == pytest.approx(26)


def test_compute_apex_distance_30_degree(baseball_example):
    initial_velocity, _, k = baseball_example
    angle_in_degrees = 30
    print(f'{locals()=}')
    g = EARTH_GRAVITY_CONSTANT
    chudinov_apex_distance = chudinov_2014.compute_apex_distance(initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT)
    assert chudinov_apex_distance == pytest.approx(50.80404447278348)
    assert get_dimensionless_X(chudinov_apex_distance, initial_velocity, g)==pytest.approx(0.311492, abs=1e-6)
    p = compute_p(initial_velocity, k)
    apex_d = compute_apex_distance_dimensionless(p, angle_in_degrees)
    print(f"{apex_d=} {from_X_to_x(apex_d, initial_velocity, g)=}")
    assert apex_d==pytest.approx(0.3045926)


    dist = compute_apex_distance(initial_velocity, k, angle_in_degrees, g)
    print(f"{dist=} {get_dimensionless_X(dist,initial_velocity, g)=}")
    assert dist == pytest.approx(49.67872156909495)
    assert get_dimensionless_X(dist,initial_velocity, g)==pytest.approx(0.304592, abs=1e-6)



def test_compute_apex_distance(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    print(f'{locals()=}')
    g = EARTH_GRAVITY_CONSTANT
    chudinov_apex_distance = chudinov_2014.compute_apex_distance(initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT)
    assert chudinov_apex_distance == pytest.approx(53.680469004276965)
    assert get_dimensionless_X(chudinov_apex_distance, initial_velocity, g)==pytest.approx(0.329128, abs=1e-6)
    p = compute_p(initial_velocity, k)
    apex_d = compute_apex_distance_dimensionless(p, angle_in_degrees)
    print(f"{apex_d=} {from_X_to_x(apex_d, initial_velocity, g)=}")
    assert apex_d==pytest.approx(0.323581)


    dist = compute_apex_distance(initial_velocity, k, angle_in_degrees, g)
    print(f"{dist=} {get_dimensionless_X(dist,initial_velocity, g)=}")
    assert dist == pytest.approx(52.77572795863861)
    assert get_dimensionless_X(dist,initial_velocity, g)==pytest.approx(0.323581)


def test_compute_height(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    print(f'{initial_velocity=} {angle_in_degrees=} {k=}')
    print(f"{ compute_p(initial_velocity,k)=} {initial_velocity=}")
    # k = chudinov_2014.compute_k_from_p(0.1, initial_velocity)
    # print(f'p=0.1 {k=}')

    chudinov_height = chudinov_2014.compute_apex_height(
        initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT
    )
    assert chudinov_height == pytest.approx(30.1242048944448)
    dimensionless_H = get_dimensionless_Y(
        chudinov_height, initial_velocity, EARTH_GRAVITY_CONSTANT
    )
    assert dimensionless_H == pytest.approx(0.184699)

    height = compute_apex_height(initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT)
    assert height == pytest.approx(29.698609582059117)
    Y = get_dimensionless_Y(height, initial_velocity, EARTH_GRAVITY_CONSTANT)
    assert Y == pytest.approx(0.1820896)


@pytest.mark.parametrize("angle_in_degrees", ANGLES_BETWEEN_5_AND_85)
def test_dimensionless_and_normal_apex_distance_correspondence(angle_in_degrees, baseball_example):
    initial_velocity, _, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    #for angle in range(5, 90, 5):
    p = compute_p(initial_velocity, k)
    apex_x = compute_apex_distance(initial_velocity, k, angle_in_degrees, g)
    apex_x_dimless = compute_apex_distance_dimensionless(p, angle_in_degrees)
    print(f'{apex_x=} {apex_x_dimless=}')
    assert get_dimensionless_X(apex_x, initial_velocity, g)==pytest.approx(apex_x_dimless)
    #assert from_Y_to_y(height_dimless, initial_velocity, g) == pytest.approx(height)

@pytest.mark.parametrize("angle_in_degrees", ANGLES_BETWEEN_5_AND_85)
def test_dimensionless_and_normal_height_computation_correspondence(angle_in_degrees, baseball_example):
    initial_velocity, _, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    #for angle in range(5, 90, 5):
    p = compute_p(initial_velocity, k)
    height = compute_apex_height(initial_velocity, k, angle_in_degrees, g)
    height_dimless = compute_apex_height_dimensionless(p, angle_in_degrees)
    print(f'{height=} {height_dimless=}')
    assert get_dimensionless_Y(height, initial_velocity, g)==pytest.approx(height_dimless)
    #assert from_Y_to_y(height_dimless, initial_velocity, g) == pytest.approx(height)


def test_compute_height_dimensionless(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example

    p = compute_p(initial_velocity, k)
    cos_theta_0 = math.cos(math.radians(angle_in_degrees))
    print(f'{p=} {angle_in_degrees=} {cos_theta_0=} {cos_theta_0**2=}')
    height_t = compute_apex_height_dimensionless(p, angle_in_degrees)
    print(f"{height_t=}")
    assert height_t==pytest.approx(0.1820896)

def test_compute_height_dimensionless_3(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    angle_in_degrees = 30

    p = compute_p(initial_velocity, k)
    cos_theta_0 = math.cos(math.radians(angle_in_degrees))
    print(f'{p=} {angle_in_degrees=} {cos_theta_0=} {cos_theta_0**2=}')
    height_t = compute_apex_height_dimensionless(p, angle_in_degrees)
    print(f"{height_t=}")
    print(f"{from_Y_to_y(0.09734150915953972, initial_velocity, EARTH_GRAVITY_CONSTANT)=}")
    assert height_t==pytest.approx(0.09734150915953972)

def test_compute_height_2(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    angle_in_degrees = 76.71747441146101
    chudinov_height = chudinov_2014.compute_apex_height(
        initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT
    )
    assert chudinov_height == pytest.approx(51.959817646673784)
    height = compute_apex_height(initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT)
    assert height == pytest.approx(53.476889754396936)


def test_compute_height_3(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    angle_in_degrees = 30
    chudinov_height = chudinov_2014.compute_apex_height(
        initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT
    )
    assert chudinov_height == pytest.approx(16.309887869520892)
    print(f"{from_Y_to_y(0.104550438038688, initial_velocity, EARTH_GRAVITY_CONSTANT)=}")
    height = compute_apex_height(initial_velocity, k, angle_in_degrees, EARTH_GRAVITY_CONSTANT)
    assert height == pytest.approx(15.87629099442034)

def test_compute_time_of_ascent(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    print(f'{initial_velocity=} {angle_in_degrees=} {k=}')

    print(f'{from_dimensionless_tau_to_time(0.565427, initial_velocity, EARTH_GRAVITY_CONSTANT)=}')

    print(f"{chudinov_2014.compute_time_of_ascent(initial_velocity, k, angle_in_degrees)=}")

    time_of_ascent = compute_time_of_ascent(initial_velocity, k, angle_in_degrees,EARTH_GRAVITY_CONSTANT)
    print(f'{time_of_ascent=}')
    assert time_of_ascent==pytest.approx(2.305512742099898)
    assert get_dimensionless_time(time_of_ascent, initial_velocity, EARTH_GRAVITY_CONSTANT)==pytest.approx(0.565427)


def test_compute_time_of_ascent_dimless(baseball_example):
    initial_velocity, angle_in_degrees, k = baseball_example
    print(f'{initial_velocity=} {angle_in_degrees=} {k=}')
    p = compute_p(initial_velocity, k)
    computed_tau = compute_time_of_ascent_dimensionless(p, angle_in_degrees)
    assert computed_tau == pytest.approx(0.565427)


@pytest.mark.parametrize("angle_in_degrees", ANGLES_BETWEEN_5_AND_85)
def test_dimensionless_and_normal_time_of_ascent_correspondence(angle_in_degrees, baseball_example):
    initial_velocity, _, k = baseball_example
    g = EARTH_GRAVITY_CONSTANT
    p = compute_p(initial_velocity, k)
    time = compute_time_of_ascent(initial_velocity, k, angle_in_degrees, g)
    time_dimless = compute_time_of_ascent_dimensionless(p, angle_in_degrees)
    print(f'{angle_in_degrees=} {time=} {get_dimensionless_time(time, initial_velocity, g)=} {time_dimless=}')
    assert get_dimensionless_time(time, initial_velocity, g)==pytest.approx(time_dimless)


# def test_find_optimal_launch_angle(baseball_example):
#     initial_velocity, angle_in_degrees, k = baseball_example
#     start_time = time.time_ns()
#     optimal_angle_in_degrees = compute_max_range_launch_angle(initial_velocity, k)
#     end_time = time.time_ns()
#
#     print(f'{optimal_angle_in_degrees=} Elapsed time: {(end_time-start_time)/1e9: 0.9f}')
#     assert optimal_angle_in_degrees==pytest.approx(40.8808370318302)
#
#     max_range = compute_flight_distance(initial_velocity, k, optimal_angle_in_degrees)
#     range_of_flight = compute_flight_distance(initial_velocity, k, angle_in_degrees)
#     print(f'{max_range=} {range_of_flight=}')
#     assert range_of_flight<max_range
#     assert range_of_flight==pytest.approx(94.97893502193648)
#     assert max_range==pytest.approx(95.73728008637998)
#
#
# def test_launch_angle_for_point(baseball_example):
#     initial_velocity, _, k = baseball_example
#     y = 20
#     x = 10
#     launch_angle, solution_exists = compute_launch_angle_for_point(initial_velocity, k, x, y)
#     print(f'{launch_angle=}')
#     assert launch_angle == pytest.approx(86.689, abs=1e-3)
#     assert solution_exists
