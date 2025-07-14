import math

import pytest

import py_ballisticcalc
from py_ballisticcalc import (
    Angular,
    HitResult,
    Distance,
    loadMetricUnits,
    RangeError,
    Weight,
    DragModel,
    TableG1,
    Ammo,
    Velocity,
    Temperature,
    Pressure,
    Atmo,
    Weapon,
    Wind,
    Shot,
    TableG7,
    TrajFlag,
)
from tests.fixtures_and_helpers import (
    print_out_trajectory_compact,
    print_out_trajectory_list_compact,
)

ANGLE_EPSILON_IN_DEGREES = 0.0009


class PreferredUnitsContextManager:
    def __enter__(self):
        # print("Storing preferred units...")
        # print(f"On Enter: {PreferredUnits=}")
        values_copies = {}
        for field in getattr(py_ballisticcalc.PreferredUnits, "__dataclass_fields__"):
            values_copies[field] = getattr(py_ballisticcalc.PreferredUnits, field)
        self.preferred_unit_dict = values_copies
        # print(f"{self.preferred_unit_dict=}")

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print("Leaving the context...")
        # print(f"Before restoring: {PreferredUnits=} ")
        # print(exc_type, exc_value, exc_tb, sep="\n")
        py_ballisticcalc.PreferredUnits.set(**self.preferred_unit_dict)
        # print(f"After restoring: {PreferredUnits=} ")


def create_zero_velocity_zero_min_altitude_calc(
    engine_name, method, max_iteration: int = 60
):
    config = py_ballisticcalc.SciPyEngineConfigDict(
        cMinimumVelocity=0,
        cMinimumAltitude=0,
        integration_method=method,
        cMaxIterations=max_iteration,
    )
    return py_ballisticcalc.Calculator(config, engine=engine_name)


@pytest.fixture(scope="session")
def scipy_calc():
    return create_zero_velocity_zero_min_altitude_calc("scipy_engine", "RK45")


def create_23_mm_shot():
    drag_model = py_ballisticcalc.DragModel(
        bc=0.759,
        drag_table=py_ballisticcalc.TableG1,
        weight=py_ballisticcalc.Weight.Gram(108),  # noqa: F821
        diameter=py_ballisticcalc.Distance.Millimeter(23),
        length=py_ballisticcalc.Distance.Millimeter(108.2),
    )
    ammo = py_ballisticcalc.Ammo(drag_model, py_ballisticcalc.Velocity.MPS(930))
    gun = py_ballisticcalc.Weapon()
    shot = py_ballisticcalc.Shot(
        weapon=gun,
        ammo=ammo,
    )
    return shot

def create_0_308_caliber_shot():
    drag_model = py_ballisticcalc.DragModel(
        bc=0.233,
        drag_table=py_ballisticcalc.TableG7,
        weight=py_ballisticcalc.Weight.Grain(155),
        diameter=py_ballisticcalc.Distance.Inch(0.308),
        length=py_ballisticcalc.Distance.Inch(1.2),
    )
    ammo = py_ballisticcalc.Ammo(drag_model, py_ballisticcalc.Velocity.MPS(914.6))
    gun = py_ballisticcalc.Weapon()
    shot = py_ballisticcalc.Shot(
        weapon=gun,
        ammo=ammo,
    )
    return shot


def find_max_height_robust(scipy_calc, shot_factory):
    shot = shot_factory()
    shot.relative_angle = py_ballisticcalc.Angular.Degree(90)
    try:
        hit_result = scipy_calc.fire(
            shot, py_ballisticcalc.Distance.Meter(1), time_step=0.001, extra_data=True
        )
    except py_ballisticcalc.RangeError as e:
        print(f"Range error et vertical shot {e}")
        hit_result = py_ballisticcalc.HitResult(shot, e.incomplete_trajectory, extra=True)
    apex_point = hit_result.flag(py_ballisticcalc.TrajFlag.APEX)
    return apex_point.height


def get_calc_step(scipy_calc):
    calc_step = scipy_calc._engine_instance.get_calc_step()
    cals_step_meters = py_ballisticcalc.Distance.Foot(calc_step) >> py_ballisticcalc.Distance.Meter
    print(f"{calc_step=} {cals_step_meters=}")
    assert cals_step_meters == pytest.approx(0.07619999999999999, abs=0.01)
    return cals_step_meters


def check_shot_out_of_range(scipy_calc, shot_factory, point_x, point_y):
    look_angle_in_degrees = math.degrees(math.atan2(point_y, point_x))
    shot = shot_factory()
    shot.look_angle = py_ballisticcalc.Angular.Degree(look_angle_in_degrees)
    distance_in_meters = math.sqrt(point_x**2 + point_y**2)
    # both find_zero_angle and zero_angle should throw OutOfRangeError
    with pytest.raises(py_ballisticcalc.OutOfRangeError):
        scipy_calc._engine_instance.find_zero_angle(
            shot, py_ballisticcalc.Distance.Meter(distance_in_meters)
        )
    with pytest.raises(py_ballisticcalc.OutOfRangeError):
        scipy_calc._engine_instance.zero_angle(shot, py_ballisticcalc.Distance.Meter(distance_in_meters))


def check_shot_result_equivalent(scipy_calc, shot_factory, point_x, point_y):
    distance, look_angle = compute_look_angle_and_distance_for_point(point_x, point_y)
    print(
        f"{shot_factory.__name__} - point: {point_x=} {point_y=} {distance>>py_ballisticcalc.Distance.Meter=} {look_angle>>py_ballisticcalc.Angular.Degree=}"
    )
    shot = shot_factory()
    shot.look_angle = look_angle

    # both find_zero_angle and zero_angle should throw OutOfRangeError
    find_zero_angle_error_flag = False
    find_zero_angle_error = None
    zero_angle_error_flag = False
    zero_angle_error = None
    find_zero_angle_result_degrees = float("NaN")
    zero_angle_result_degrees = float("NaN")

    try:
        find_zero_angle_result = scipy_calc._engine_instance.find_zero_angle(
            shot, distance
        )
        find_zero_angle_result_degrees = find_zero_angle_result >> py_ballisticcalc.Angular.Degree
        print(f"{find_zero_angle_result_degrees=}")
    except (py_ballisticcalc.OutOfRangeError, py_ballisticcalc.ZeroFindingError) as e:
        print(f"find_zero_angle throws error {e} for point ({point_x=}, {point_y=})")
        find_zero_angle_error_flag = True
        find_zero_angle_error = e
    try:
        zero_angle_result = scipy_calc._engine_instance.zero_angle(shot, distance)
        zero_angle_result_degrees = zero_angle_result >> py_ballisticcalc.Angular.Degree
        print(f"{zero_angle_result_degrees=}")
    except (py_ballisticcalc.OutOfRangeError, ZeroDivisionError) as e:
        print(f"zero_angle throws error {e} for point ({point_x=}, {point_y=})")
        zero_angle_error_flag = True
        zero_angle_error = e
    if not find_zero_angle_error_flag and not zero_angle_error_flag:
        assert find_zero_angle_result_degrees == pytest.approx(
            zero_angle_result_degrees, abs=ANGLE_EPSILON_IN_DEGREES
        )
    else:
        assert find_zero_angle_error_flag == zero_angle_error_flag, (
            f"Error should occur both in find_zero_error:{find_zero_angle_error_flag=} and zero_angle:{zero_angle_error_flag=}"
        )
        assert type(find_zero_angle_error) == type(zero_angle_error), (
            f"Type of error should match: {find_zero_angle_error=} {zero_angle_error=}"
        )


def check_shot_angle_equals(scipy_calc, shot_factory, point_x, point_y):
    distance, look_angle = compute_look_angle_and_distance_for_point(point_x, point_y)
    print(f"{distance>>py_ballisticcalc.Distance.Meter=} {look_angle>>py_ballisticcalc.Angular.Degree=}")
    shot = shot_factory()
    shot.look_angle = look_angle

    # both find_zero_angle and zero_angle should throw OutOfRangeError
    unexpected_find_zero_angle_error_flag = False
    unexpected_find_zero_angle_error = None
    unexpected_zero_angle_error_flag = False
    unexpected_zero_angle_error = None
    find_zero_angle_result_degrees = float("NaN")
    zero_angle_result_degrees = float("NaN")

    try:
        find_zero_angle_result = scipy_calc._engine_instance.find_zero_angle(
            shot, distance
        )
        find_zero_angle_result_degrees = find_zero_angle_result >> py_ballisticcalc.Angular.Degree
        print(f"{find_zero_angle_result_degrees=}")
    except (py_ballisticcalc.OutOfRangeError, py_ballisticcalc.ZeroFindingError) as e:
        print(f"find_zero_angle raised unexpected {e}")
        unexpected_find_zero_angle_error_flag = True
        unexpected_find_zero_angle_error = e
    try:
        zero_angle_result = scipy_calc._engine_instance.zero_angle(shot, distance)
        zero_angle_result_degrees = zero_angle_result >> py_ballisticcalc.Angular.Degree
        print(f"{zero_angle_result_degrees=}")
    except (py_ballisticcalc.OutOfRangeError, py_ballisticcalc.ZeroFindingError) as e:
        print(f"zero_angle raised unexpected {e}")
        unexpected_zero_angle_error_flag = True
        unexpected_zero_angle_error = e
    assert (not unexpected_find_zero_angle_error_flag) and (
        not unexpected_zero_angle_error_flag
    ), f"{unexpected_find_zero_angle_error_flag=} != {unexpected_zero_angle_error_flag}"
    # at this point either both errors are present or both are absent
    if unexpected_find_zero_angle_error:
        assert type(unexpected_find_zero_angle_error) == type(
            unexpected_zero_angle_error
        )
    else:
        assert find_zero_angle_result_degrees == pytest.approx(
            zero_angle_result_degrees, abs=ANGLE_EPSILON_IN_DEGREES
        )


def compute_look_angle_and_distance_for_point(
    point_x_in_meters: float, point_y_in_meters: float
):
    look_angle_in_degrees = math.degrees(
        math.atan2(point_y_in_meters, point_x_in_meters)
    )
    distance_in_meters = (point_x_in_meters**2 + point_y_in_meters**2) ** 0.5
    look_angle = py_ballisticcalc.Angular.Degree(look_angle_in_degrees)
    distance = py_ballisticcalc.Distance.Meter(distance_in_meters)
    # print(f'{point_x_in_meters=} {point_y_in_meters=} {sight_height_meters=} {sight_height_y=} {height_diff=} {look_angle=} {distance_in_meters=}')
    return distance, look_angle


def check_almost_max_height(scipy_calc, shot_factory):
    cals_step_meters = get_calc_step(scipy_calc)
    max_height = find_max_height_robust(scipy_calc, shot_factory)
    max_height_in_meters = max_height >> py_ballisticcalc.Distance.Meter
    point_x = cals_step_meters
    point_y = max_height_in_meters
    check_shot_out_of_range(scipy_calc, shot_factory, point_x, point_y)


def check_reachable_almost_max_height(scipy_calc, shot_factory):
    cals_step_meters = get_calc_step(scipy_calc)
    assert cals_step_meters == pytest.approx(0.07619999999999999, abs=0.01)
    max_height = find_max_height_robust(scipy_calc, shot_factory)
    max_height_in_meters = max_height >> py_ballisticcalc.Distance.Meter
    diffs = [2, 8.7, 10, 22]
    for diff in diffs:
        point_x = cals_step_meters
        point_y = max_height_in_meters - diff
        check_shot_result_equivalent(scipy_calc, shot_factory, point_x, point_y)


def check_find_angles_max_range(scipy_calc, shot_factory):
    shot = shot_factory()
    max_range, max_range_launch_angle = scipy_calc._engine_instance.find_max_range(shot)
    max_range_launch_angle_in_degrees = max_range_launch_angle >> py_ballisticcalc.Angular.Degree
    max_range_in_meters = max_range >> py_ballisticcalc.Distance.Meter
    print(
        f"for {shot_factory.__name__} { max_range_in_meters=} { max_range_launch_angle_in_degrees=}"
    )
    zero_angle_shot = shot_factory()
    zero_launch_angle = scipy_calc.set_weapon_zero(zero_angle_shot, max_range)
    zero_launch_angle_in_degrees = zero_launch_angle >> py_ballisticcalc.Angular.Degree
    print(
        f"Zero for { max_range_in_meters=} m is elevation={zero_launch_angle_in_degrees} degree"
    )
    assert max_range_launch_angle_in_degrees == pytest.approx(
        zero_launch_angle_in_degrees, abs=ANGLE_EPSILON_IN_DEGREES
    )
    find_zero_angle_shot = shot_factory()
    find_zero_max_launch_angle = scipy_calc.find_zero_angle(find_zero_angle_shot, max_range)
    find_zero_max_launch_angle_in_degrees = find_zero_max_launch_angle >> py_ballisticcalc.Angular.Degree
    assert find_zero_max_launch_angle_in_degrees == pytest.approx(
        max_range_launch_angle_in_degrees, abs=ANGLE_EPSILON_IN_DEGREES
    )
    shot = shot_factory()
    shot.relative_angle = Angular.Degree(max_range_launch_angle_in_degrees)
    print(f'{max_range>>Distance.Meter=} m')
    extra = False
    try:
        hit_results = scipy_calc.fire(shot, max_range, extra_data=extra)
    except RangeError as e:
        # assert False
        print(f'Got range error {e} for shot {shot_factory.__name__} distance: {max_range>>Distance.Meter} m')
        hit_results = HitResult(shot, e.incomplete_trajectory, extra=extra)
    print_out_trajectory_compact(hit_results)
    hit_distance_meters = (hit_results[-1].distance >> Distance.Meter)
    hit_height_meters = (hit_results[-1].height >> Distance.Meter)

    assert abs(hit_distance_meters-(max_range>>Distance.Meter))<0.01
    assert abs(hit_height_meters)<0.01


def check_handling_zero_point(scipy_calc, shot_factory):
    check_shot_angle_equals(scipy_calc, shot_factory, 0, 0)


def check_calc_same_step_shots(scipy_calc, shot_factory, step):
    check_shot_angle_equals(scipy_calc, shot_factory, step, step)


def test_find_max_height_23_mm(scipy_calc):
    shot_factory = create_23_mm_shot
    shot = shot_factory()

    shot.look_angle = py_ballisticcalc.Angular.Degree(90)
    max_height, max_angle = scipy_calc._engine_instance.find_max_range(shot)
    max_height_in_meters_1 = max_height >> py_ballisticcalc.Distance.Meter
    print(f"1. { max_height_in_meters_1=} {max_angle>>py_ballisticcalc.Angular.Degree=}")

    max_height_2 = find_max_height_robust(scipy_calc, shot_factory)
    max_height_in_meters_2 = max_height_2 >> py_ballisticcalc.Distance.Meter
    print(f"2. {max_height_in_meters_2=}")
    assert max_height_in_meters_1 == max_height_in_meters_2


def test_find_max_range_23_mm(scipy_calc):
    shot_factory = create_23_mm_shot
    shot = shot_factory()
    max_range, max_range_launch_angle = scipy_calc._engine_instance.find_max_range(shot)
    max_range_launch_angle_in_degrees = max_range_launch_angle >> py_ballisticcalc.Angular.Degree
    max_range_in_meters = max_range >> py_ballisticcalc.Distance.Meter
    print(f"{max_range_in_meters=} {max_range_launch_angle_in_degrees=}")
    assert max_range_in_meters == pytest.approx(7136.255, abs=0.01)
    assert max_range_launch_angle_in_degrees == pytest.approx(
        38.8236, abs=ANGLE_EPSILON_IN_DEGREES
    )



def test_7_62_point_mismatch(scipy_calc):
    point_x = 67.79956401327206
    point_y = 83.47708433996526
    shot = create_7_62_mm_shot_neg_sight_height()
    look_angle_wo_sight_degrees = math.degrees(math.atan2(point_y, point_x))
    sight_height_meters = shot.weapon.sight_height >> py_ballisticcalc.Distance.Meter
    height_diff_target_trajectory_start = point_y + sight_height_meters
    angle_from_start_degrees = math.degrees(
        math.atan2(height_diff_target_trajectory_start, point_x)
    )
    print(
        f"{point_x=} {point_y=} {sight_height_meters=} { height_diff_target_trajectory_start=}"
    )
    print(f"{look_angle_wo_sight_degrees=} {angle_from_start_degrees=}")
    print(f"{(point_x**2+height_diff_target_trajectory_start**2)**0.5=}")
    check_shot_angle_equals(scipy_calc, create_7_62_mm_shot_neg_sight_height, point_x, point_y)


def create_7_62_mm_shot_neg_sight_height_positive_altitude():
    diameter = Distance.Millimeter(7.62)
    length: Distance = Distance.Millimeter(32.5628)
    weight = Weight.Grain(180)
    dm = DragModel(bc=0.2860,
                   drag_table=TableG1,
                   weight=weight,
                   diameter=diameter,
                   length=length)


    ammo = Ammo(dm, mv=Velocity.MPS(800), powder_temp=Temperature.Celsius(20), use_powder_sensitivity=False)
    ammo.calc_powder_sens(other_velocity=Velocity.MPS(805), other_temperature=Temperature.Celsius(15))
    current_atmo = Atmo(altitude=Distance.Meter(400), pressure=Pressure.hPa(967.24), temperature=Temperature.Celsius(20),
                        humidity=60)
    gun = Weapon(sight_height=Distance.Millimeter(-100), twist=Distance.Millimeter(300))
    current_winds = [Wind(0, 0)]
    new_shot = Shot(weapon=gun, ammo=ammo, atmo=current_atmo,
                    winds=current_winds)  # Copy the zero properties; NB: Not a deepcopy!
    return new_shot


def create_7_62_mm_shot_neg_sight_height():
    diameter = py_ballisticcalc.Distance.Millimeter(7.62)
    length: py_ballisticcalc.Distance = py_ballisticcalc.Distance.Millimeter(32.5628)
    weight = py_ballisticcalc.Weight.Grain(180)
    dm = py_ballisticcalc.DragModel(
        bc=0.2860, drag_table=py_ballisticcalc.TableG1, weight=weight, diameter=diameter, length=length
    )

    ammo = py_ballisticcalc.Ammo(dm, mv=py_ballisticcalc.Velocity.MPS(800),
                                 powder_temp=py_ballisticcalc.Temperature.Celsius(20))
    gun = py_ballisticcalc.Weapon(sight_height=py_ballisticcalc.Distance.Millimeter(-100), twist=py_ballisticcalc.Distance.Millimeter(300))
    new_shot = py_ballisticcalc.Shot(
        weapon=gun, ammo=ammo
    )
    return new_shot

def create_nato_7_62_mm():
    # 7.62x51mm NATO M118
    diameter = py_ballisticcalc.Distance.Millimeter(7.62)
    length: py_ballisticcalc.Distance = py_ballisticcalc.Distance.Millimeter(32.0)
    weight = py_ballisticcalc.Weight.Grain(175)
    dm = py_ballisticcalc.DragModel(
        bc=0.243,
        drag_table=py_ballisticcalc.TableG7,
        weight=weight,
        diameter=diameter,
        length=length,
    )
    ammo = py_ballisticcalc.Ammo(dm, mv=py_ballisticcalc.Velocity.MPS(800))
    gun = py_ballisticcalc.Weapon()
    shot = py_ballisticcalc.Shot(
        weapon=gun, ammo=ammo
    )
    return shot


def create_nato_5_56_mm_shot_pos_sight_height():
    # 5.56x45mm NATO SS109
    diameter = py_ballisticcalc.Distance.Millimeter(5.56)
    length: py_ballisticcalc.Distance = py_ballisticcalc.Distance.Millimeter(21.0)
    weight = py_ballisticcalc.Weight.Grain(62)
    dm = py_ballisticcalc.DragModel(
        bc=0.1510, drag_table=py_ballisticcalc.TableG7, weight=weight, diameter=diameter, length=length
    )
    ammo = py_ballisticcalc.Ammo(dm, mv=py_ballisticcalc.Velocity.MPS(900))
    gun = py_ballisticcalc.Weapon(sight_height=Distance.Centimeter(9))
    shot = py_ballisticcalc.Shot(
        weapon=gun,
        ammo=ammo,
    )
    return shot


def get_meter_coords(point: py_ballisticcalc.TrajectoryData)->tuple[float, float]:
    return (point.distance >> py_ballisticcalc.Distance.Meter, point.height >> py_ballisticcalc.Distance.Meter)


def point_distance(p, point_x, point_y):
    p_coords = get_meter_coords(p)
    distance = ((p_coords[0] - point_x) ** 2 + (p_coords[1] - point_y) ** 2) ** 0.5
    return distance

def find_min_dev_point(hit_result, point_x, point_y):
    trajectory = hit_result.trajectory
    # print(f"({apex_point.distance>>Distance.Meter} {apex_point.height>>Distance.Meter}) "
    #      f"({ascending_traj_part[-1].distance>>Distance.Meter} {ascending_traj_part[-1].height>>Distance.Meter})")
    min_dist = float('inf')
    min_index = None
    for p_index, p in enumerate(trajectory):
        distance = point_distance(p, point_x, point_y)
        if distance < min_dist:
            min_index = p_index
            min_dist = distance
    min_dev_point = trajectory[min_index]
    return min_dev_point

TESTED_SHOTS = [create_23_mm_shot, create_0_308_caliber_shot, create_7_62_mm_shot_neg_sight_height,
                create_7_62_mm_shot_neg_sight_height_positive_altitude,
                create_nato_7_62_mm, create_nato_5_56_mm_shot_pos_sight_height]
#SMALL_TESTED_SHOTS = [create_23_mm_shot]
TESTED_ANGLES = list(range(0, 91, 1))

@pytest.mark.skip(reason="very expensive to run. Run them by significant changes to max range")
@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
@pytest.mark.parametrize("look_angle_degrees", TESTED_ANGLES)
def test_find_max_range_different_angles(shot_factory, look_angle_degrees, scipy_calc):
    shot = shot_factory()
    shot.look_angle = py_ballisticcalc.Angular.Degree(look_angle_degrees)
    max_range, max_angle = scipy_calc._engine_instance.find_max_range(shot)
    max_slant_range_in_meters = max_range >> py_ballisticcalc.Distance.Meter
    max_horizontal_range_meters = max_slant_range_in_meters*math.cos(math.radians(look_angle_degrees))
    max_range_point_height_meters = max_slant_range_in_meters*math.sin(math.radians(look_angle_degrees))
    print(f'{look_angle_degrees=} { max_slant_range_in_meters=}  {max_angle>>py_ballisticcalc.Angular.Degree=} {max_horizontal_range_meters=} {max_range_point_height_meters}')
    other_shot = shot_factory()
    other_shot.relative_angle = max_angle
    try:
        hit_result= scipy_calc.fire(other_shot, py_ballisticcalc.Distance.Meter(max_horizontal_range_meters if look_angle_degrees != 90 else 1), time_step=0.001, extra_data=True)
    except py_ballisticcalc.RangeError as e:
        print(f'Got e{e}')
        hit_result = py_ballisticcalc.HitResult(other_shot, e.incomplete_trajectory, extra=True)

    print_out_trajectory_compact(hit_result)
    min_dev_point = find_min_dev_point(hit_result,  max_horizontal_range_meters, max_range_point_height_meters)
    distance_min_point_max_range_point_meters = point_distance(min_dev_point, max_horizontal_range_meters, max_range_point_height_meters)
    print(f'distance: {distance_min_point_max_range_point_meters} coords: {get_meter_coords(min_dev_point)}'
          f'expected coords ({max_horizontal_range_meters} {max_range_point_height_meters})  Min Point: {min_dev_point=}')
    assert distance_min_point_max_range_point_meters<0.01

    other_shot: py_ballisticcalc.Shot = shot_factory()
    other_shot.look_angle = Angular.Degree(look_angle_degrees)
    other_shot.weapon.zero_elevation = Angular.Radian(
        (max_angle >> Angular.Radian) - math.radians(look_angle_degrees)
    )
    extra_data_flag = False
    try:
        hit_result = scipy_calc.fire(other_shot, max_range, time_step=0.001, extra_data=extra_data_flag)
    except py_ballisticcalc.RangeError as e:
        print(f'Got e{e} for zero_elevation_shot')
        hit_result = HitResult(other_shot, e.incomplete_trajectory, extra=extra_data_flag)

    min_dev_point = find_min_dev_point(hit_result,  max_horizontal_range_meters, max_range_point_height_meters)
    distance_min_point_max_range_point_meters = point_distance(min_dev_point, max_horizontal_range_meters, max_range_point_height_meters)
    print(f'distance: {distance_min_point_max_range_point_meters} coords: {get_meter_coords(min_dev_point)}'
          f'expected coords ({max_horizontal_range_meters} {max_range_point_height_meters})  Min Point: {min_dev_point=}')
    print_out_trajectory_compact(hit_result)
    #assert distance_min_point_max_range_point_meters<0.01
    assert distance_min_point_max_range_point_meters<0.1

def test_danger_space_shot(scipy_calc):
    shot = create_danger_space_shot_with_wind()
    elevation = scipy_calc.set_weapon_zero(shot, Distance.Foot(300))
    print(f'{elevation>>Angular.Degree}')
    assert pytest.approx(elevation>>Angular.Degree, abs=0.001)==0.054


def create_danger_space_shot_with_wind():
    dm = DragModel(
        0.223, TableG7, Weight.Grain(168), Distance.Inch(0.308), Distance.Inch(1.282)
    )
    ammo = Ammo(dm, Velocity.FPS(2750), Temperature.Celsius(15))
    ammo.calc_powder_sens(Velocity.FPS(2723), Temperature.Celsius(0))
    current_winds = [Wind(Velocity.FPS(2), Angular.Degree(90))]
    shot = Shot(weapon=Weapon(sight_height=Distance.Inch(1)), ammo=ammo, winds=current_winds)
    return shot


@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_set_weapon_zero_max_range(shot_factory, scipy_calc):
    with PreferredUnitsContextManager():
        loadMetricUnits()
        check_find_angles_max_range(scipy_calc, shot_factory)


@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_find_almost_max_height(shot_factory, scipy_calc):
    check_almost_max_height(scipy_calc, shot_factory)

def test_length_of_shot(scipy_calc):
    with PreferredUnitsContextManager():
        py_ballisticcalc.loadMetricUnits()

        shot_factory = create_nato_5_56_mm_shot_pos_sight_height
        tested_shot = shot_factory()
        #tested_shot.zero_angle = Angular.Degree(89.99828858095566)
        target_x_ft = 0.25000000000092704
        target_y_ft = target_x_ft*math.tan(tested_shot.look_angle>>Angular.Radian)
        print(f'{target_y_ft=} {Distance.Foot(target_y_ft)>>Distance.Meter=}')

        distance = Distance.Meter(2551.0633494297517)
        scipy_calc._init_zero_calculation(tested_shot, distance)
        tested_shot.look_angle = Angular.Degree(89.99828858095566)
        hit_result = scipy_calc._engine_instance._integrate(tested_shot, target_x_ft, target_x_ft, TrajFlag.NONE, stop_at_zero=True)
        print(f"{len(hit_result)=}")
        #hit_result = scipy_calc._engine_instance._integrate(
        #    tested_shot, target_x_ft, target_x_ft, TrajFlag.NONE, stop_at_zero=True
        #)
        print_out_trajectory_list_compact(hit_result, len(hit_result), "hit_result")
        assert len(hit_result)>0

@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_handling_zero_point(shot_factory, scipy_calc):
    check_handling_zero_point(scipy_calc, shot_factory)


@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_reachable_almost_max_height(shot_factory, scipy_calc):
    with PreferredUnitsContextManager():
        py_ballisticcalc.loadMetricUnits()
        # print(f"{PreferredUnits=}")
        check_reachable_almost_max_height(scipy_calc, shot_factory)

EXCEPTION_CASES = [
    ((7071.630552600922, 153.97926113995862), create_23_mm_shot),
    ((4691.07221312, 105.36063621068715), create_0_308_caliber_shot),
    ((3749.876564130784, 0), create_7_62_mm_shot_neg_sight_height_positive_altitude),
    ((3574.6008249856727, 24.05582994099316), create_7_62_mm_shot_neg_sight_height),
    ((3600.140741624915, 14.463475504373086), create_7_62_mm_shot_neg_sight_height),
]
@pytest.mark.parametrize("exception_data",EXCEPTION_CASES)
def test_exception_case(exception_data, scipy_calc):
    with PreferredUnitsContextManager():
        py_ballisticcalc.loadMetricUnits()

        point, shot_factory = exception_data
        check_shot_result_equivalent(scipy_calc, shot_factory, point[0], point[1])

def test_0_308_runtime_warning_point(scipy_calc):
    with PreferredUnitsContextManager():
        py_ballisticcalc.loadMetricUnits()
        # this point leads to the situation, when one gets division by zero
        # and NaN correction
        check_shot_out_of_range(
            scipy_calc, create_0_308_caliber_shot, 4733.942049290925, 3463.4247025051754
        )


@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_calc_step_point_shots(shot_factory, scipy_calc):
    calc_step = get_calc_step(scipy_calc)
    check_calc_same_step_shots(scipy_calc, shot_factory, calc_step)


@pytest.mark.parametrize("shot_factory", TESTED_SHOTS)
def test_half_calc_step_point_shots(shot_factory, scipy_calc):
    calc_step = get_calc_step(scipy_calc)
    check_calc_same_step_shots(scipy_calc, shot_factory, calc_step / 2)
