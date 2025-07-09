#include "types.h"

/**
 * @brief Litz spin-drift approximation
 * @param shot_data_ptr Pointer to ShotData_t containing shot parameters.
 * @param time Time of flight in seconds.
 * @return Windage due to spin drift, in feet.
 */
double ShotT_spin_drift(const ShotData_t *shot_data_ptr, double time) {
    double sign;

    // Check if twist and stability_coefficient are non-zero.
    // In C, comparing doubles directly to 0 can sometimes be problematic due to
    // floating-point precision. However, for typical use cases here, direct
    // comparison with 0 is often acceptable if the values are expected to be
    // exactly 0 or significantly non-zero. If extreme precision is needed for
    // checking "effectively zero", you might use an epsilon (e.g., fabs(val) > EPSILON).
    if (shot_data_ptr->twist != 0 && shot_data_ptr->stability_coefficient != 0) {
        // Determine the sign based on twist direction.
        if (shot_data_ptr->twist > 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
        }
        // Calculate the spin drift using the Litz approximation formula.
        // The division by 12 converts the result from inches (implied by Litz formula) to feet.
        return sign * (1.25 * (shot_data_ptr->stability_coefficient + 1.2) * pow(time, 1.83)) / 12.0;
    }
    // If either twist or stability_coefficient is zero, return 0.
    return 0.0;
}

int ShotT_update_stability_coefficient(ShotData_t *shot_data_ptr) {
    /* Miller stability coefficient */
    double twist_rate, length, sd, fv, ft, pt, ftp;

    // Check for non-zero or valid input values before calculation
    if (shot_data_ptr->twist != 0.0 &&
        shot_data_ptr->length != 0.0 &&
        shot_data_ptr->diameter != 0.0 &&
        shot_data_ptr->atmo._p0 != 0.0)
    {
        twist_rate = fabs(shot_data_ptr->twist) / shot_data_ptr->diameter;
        length = shot_data_ptr->length / shot_data_ptr->diameter;

        // Ensure denominator components are non-zero to avoid division by zero
        // This check is crucial for robustness in C
        double denom_part1 = pow(twist_rate, 2);
        double denom_part2 = pow(shot_data_ptr->diameter, 3);
        double denom_part3 = length;
        double denom_part4 = (1 + pow(length, 2));

        if (denom_part1 != 0.0 && denom_part2 != 0.0 && denom_part3 != 0.0 && denom_part4 != 0.0) {
            sd = 30.0 * shot_data_ptr->weight / (denom_part1 * denom_part2 * denom_part3 * denom_part4);
        } else {
            shot_data_ptr->stability_coefficient = 0.0;
            return -1; // Exit if denominator is zero
        }

        fv = pow(shot_data_ptr->muzzle_velocity / 2800.0, 1.0 / 3.0);
        ft = (shot_data_ptr->atmo._t0 * 9.0 / 5.0) + 32.0;  // Convert from Celsius to Fahrenheit
        pt = shot_data_ptr->atmo._p0 / 33.8639;  // Convert hPa to inHg

        // Ensure pt is not zero before division
        if (pt != 0.0) {
            ftp = ((ft + 460.0) / (59.0 + 460.0)) * (29.92 / pt);
        } else {
            shot_data_ptr->stability_coefficient = 0.0;
            return -1; // Exit if pt is zero
        }

        shot_data_ptr->stability_coefficient = sd * fv * ftp;
    } else {
        shot_data_ptr->stability_coefficient = 0.0;
    }
    return 0;
}