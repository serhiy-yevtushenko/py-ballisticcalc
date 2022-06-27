import math


# identifiers for ballistic table type
DragTableG1 = 1
DragTableG2 = 2
DragTableG5 = 3
DragTableG6 = 4
DragTableG7 = 5
DragTableG8 = 6
DragTableGS = 7
DragTableGI = 8

"""
BallisticCoefficient keeps data about ballistic coefficient
of a projectile

The ballistic coefficient (BC) of a body is a measure of its
ability to overcome air resistance in flight.

The small arm ballistics, BC is expressed vs
a standard projectile. Different ballistic tables
uses different standard projectiles, for example G1 uses
flat based 2 caliber length with a 2 caliber give

G1 and G7 are the most used for small arms ballistics
"""
DRAG_TABLES = {

    DragTableG1: {
        {'A': 0.00, 'B': 0.2629},
        {'A': 0.05, 'B': 0.2558},
        {'A': 0.10, 'B': 0.2487},
        {'A': 0.15, 'B': 0.2413},
        {'A': 0.20, 'B': 0.2344},
        {'A': 0.25, 'B': 0.2278},
        {'A': 0.30, 'B': 0.2214},
        {'A': 0.35, 'B': 0.2155},
        {'A': 0.40, 'B': 0.2104},
        {'A': 0.45, 'B': 0.2061},
        {'A': 0.50, 'B': 0.2032},
        {'A': 0.55, 'B': 0.2020},
        {'A': 0.60, 'B': 0.2034},
        {'A': 0.70, 'B': 0.2165},
        {'A': 0.725, 'B': 0.2230},
        {'A': 0.75, 'B': 0.2313},
        {'A': 0.775, 'B': 0.2417},
        {'A': 0.80, 'B': 0.2546},
        {'A': 0.825, 'B': 0.2706},
        {'A': 0.85, 'B': 0.2901},
        {'A': 0.875, 'B': 0.3136},
        {'A': 0.90, 'B': 0.3415},
        {'A': 0.925, 'B': 0.3734},
        {'A': 0.95, 'B': 0.4084},
        {'A': 0.975, 'B': 0.4448},
        {'A': 1.0, 'B': 0.4805},
        {'A': 1.025, 'B': 0.5136},
        {'A': 1.05, 'B': 0.5427},
        {'A': 1.075, 'B': 0.5677},
        {'A': 1.10, 'B': 0.5883},
        {'A': 1.125, 'B': 0.6053},
        {'A': 1.15, 'B': 0.6191},
        {'A': 1.20, 'B': 0.6393},
        {'A': 1.25, 'B': 0.6518},
        {'A': 1.30, 'B': 0.6589},
        {'A': 1.35, 'B': 0.6621},
        {'A': 1.40, 'B': 0.6625},
        {'A': 1.45, 'B': 0.6607},
        {'A': 1.50, 'B': 0.6573},
        {'A': 1.55, 'B': 0.6528},
        {'A': 1.60, 'B': 0.6474},
        {'A': 1.65, 'B': 0.6413},
        {'A': 1.70, 'B': 0.6347},
        {'A': 1.75, 'B': 0.6280},
        {'A': 1.80, 'B': 0.6210},
        {'A': 1.85, 'B': 0.6141},
        {'A': 1.90, 'B': 0.6072},
        {'A': 1.95, 'B': 0.6003},
        {'A': 2.00, 'B': 0.5934},
        {'A': 2.05, 'B': 0.5867},
        {'A': 2.10, 'B': 0.5804},
        {'A': 2.15, 'B': 0.5743},
        {'A': 2.20, 'B': 0.5685},
        {'A': 2.25, 'B': 0.5630},
        {'A': 2.30, 'B': 0.5577},
        {'A': 2.35, 'B': 0.5527},
        {'A': 2.40, 'B': 0.5481},
        {'A': 2.45, 'B': 0.5438},
        {'A': 2.50, 'B': 0.5397},
        {'A': 2.60, 'B': 0.5325},
        {'A': 2.70, 'B': 0.5264},
        {'A': 2.80, 'B': 0.5211},
        {'A': 2.90, 'B': 0.5168},
        {'A': 3.00, 'B': 0.5133},
        {'A': 3.10, 'B': 0.5105},
        {'A': 3.20, 'B': 0.5084},
        {'A': 3.30, 'B': 0.5067},
        {'A': 3.40, 'B': 0.5054},
        {'A': 3.50, 'B': 0.5040},
        {'A': 3.60, 'B': 0.5030},
        {'A': 3.70, 'B': 0.5022},
        {'A': 3.80, 'B': 0.5016},
        {'A': 3.90, 'B': 0.5010},
        {'A': 4.00, 'B': 0.5006},
        {'A': 4.20, 'B': 0.4998},
        {'A': 4.40, 'B': 0.4995},
        {'A': 4.60, 'B': 0.4992},
        {'A': 4.80, 'B': 0.4990},
        {'A': 5.00, 'B': 0.4988},
    },

    DragTableG7: {
        {'A': 0.00, 'B': 0.1198},
        {'A': 0.05, 'B': 0.1197},
        {'A': 0.10, 'B': 0.1196},
        {'A': 0.15, 'B': 0.1194},
        {'A': 0.20, 'B': 0.1193},
        {'A': 0.25, 'B': 0.1194},
        {'A': 0.30, 'B': 0.1194},
        {'A': 0.35, 'B': 0.1194},
        {'A': 0.40, 'B': 0.1193},
        {'A': 0.45, 'B': 0.1193},
        {'A': 0.50, 'B': 0.1194},
        {'A': 0.55, 'B': 0.1193},
        {'A': 0.60, 'B': 0.1194},
        {'A': 0.65, 'B': 0.1197},
        {'A': 0.70, 'B': 0.1202},
        {'A': 0.725, 'B': 0.1207},
        {'A': 0.75, 'B': 0.1215},
        {'A': 0.775, 'B': 0.1226},
        {'A': 0.80, 'B': 0.1242},
        {'A': 0.825, 'B': 0.1266},
        {'A': 0.85, 'B': 0.1306},
        {'A': 0.875, 'B': 0.1368},
        {'A': 0.90, 'B': 0.1464},
        {'A': 0.925, 'B': 0.1660},
        {'A': 0.95, 'B': 0.2054},
        {'A': 0.975, 'B': 0.2993},
        {'A': 1.0, 'B': 0.3803},
        {'A': 1.025, 'B': 0.4015},
        {'A': 1.05, 'B': 0.4043},
        {'A': 1.075, 'B': 0.4034},
        {'A': 1.10, 'B': 0.4014},
        {'A': 1.125, 'B': 0.3987},
        {'A': 1.15, 'B': 0.3955},
        {'A': 1.20, 'B': 0.3884},
        {'A': 1.25, 'B': 0.3810},
        {'A': 1.30, 'B': 0.3732},
        {'A': 1.35, 'B': 0.3657},
        {'A': 1.40, 'B': 0.3580},
        {'A': 1.50, 'B': 0.3440},
        {'A': 1.55, 'B': 0.3376},
        {'A': 1.60, 'B': 0.3315},
        {'A': 1.65, 'B': 0.3260},
        {'A': 1.70, 'B': 0.3209},
        {'A': 1.75, 'B': 0.3160},
        {'A': 1.80, 'B': 0.3117},
        {'A': 1.85, 'B': 0.3078},
        {'A': 1.90, 'B': 0.3042},
        {'A': 1.95, 'B': 0.3010},
        {'A': 2.00, 'B': 0.2980},
        {'A': 2.05, 'B': 0.2951},
        {'A': 2.10, 'B': 0.2922},
        {'A': 2.15, 'B': 0.2892},
        {'A': 2.20, 'B': 0.2864},
        {'A': 2.25, 'B': 0.2835},
        {'A': 2.30, 'B': 0.2807},
        {'A': 2.35, 'B': 0.2779},
        {'A': 2.40, 'B': 0.2752},
        {'A': 2.45, 'B': 0.2725},
        {'A': 2.50, 'B': 0.2697},
        {'A': 2.55, 'B': 0.2670},
        {'A': 2.60, 'B': 0.2643},
        {'A': 2.65, 'B': 0.2615},
        {'A': 2.70, 'B': 0.2588},
        {'A': 2.75, 'B': 0.2561},
        {'A': 2.80, 'B': 0.2533},
        {'A': 2.85, 'B': 0.2506},
        {'A': 2.90, 'B': 0.2479},
        {'A': 2.95, 'B': 0.2451},
        {'A': 3.00, 'B': 0.2424},
        {'A': 3.10, 'B': 0.2368},
        {'A': 3.20, 'B': 0.2313},
        {'A': 3.30, 'B': 0.2258},
        {'A': 3.40, 'B': 0.2205},
        {'A': 3.50, 'B': 0.2154},
        {'A': 3.60, 'B': 0.2106},
        {'A': 3.70, 'B': 0.2060},
        {'A': 3.80, 'B': 0.2017},
        {'A': 3.90, 'B': 0.1975},
        {'A': 4.00, 'B': 0.1935},
        {'A': 4.20, 'B': 0.1861},
        {'A': 4.40, 'B': 0.1793},
        {'A': 4.60, 'B': 0.1730},
        {'A': 4.80, 'B': 0.1672},
        {'A': 5.00, 'B': 0.1618},
    },

    DragTableG2: {
        {'A': 0.00, 'B': 0.2303},
        {'A': 0.05, 'B': 0.2298},
        {'A': 0.10, 'B': 0.2287},
        {'A': 0.15, 'B': 0.2271},
        {'A': 0.20, 'B': 0.2251},
        {'A': 0.25, 'B': 0.2227},
        {'A': 0.30, 'B': 0.2196},
        {'A': 0.35, 'B': 0.2156},
        {'A': 0.40, 'B': 0.2107},
        {'A': 0.45, 'B': 0.2048},
        {'A': 0.50, 'B': 0.1980},
        {'A': 0.55, 'B': 0.1905},
        {'A': 0.60, 'B': 0.1828},
        {'A': 0.65, 'B': 0.1758},
        {'A': 0.70, 'B': 0.1702},
        {'A': 0.75, 'B': 0.1669},
        {'A': 0.775, 'B': 0.1664},
        {'A': 0.80, 'B': 0.1667},
        {'A': 0.825, 'B': 0.1682},
        {'A': 0.85, 'B': 0.1711},
        {'A': 0.875, 'B': 0.1761},
        {'A': 0.90, 'B': 0.1831},
        {'A': 0.925, 'B': 0.2004},
        {'A': 0.95, 'B': 0.2589},
        {'A': 0.975, 'B': 0.3492},
        {'A': 1.0, 'B': 0.3983},
        {'A': 1.025, 'B': 0.4075},
        {'A': 1.05, 'B': 0.4103},
        {'A': 1.075, 'B': 0.4114},
        {'A': 1.10, 'B': 0.4106},
        {'A': 1.125, 'B': 0.4089},
        {'A': 1.15, 'B': 0.4068},
        {'A': 1.175, 'B': 0.4046},
        {'A': 1.20, 'B': 0.4021},
        {'A': 1.25, 'B': 0.3966},
        {'A': 1.30, 'B': 0.3904},
        {'A': 1.35, 'B': 0.3835},
        {'A': 1.40, 'B': 0.3759},
        {'A': 1.45, 'B': 0.3678},
        {'A': 1.50, 'B': 0.3594},
        {'A': 1.55, 'B': 0.3512},
        {'A': 1.60, 'B': 0.3432},
        {'A': 1.65, 'B': 0.3356},
        {'A': 1.70, 'B': 0.3282},
        {'A': 1.75, 'B': 0.3213},
        {'A': 1.80, 'B': 0.3149},
        {'A': 1.85, 'B': 0.3089},
        {'A': 1.90, 'B': 0.3033},
        {'A': 1.95, 'B': 0.2982},
        {'A': 2.00, 'B': 0.2933},
        {'A': 2.05, 'B': 0.2889},
        {'A': 2.10, 'B': 0.2846},
        {'A': 2.15, 'B': 0.2806},
        {'A': 2.20, 'B': 0.2768},
        {'A': 2.25, 'B': 0.2731},
        {'A': 2.30, 'B': 0.2696},
        {'A': 2.35, 'B': 0.2663},
        {'A': 2.40, 'B': 0.2632},
        {'A': 2.45, 'B': 0.2602},
        {'A': 2.50, 'B': 0.2572},
        {'A': 2.55, 'B': 0.2543},
        {'A': 2.60, 'B': 0.2515},
        {'A': 2.65, 'B': 0.2487},
        {'A': 2.70, 'B': 0.2460},
        {'A': 2.75, 'B': 0.2433},
        {'A': 2.80, 'B': 0.2408},
        {'A': 2.85, 'B': 0.2382},
        {'A': 2.90, 'B': 0.2357},
        {'A': 2.95, 'B': 0.2333},
        {'A': 3.00, 'B': 0.2309},
        {'A': 3.10, 'B': 0.2262},
        {'A': 3.20, 'B': 0.2217},
        {'A': 3.30, 'B': 0.2173},
        {'A': 3.40, 'B': 0.2132},
        {'A': 3.50, 'B': 0.2091},
        {'A': 3.60, 'B': 0.2052},
        {'A': 3.70, 'B': 0.2014},
        {'A': 3.80, 'B': 0.1978},
        {'A': 3.90, 'B': 0.1944},
        {'A': 4.00, 'B': 0.1912},
        {'A': 4.20, 'B': 0.1851},
        {'A': 4.40, 'B': 0.1794},
        {'A': 4.60, 'B': 0.1741},
        {'A': 4.80, 'B': 0.1693},
        {'A': 5.00, 'B': 0.1648},
    },

    DragTableG5: {
        {'A': 0.00, 'B': 0.1710},
        {'A': 0.05, 'B': 0.1719},
        {'A': 0.10, 'B': 0.1727},
        {'A': 0.15, 'B': 0.1732},
        {'A': 0.20, 'B': 0.1734},
        {'A': 0.25, 'B': 0.1730},
        {'A': 0.30, 'B': 0.1718},
        {'A': 0.35, 'B': 0.1696},
        {'A': 0.40, 'B': 0.1668},
        {'A': 0.45, 'B': 0.1637},
        {'A': 0.50, 'B': 0.1603},
        {'A': 0.55, 'B': 0.1566},
        {'A': 0.60, 'B': 0.1529},
        {'A': 0.65, 'B': 0.1497},
        {'A': 0.70, 'B': 0.1473},
        {'A': 0.75, 'B': 0.1463},
        {'A': 0.80, 'B': 0.1489},
        {'A': 0.85, 'B': 0.1583},
        {'A': 0.875, 'B': 0.1672},
        {'A': 0.90, 'B': 0.1815},
        {'A': 0.925, 'B': 0.2051},
        {'A': 0.95, 'B': 0.2413},
        {'A': 0.975, 'B': 0.2884},
        {'A': 1.0, 'B': 0.3379},
        {'A': 1.025, 'B': 0.3785},
        {'A': 1.05, 'B': 0.4032},
        {'A': 1.075, 'B': 0.4147},
        {'A': 1.10, 'B': 0.4201},
        {'A': 1.15, 'B': 0.4278},
        {'A': 1.20, 'B': 0.4338},
        {'A': 1.25, 'B': 0.4373},
        {'A': 1.30, 'B': 0.4392},
        {'A': 1.35, 'B': 0.4403},
        {'A': 1.40, 'B': 0.4406},
        {'A': 1.45, 'B': 0.4401},
        {'A': 1.50, 'B': 0.4386},
        {'A': 1.55, 'B': 0.4362},
        {'A': 1.60, 'B': 0.4328},
        {'A': 1.65, 'B': 0.4286},
        {'A': 1.70, 'B': 0.4237},
        {'A': 1.75, 'B': 0.4182},
        {'A': 1.80, 'B': 0.4121},
        {'A': 1.85, 'B': 0.4057},
        {'A': 1.90, 'B': 0.3991},
        {'A': 1.95, 'B': 0.3926},
        {'A': 2.00, 'B': 0.3861},
        {'A': 2.05, 'B': 0.3800},
        {'A': 2.10, 'B': 0.3741},
        {'A': 2.15, 'B': 0.3684},
        {'A': 2.20, 'B': 0.3630},
        {'A': 2.25, 'B': 0.3578},
        {'A': 2.30, 'B': 0.3529},
        {'A': 2.35, 'B': 0.3481},
        {'A': 2.40, 'B': 0.3435},
        {'A': 2.45, 'B': 0.3391},
        {'A': 2.50, 'B': 0.3349},
        {'A': 2.60, 'B': 0.3269},
        {'A': 2.70, 'B': 0.3194},
        {'A': 2.80, 'B': 0.3125},
        {'A': 2.90, 'B': 0.3060},
        {'A': 3.00, 'B': 0.2999},
        {'A': 3.10, 'B': 0.2942},
        {'A': 3.20, 'B': 0.2889},
        {'A': 3.30, 'B': 0.2838},
        {'A': 3.40, 'B': 0.2790},
        {'A': 3.50, 'B': 0.2745},
        {'A': 3.60, 'B': 0.2703},
        {'A': 3.70, 'B': 0.2662},
        {'A': 3.80, 'B': 0.2624},
        {'A': 3.90, 'B': 0.2588},
        {'A': 4.00, 'B': 0.2553},
        {'A': 4.20, 'B': 0.2488},
        {'A': 4.40, 'B': 0.2429},
        {'A': 4.60, 'B': 0.2376},
        {'A': 4.80, 'B': 0.2326},
        {'A': 5.00, 'B': 0.2280},
    },

    DragTableG6: {
        {'A': 0.00, 'B': 0.2617},
        {'A': 0.05, 'B': 0.2553},
        {'A': 0.10, 'B': 0.2491},
        {'A': 0.15, 'B': 0.2432},
        {'A': 0.20, 'B': 0.2376},
        {'A': 0.25, 'B': 0.2324},
        {'A': 0.30, 'B': 0.2278},
        {'A': 0.35, 'B': 0.2238},
        {'A': 0.40, 'B': 0.2205},
        {'A': 0.45, 'B': 0.2177},
        {'A': 0.50, 'B': 0.2155},
        {'A': 0.55, 'B': 0.2138},
        {'A': 0.60, 'B': 0.2126},
        {'A': 0.65, 'B': 0.2121},
        {'A': 0.70, 'B': 0.2122},
        {'A': 0.75, 'B': 0.2132},
        {'A': 0.80, 'B': 0.2154},
        {'A': 0.85, 'B': 0.2194},
        {'A': 0.875, 'B': 0.2229},
        {'A': 0.90, 'B': 0.2297},
        {'A': 0.925, 'B': 0.2449},
        {'A': 0.95, 'B': 0.2732},
        {'A': 0.975, 'B': 0.3141},
        {'A': 1.0, 'B': 0.3597},
        {'A': 1.025, 'B': 0.3994},
        {'A': 1.05, 'B': 0.4261},
        {'A': 1.075, 'B': 0.4402},
        {'A': 1.10, 'B': 0.4465},
        {'A': 1.125, 'B': 0.4490},
        {'A': 1.15, 'B': 0.4497},
        {'A': 1.175, 'B': 0.4494},
        {'A': 1.20, 'B': 0.4482},
        {'A': 1.225, 'B': 0.4464},
        {'A': 1.25, 'B': 0.4441},
        {'A': 1.30, 'B': 0.4390},
        {'A': 1.35, 'B': 0.4336},
        {'A': 1.40, 'B': 0.4279},
        {'A': 1.45, 'B': 0.4221},
        {'A': 1.50, 'B': 0.4162},
        {'A': 1.55, 'B': 0.4102},
        {'A': 1.60, 'B': 0.4042},
        {'A': 1.65, 'B': 0.3981},
        {'A': 1.70, 'B': 0.3919},
        {'A': 1.75, 'B': 0.3855},
        {'A': 1.80, 'B': 0.3788},
        {'A': 1.85, 'B': 0.3721},
        {'A': 1.90, 'B': 0.3652},
        {'A': 1.95, 'B': 0.3583},
        {'A': 2.00, 'B': 0.3515},
        {'A': 2.05, 'B': 0.3447},
        {'A': 2.10, 'B': 0.3381},
        {'A': 2.15, 'B': 0.3314},
        {'A': 2.20, 'B': 0.3249},
        {'A': 2.25, 'B': 0.3185},
        {'A': 2.30, 'B': 0.3122},
        {'A': 2.35, 'B': 0.3060},
        {'A': 2.40, 'B': 0.3000},
        {'A': 2.45, 'B': 0.2941},
        {'A': 2.50, 'B': 0.2883},
        {'A': 2.60, 'B': 0.2772},
        {'A': 2.70, 'B': 0.2668},
        {'A': 2.80, 'B': 0.2574},
        {'A': 2.90, 'B': 0.2487},
        {'A': 3.00, 'B': 0.2407},
        {'A': 3.10, 'B': 0.2333},
        {'A': 3.20, 'B': 0.2265},
        {'A': 3.30, 'B': 0.2202},
        {'A': 3.40, 'B': 0.2144},
        {'A': 3.50, 'B': 0.2089},
        {'A': 3.60, 'B': 0.2039},
        {'A': 3.70, 'B': 0.1991},
        {'A': 3.80, 'B': 0.1947},
        {'A': 3.90, 'B': 0.1905},
        {'A': 4.00, 'B': 0.1866},
        {'A': 4.20, 'B': 0.1794},
        {'A': 4.40, 'B': 0.1730},
        {'A': 4.60, 'B': 0.1673},
        {'A': 4.80, 'B': 0.1621},
        {'A': 5.00, 'B': 0.1574},
    },

    DragTableG8: {
        {'A': 0.00, 'B': 0.2105},
        {'A': 0.05, 'B': 0.2105},
        {'A': 0.10, 'B': 0.2104},
        {'A': 0.15, 'B': 0.2104},
        {'A': 0.20, 'B': 0.2103},
        {'A': 0.25, 'B': 0.2103},
        {'A': 0.30, 'B': 0.2103},
        {'A': 0.35, 'B': 0.2103},
        {'A': 0.40, 'B': 0.2103},
        {'A': 0.45, 'B': 0.2102},
        {'A': 0.50, 'B': 0.2102},
        {'A': 0.55, 'B': 0.2102},
        {'A': 0.60, 'B': 0.2102},
        {'A': 0.65, 'B': 0.2102},
        {'A': 0.70, 'B': 0.2103},
        {'A': 0.75, 'B': 0.2103},
        {'A': 0.80, 'B': 0.2104},
        {'A': 0.825, 'B': 0.2104},
        {'A': 0.85, 'B': 0.2105},
        {'A': 0.875, 'B': 0.2106},
        {'A': 0.90, 'B': 0.2109},
        {'A': 0.925, 'B': 0.2183},
        {'A': 0.95, 'B': 0.2571},
        {'A': 0.975, 'B': 0.3358},
        {'A': 1.0, 'B': 0.4068},
        {'A': 1.025, 'B': 0.4378},
        {'A': 1.05, 'B': 0.4476},
        {'A': 1.075, 'B': 0.4493},
        {'A': 1.10, 'B': 0.4477},
        {'A': 1.125, 'B': 0.4450},
        {'A': 1.15, 'B': 0.4419},
        {'A': 1.20, 'B': 0.4353},
        {'A': 1.25, 'B': 0.4283},
        {'A': 1.30, 'B': 0.4208},
        {'A': 1.35, 'B': 0.4133},
        {'A': 1.40, 'B': 0.4059},
        {'A': 1.45, 'B': 0.3986},
        {'A': 1.50, 'B': 0.3915},
        {'A': 1.55, 'B': 0.3845},
        {'A': 1.60, 'B': 0.3777},
        {'A': 1.65, 'B': 0.3710},
        {'A': 1.70, 'B': 0.3645},
        {'A': 1.75, 'B': 0.3581},
        {'A': 1.80, 'B': 0.3519},
        {'A': 1.85, 'B': 0.3458},
        {'A': 1.90, 'B': 0.3400},
        {'A': 1.95, 'B': 0.3343},
        {'A': 2.00, 'B': 0.3288},
        {'A': 2.05, 'B': 0.3234},
        {'A': 2.10, 'B': 0.3182},
        {'A': 2.15, 'B': 0.3131},
        {'A': 2.20, 'B': 0.3081},
        {'A': 2.25, 'B': 0.3032},
        {'A': 2.30, 'B': 0.2983},
        {'A': 2.35, 'B': 0.2937},
        {'A': 2.40, 'B': 0.2891},
        {'A': 2.45, 'B': 0.2845},
        {'A': 2.50, 'B': 0.2802},
        {'A': 2.60, 'B': 0.2720},
        {'A': 2.70, 'B': 0.2642},
        {'A': 2.80, 'B': 0.2569},
        {'A': 2.90, 'B': 0.2499},
        {'A': 3.00, 'B': 0.2432},
        {'A': 3.10, 'B': 0.2368},
        {'A': 3.20, 'B': 0.2308},
        {'A': 3.30, 'B': 0.2251},
        {'A': 3.40, 'B': 0.2197},
        {'A': 3.50, 'B': 0.2147},
        {'A': 3.60, 'B': 0.2101},
        {'A': 3.70, 'B': 0.2058},
        {'A': 3.80, 'B': 0.2019},
        {'A': 3.90, 'B': 0.1983},
        {'A': 4.00, 'B': 0.1950},
        {'A': 4.20, 'B': 0.1890},
        {'A': 4.40, 'B': 0.1837},
        {'A': 4.60, 'B': 0.1791},
        {'A': 4.80, 'B': 0.1750},
        {'A': 5.00, 'B': 0.1713},
    },

    DragTableGI: {
        {'A': 0.00, 'B': 0.2282},
        {'A': 0.05, 'B': 0.2282},
        {'A': 0.10, 'B': 0.2282},
        {'A': 0.15, 'B': 0.2282},
        {'A': 0.20, 'B': 0.2282},
        {'A': 0.25, 'B': 0.2282},
        {'A': 0.30, 'B': 0.2282},
        {'A': 0.35, 'B': 0.2282},
        {'A': 0.40, 'B': 0.2282},
        {'A': 0.45, 'B': 0.2282},
        {'A': 0.50, 'B': 0.2282},
        {'A': 0.55, 'B': 0.2282},
        {'A': 0.60, 'B': 0.2282},
        {'A': 0.65, 'B': 0.2282},
        {'A': 0.70, 'B': 0.2282},
        {'A': 0.725, 'B': 0.2353},
        {'A': 0.75, 'B': 0.2434},
        {'A': 0.775, 'B': 0.2515},
        {'A': 0.80, 'B': 0.2596},
        {'A': 0.825, 'B': 0.2677},
        {'A': 0.85, 'B': 0.2759},
        {'A': 0.875, 'B': 0.2913},
        {'A': 0.90, 'B': 0.3170},
        {'A': 0.925, 'B': 0.3442},
        {'A': 0.95, 'B': 0.3728},
        {'A': 1.0, 'B': 0.4349},
        {'A': 1.05, 'B': 0.5034},
        {'A': 1.075, 'B': 0.5402},
        {'A': 1.10, 'B': 0.5756},
        {'A': 1.125, 'B': 0.5887},
        {'A': 1.15, 'B': 0.6018},
        {'A': 1.175, 'B': 0.6149},
        {'A': 1.20, 'B': 0.6279},
        {'A': 1.225, 'B': 0.6418},
        {'A': 1.25, 'B': 0.6423},
        {'A': 1.30, 'B': 0.6423},
        {'A': 1.35, 'B': 0.6423},
        {'A': 1.40, 'B': 0.6423},
        {'A': 1.45, 'B': 0.6423},
        {'A': 1.50, 'B': 0.6423},
        {'A': 1.55, 'B': 0.6423},
        {'A': 1.60, 'B': 0.6423},
        {'A': 1.625, 'B': 0.6407},
        {'A': 1.65, 'B': 0.6378},
        {'A': 1.70, 'B': 0.6321},
        {'A': 1.75, 'B': 0.6266},
        {'A': 1.80, 'B': 0.6213},
        {'A': 1.85, 'B': 0.6163},
        {'A': 1.90, 'B': 0.6113},
        {'A': 1.95, 'B': 0.6066},
        {'A': 2.00, 'B': 0.6020},
        {'A': 2.05, 'B': 0.5976},
        {'A': 2.10, 'B': 0.5933},
        {'A': 2.15, 'B': 0.5891},
        {'A': 2.20, 'B': 0.5850},
        {'A': 2.25, 'B': 0.5811},
        {'A': 2.30, 'B': 0.5773},
        {'A': 2.35, 'B': 0.5733},
        {'A': 2.40, 'B': 0.5679},
        {'A': 2.45, 'B': 0.5626},
        {'A': 2.50, 'B': 0.5576},
        {'A': 2.60, 'B': 0.5478},
        {'A': 2.70, 'B': 0.5386},
        {'A': 2.80, 'B': 0.5298},
        {'A': 2.90, 'B': 0.5215},
        {'A': 3.00, 'B': 0.5136},
        {'A': 3.10, 'B': 0.5061},
        {'A': 3.20, 'B': 0.4989},
        {'A': 3.30, 'B': 0.4921},
        {'A': 3.40, 'B': 0.4855},
        {'A': 3.50, 'B': 0.4792},
        {'A': 3.60, 'B': 0.4732},
        {'A': 3.70, 'B': 0.4674},
        {'A': 3.80, 'B': 0.4618},
        {'A': 3.90, 'B': 0.4564},
        {'A': 4.00, 'B': 0.4513},
        {'A': 4.20, 'B': 0.4415},
        {'A': 4.40, 'B': 0.4323},
        {'A': 4.60, 'B': 0.4238},
        {'A': 4.80, 'B': 0.4157},
        {'A': 5.00, 'B': 0.4082},
    },

    DragTableGS: {
        {'A': 0.00, 'B': 0.4662},
        {'A': 0.05, 'B': 0.4689},
        {'A': 0.10, 'B': 0.4717},
        {'A': 0.15, 'B': 0.4745},
        {'A': 0.20, 'B': 0.4772},
        {'A': 0.25, 'B': 0.4800},
        {'A': 0.30, 'B': 0.4827},
        {'A': 0.35, 'B': 0.4852},
        {'A': 0.40, 'B': 0.4882},
        {'A': 0.45, 'B': 0.4920},
        {'A': 0.50, 'B': 0.4970},
        {'A': 0.55, 'B': 0.5080},
        {'A': 0.60, 'B': 0.5260},
        {'A': 0.65, 'B': 0.5590},
        {'A': 0.70, 'B': 0.5920},
        {'A': 0.75, 'B': 0.6258},
        {'A': 0.80, 'B': 0.6610},
        {'A': 0.85, 'B': 0.6985},
        {'A': 0.90, 'B': 0.7370},
        {'A': 0.95, 'B': 0.7757},
        {'A': 1.0, 'B': 0.8140},
        {'A': 1.05, 'B': 0.8512},
        {'A': 1.10, 'B': 0.8870},
        {'A': 1.15, 'B': 0.9210},
        {'A': 1.20, 'B': 0.9510},
        {'A': 1.25, 'B': 0.9740},
        {'A': 1.30, 'B': 0.9910},
        {'A': 1.35, 'B': 0.9990},
        {'A': 1.40, 'B': 1.0030},
        {'A': 1.45, 'B': 1.0060},
        {'A': 1.50, 'B': 1.0080},
        {'A': 1.55, 'B': 1.0090},
        {'A': 1.60, 'B': 1.0090},
        {'A': 1.65, 'B': 1.0090},
        {'A': 1.70, 'B': 1.0090},
        {'A': 1.75, 'B': 1.0080},
        {'A': 1.80, 'B': 1.0070},
        {'A': 1.85, 'B': 1.0060},
        {'A': 1.90, 'B': 1.0040},
        {'A': 1.95, 'B': 1.0025},
        {'A': 2.00, 'B': 1.0010},
        {'A': 2.05, 'B': 0.9990},
        {'A': 2.10, 'B': 0.9970},
        {'A': 2.15, 'B': 0.9956},
        {'A': 2.20, 'B': 0.9940},
        {'A': 2.25, 'B': 0.9916},
        {'A': 2.30, 'B': 0.9890},
        {'A': 2.35, 'B': 0.9869},
        {'A': 2.40, 'B': 0.9850},
        {'A': 2.45, 'B': 0.9830},
        {'A': 2.50, 'B': 0.9810},
        {'A': 2.55, 'B': 0.9790},
        {'A': 2.60, 'B': 0.9770},
        {'A': 2.65, 'B': 0.9750},
        {'A': 2.70, 'B': 0.9730},
        {'A': 2.75, 'B': 0.9710},
        {'A': 2.80, 'B': 0.9690},
        {'A': 2.85, 'B': 0.9670},
        {'A': 2.90, 'B': 0.9650},
        {'A': 2.95, 'B': 0.9630},
        {'A': 3.00, 'B': 0.9610},
        {'A': 3.05, 'B': 0.9589},
        {'A': 3.10, 'B': 0.9570},
        {'A': 3.15, 'B': 0.9555},
        {'A': 3.20, 'B': 0.9540},
        {'A': 3.25, 'B': 0.9520},
        {'A': 3.30, 'B': 0.9500},
        {'A': 3.35, 'B': 0.9485},
        {'A': 3.40, 'B': 0.9470},
        {'A': 3.45, 'B': 0.9450},
        {'A': 3.50, 'B': 0.9430},
        {'A': 3.55, 'B': 0.9414},
        {'A': 3.60, 'B': 0.9400},
        {'A': 3.65, 'B': 0.9385},
        {'A': 3.70, 'B': 0.9370},
        {'A': 3.75, 'B': 0.9355},
        {'A': 3.80, 'B': 0.9340},
        {'A': 3.85, 'B': 0.9325},
        {'A': 3.90, 'B': 0.9310},
        {'A': 3.95, 'B': 0.9295},
        {'A': 4.00, 'B': 0.9280},
    },

}


class DataPoint(object):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b


class CurvePoint(object):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c


class DragFunction(float):
    pass


class BallisticCoefficient(object):
    _value: float
    _table: int
    _drag: [DragFunction, None]

    def __init__(self, value: float, drag_table: int):
        """
        creates ballistic coefficient object using the
        ballistic coefficient value and ballistic table.
        :param value: bc_value
        :param drag_table: drag_table flag
        """

        self._error = None

        if drag_table < DragTableG1 or DragTableG1 > DragTableGI:
            self._error = f"BallisticCoefficient: Unknown drag table {drag_table}"

        if value <= 0:
            self._error = 'BallisticCoefficient: Drag coefficient must be greater than zero'

        self._value = value
        self._table = drag_table
        if not self._error:
            self._drag = DragCalculate.drag_function_factory(drag_table)
        else:
            self._drag = None

    @property
    def value(self) -> float:
        """
        :return: the ballistic coefficient value
        """
        return self._value

    @property
    def table(self) -> int:
        """
        :return: the name of the ballistic table
        """
        return self._table

    def drag(self, mach) -> float:
        return self._drag(mach)


class DragCalculate(object):
    """ Class with static data calculations methods """

    @staticmethod
    def make_data_points(drag_table) -> list[DataPoint]:
        if not isinstance(drag_table, dict):
            table = dict(drag_table)
        else:
            table = drag_table

        return [DataPoint(a, b) for a, b in table.values()]

    @staticmethod
    def calculate_curve(data_points: list[DataPoint]) -> list[CurvePoint]:
        rate = (data_points[1].b - data_points[0].b) / (data_points[1].a - data_points[0].a)
        curve = [CurvePoint(0, rate, data_points[0].b - data_points[0].a * rate)]

        """ rest as 2nd degree polynomials on three adjacent points """
        for i, data_point in enumerate(data_points):
            x1 = data_points[i - 1].a
            x2 = data_point.a
            x3 = data_points[i + 1].a
            y1 = data_points[i - 1].b
            y2 = data_point.b
            y3 = data_points[i + 1].b
            a = ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1)) / (
                        (x3 * x3 - x1 * x1) * (x2 - x1) - (x2 * x2 - x1 * x1) * (x3 - x1))
            b = (y2 - y1 - a * (x2 * x2 - x1 * x1)) / (x2 - x1)
            c = y1 - (a * x1 * x1 + b * x1)
            curve.append(CurvePoint(a, b, c))

        num_points = len(data_points)
        rate = (data_points[num_points - 1].b - data_points[num_points - 2].b) / \
               (data_points[num_points - 1].a - data_points[num_points - 2].a)
        curve.append(CurvePoint(0, rate, data_points[num_points - 1].b - data_points[num_points - 2].a * rate))
        return curve

    @staticmethod
    def drag_function_factory(drag_table: int) -> [DragFunction, float]:
        try:
            table = DragCalculate.make_data_points(DRAG_TABLES[drag_table])
            curve = DragCalculate.calculate_curve(table)
            return lambda mach: DragCalculate.calculate_by_curve(table, curve, mach)
        except KeyError:
            raise ValueError("Unknown drag table type")

    @staticmethod
    def calculate_by_curve(data: list[DataPoint], curve: list[CurvePoint], mach: float) -> float:
        num_points = len(curve)
        mlo = 0
        mhi = num_points - 2

        while mhi - mlo > 1:
            mid = int(math.floor(mhi + mlo) / 2.0)
            if data[mid].a < mach:
                mlo = mid
            else:
                mhi = mid

        if data[mhi].a - mach > mach - data[mlo].a:
            m = mlo
        else:
            m = mhi
        return curve[m].c + mach * (curve[m].b + curve[m].a * mach)
