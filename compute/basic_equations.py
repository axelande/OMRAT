from numpy import exp, log, sqrt, sin, cos, abs as np_abs, pi
from scipy import stats
from scipy.stats import norm

def get_Fcoll(na:float, pc:float) -> float:
    """Get the accident frequncy """
    return na * pc

def get_drifting_prob(Fb:float, line_length:float, ship_speed:float) -> float:
    """Calculates the drifting probability """
    hours = line_length / ship_speed
    return 1 - exp( - Fb / (24 * 365) * hours)

def get_drift_time(distance:float, drift_speed:float) -> float:
    """Estimates the drifting time """
    return distance / drift_speed

def repairtime_function(data, x) -> float:
    if data["active_window"] == 0:
        drift = stats.lognorm(data["std"], data["loc"], data["scale"])
        repaired = drift.cdf(x)
    else:
        repaired = eval(data["func"])
    return repaired

def powered_na(distance, mean_time, ship_speed):
    ai = mean_time * ship_speed
    return exp(-distance / ai)
    
def get_not_repaired(data: dict[str,str|float|bool], drift_speed:float, dist:float) -> float:
    """Get the probability that the ship isn't repaired"""
    drift_time = get_drift_time(dist, drift_speed) / 3600
    if data['use_lognormal'] == 1:
        drift = stats.lognorm(data['std'], data['loc'], data['scale'])
        prob_not_repaired = 1 - drift.cdf(drift_time)
    else:
        x = drift_time # used in the eval func
        prob_not_repaired = 1 - eval(data['func'])
    return prob_not_repaired


# =============================================================================
# Ship-Ship Collision Equations (Hansen 2008 - IWRAP Theory)
# =============================================================================

def get_head_on_collision_candidates(
    Q1: float,           # Traffic in direction 1 (ships/year)
    Q2: float,           # Traffic in direction 2 (ships/year)
    V1: float,           # Speed direction 1 (m/s)
    V2: float,           # Speed direction 2 (m/s)
    mu1: float,          # Mean lateral position dir 1 (m)
    mu2: float,          # Mean lateral position dir 2 (m)
    sigma1: float,       # Std dev lateral position dir 1 (m)
    sigma2: float,       # Std dev lateral position dir 2 (m)
    B1: float,           # Beam of vessel type 1 (m)
    B2: float,           # Beam of vessel type 2 (m)
    L_w: float           # Leg segment length (m)
) -> float:
    """
    Calculate geometric number of head-on collision candidates.

    Uses Hansen Eq. 4.2-4.4:
    N_G = Q1 × Q2 × V_ij × P_G × L_w

    Where:
    - V_ij = V1 + V2 (relative closing speed for head-on)
    - P_G = Φ((μ_ij + B_ij)/σ_ij) - Φ((μ_ij - B_ij)/σ_ij)
    - μ_ij = μ1 + μ2 (mean lateral distance between vessels)
    - σ_ij = √(σ1² + σ2²) (combined standard deviation)
    - B_ij = (B1 + B2)/2 (average vessel breadth)

    Parameters
    ----------
    Q1 : float
        Traffic in direction 1 (ships/year)
    Q2 : float
        Traffic in direction 2 (ships/year)
    V1 : float
        Speed direction 1 (m/s)
    V2 : float
        Speed direction 2 (m/s)
    mu1 : float
        Mean lateral position dir 1 (m)
    mu2 : float
        Mean lateral position dir 2 (m)
    sigma1 : float
        Std dev lateral position dir 1 (m)
    sigma2 : float
        Std dev lateral position dir 2 (m)
    B1 : float
        Beam of vessel type 1 (m)
    B2 : float
        Beam of vessel type 2 (m)
    L_w : float
        Leg segment length (m)

    Returns
    -------
    float
        Geometric number of head-on collision candidates per year
    """
    # Relative closing speed for head-on collision
    V_ij = V1 + V2

    # Mean lateral distance between vessels (head-on: opposite directions)
    mu_ij = mu1 + mu2

    # Combined standard deviation
    sigma_ij = sqrt(sigma1**2 + sigma2**2)

    # Average vessel breadth (collision width)
    B_ij = (B1 + B2) / 2

    # Geometric collision probability using cumulative normal distribution
    # P_G = Φ((μ_ij + B_ij)/σ_ij) - Φ((μ_ij - B_ij)/σ_ij)
    if sigma_ij > 0:
        P_G = norm.cdf((mu_ij + B_ij) / sigma_ij) - norm.cdf((mu_ij - B_ij) / sigma_ij)
    else:
        # If no variance, check if collision is certain (within beam)
        P_G = 1.0 if np_abs(mu_ij) <= B_ij else 0.0

    # Number of geometric collision candidates
    # Hansen Eq. 4.2-4.4:
    # N_G = (Q1/V1) × (Q2/V2) × V_ij × P_G × L_w
    # Where Q/V converts frequency (ships/year) to density (ships/meter)
    # This gives the correct dimension: collision candidates per year

    # Avoid division by zero
    if V1 <= 0 or V2 <= 0:
        return 0.0

    # Convert to ships per meter (density)
    # Q is ships/year, V is m/s, so Q/V gives ships/year / (m/s) = ships * s / (year * m)
    # We need to convert: ships/year / (m/year) to get ships/m
    # V_year = V * seconds_per_year
    seconds_per_year = 365.25 * 24 * 3600
    density1 = Q1 / (V1 * seconds_per_year)  # ships per meter
    density2 = Q2 / (V2 * seconds_per_year)  # ships per meter

    # N_G = density1 × density2 × V_ij × P_G × L_w × seconds_per_year
    # This gives collision candidates per year
    N_G = density1 * density2 * V_ij * P_G * L_w * seconds_per_year

    return N_G


def get_overtaking_collision_candidates(
    Q_fast: float,       # Traffic of faster vessels (ships/year)
    Q_slow: float,       # Traffic of slower vessels (ships/year)
    V_fast: float,       # Speed of faster vessel (m/s)
    V_slow: float,       # Speed of slower vessel (m/s)
    mu_fast: float,      # Mean lateral position faster (m)
    mu_slow: float,      # Mean lateral position slower (m)
    sigma_fast: float,   # Std dev lateral position faster (m)
    sigma_slow: float,   # Std dev lateral position slower (m)
    B_fast: float,       # Beam of faster vessel (m)
    B_slow: float,       # Beam of slower vessel (m)
    L_w: float           # Leg segment length (m)
) -> float:
    """
    Calculate geometric number of overtaking collision candidates.

    Similar to head-on but with:
    - V_ij = |V_fast - V_slow| (must be > 0)
    - μ_ij = μ_fast - μ_slow (same direction, subtract)

    Parameters
    ----------
    Q_fast : float
        Traffic of faster vessels (ships/year)
    Q_slow : float
        Traffic of slower vessels (ships/year)
    V_fast : float
        Speed of faster vessel (m/s)
    V_slow : float
        Speed of slower vessel (m/s)
    mu_fast : float
        Mean lateral position faster (m)
    mu_slow : float
        Mean lateral position slower (m)
    sigma_fast : float
        Std dev lateral position faster (m)
    sigma_slow : float
        Std dev lateral position slower (m)
    B_fast : float
        Beam of faster vessel (m)
    B_slow : float
        Beam of slower vessel (m)
    L_w : float
        Leg segment length (m)

    Returns
    -------
    float
        Geometric number of overtaking collision candidates per year.
        Returns 0 if V_fast <= V_slow.
    """
    # Check if overtaking is possible
    if V_fast <= V_slow:
        return 0.0

    # Relative speed for overtaking (difference in speeds)
    V_ij = V_fast - V_slow

    # Mean lateral distance (same direction: subtract)
    mu_ij = mu_fast - mu_slow

    # Combined standard deviation
    sigma_ij = sqrt(sigma_fast**2 + sigma_slow**2)

    # Average vessel breadth (collision width)
    B_ij = (B_fast + B_slow) / 2

    # Geometric collision probability
    if sigma_ij > 0:
        P_G = norm.cdf((mu_ij + B_ij) / sigma_ij) - norm.cdf((mu_ij - B_ij) / sigma_ij)
    else:
        P_G = 1.0 if np_abs(mu_ij) <= B_ij else 0.0

    # Number of geometric collision candidates
    # N_G = (Q_fast/V_fast) × (Q_slow/V_slow) × V_ij × P_G × L_w
    # Convert to density using the same approach as head-on

    # Avoid division by zero
    if V_fast <= 0 or V_slow <= 0:
        return 0.0

    seconds_per_year = 365.25 * 24 * 3600
    density_fast = Q_fast / (V_fast * seconds_per_year)  # ships per meter
    density_slow = Q_slow / (V_slow * seconds_per_year)  # ships per meter

    # N_G gives collision candidates per year
    N_G = density_fast * density_slow * V_ij * P_G * L_w * seconds_per_year

    return N_G


def get_crossing_collision_candidates(
    Q1: float,           # Traffic on leg 1 (ships/year)
    Q2: float,           # Traffic on leg 2 (ships/year)
    V1: float,           # Speed on leg 1 (m/s)
    V2: float,           # Speed on leg 2 (m/s)
    L1: float,           # Length ship type 1 (m)
    L2: float,           # Length ship type 2 (m)
    B1: float,           # Beam ship type 1 (m)
    B2: float,           # Beam ship type 2 (m)
    theta: float         # Crossing angle (radians)
) -> float:
    """
    Calculate geometric number of crossing collision candidates.

    Uses Hansen Eq. 4.6:
    N_G = Q1 × Q2 × D_ij / (V_ij × sin(θ))

    Where:
    - D_ij = collision diameter based on ship dimensions and crossing angle
    - V_ij = √(V1² + V2² - 2×V1×V2×cos(θ)) (relative speed)

    The collision diameter D_ij represents the effective collision zone
    based on the geometry of the vessels and the crossing angle.

    Parameters
    ----------
    Q1 : float
        Traffic on leg 1 (ships/year)
    Q2 : float
        Traffic on leg 2 (ships/year)
    V1 : float
        Speed on leg 1 (m/s)
    V2 : float
        Speed on leg 2 (m/s)
    L1 : float
        Length ship type 1 (m)
    L2 : float
        Length ship type 2 (m)
    B1 : float
        Beam ship type 1 (m)
    B2 : float
        Beam ship type 2 (m)
    theta : float
        Crossing angle (radians)

    Returns
    -------
    float
        Geometric number of crossing collision candidates per year
    """
    # Handle edge cases for crossing angle
    sin_theta = sin(theta)
    if np_abs(sin_theta) < 1e-10:
        # Parallel or anti-parallel courses - use head-on or overtaking instead
        return 0.0

    # Relative speed using law of cosines
    # V_ij = √(V1² + V2² - 2×V1×V2×cos(θ))
    V_ij = sqrt(V1**2 + V2**2 - 2 * V1 * V2 * cos(theta))

    if V_ij < 1e-10:
        return 0.0

    # Collision diameter based on ship dimensions and crossing angle
    # D_ij accounts for the projected area of both vessels
    # D_ij = (L1 + L2) × |sin(θ)| + (B1 + B2) × |cos(θ)|
    D_ij = (L1 + L2) * np_abs(sin_theta) + (B1 + B2) * np_abs(cos(theta))

    # Number of geometric collision candidates
    # Hansen Eq. 4.6: N_G = Q1 × Q2 × D_ij / (V1 × V2 × sin(θ))
    # Note: The formula divides by V1 × V2 to convert frequencies to densities
    # and includes sin(θ) to account for the crossing geometry

    # Avoid division by zero
    if V1 <= 0 or V2 <= 0:
        return 0.0

    # Convert frequency to "ships per year passing through intersection"
    # For crossing, the formula is different from head-on/overtaking
    # N_G = Q1 × Q2 × D_ij / (V1 × V2 × sin(θ)) × conversion_factor
    # The conversion factor accounts for time: ships/year squared needs to become
    # collision candidates/year

    # Using dimensional analysis:
    # Q1, Q2 in ships/year; D_ij in meters; V1, V2 in m/s
    # Q1 × Q2 × D_ij / (V1 × V2) = ships²/year² × m / (m²/s²) = ships² × s² / (year² × m)
    # We need to multiply by (1/seconds_per_year) to get ships/year
    seconds_per_year = 365.25 * 24 * 3600
    N_G = Q1 * Q2 * D_ij / (V1 * V2 * np_abs(sin_theta) * seconds_per_year)

    return N_G


def get_bend_collision_candidates(
    Q: float,            # Traffic volume (ships/year)
    P_no_turn: float,    # Probability of not changing course at bend (≈0.01)
    L: float,            # Ship length (m)
    B: float,            # Ship beam (m)
    theta: float         # Bend angle (radians)
) -> float:
    """
    Calculate bend collision candidates at waypoints.

    At bends/waypoints, there is a probability that a vessel fails to
    turn and continues on the original course. This can lead to collision
    with vessels that do make the turn.

    N_bend = P_0 × Q × N_G^crossing(self-interaction)

    Where P_0 is the probability of failing to turn, and the crossing
    collision is calculated for the vessel interacting with itself
    (same ship type on both legs).

    Parameters
    ----------
    Q : float
        Traffic volume (ships/year)
    P_no_turn : float
        Probability of not changing course at bend (typically ≈0.01)
    L : float
        Ship length (m)
    B : float
        Ship beam (m)
    theta : float
        Bend angle (radians) - the angle change at the waypoint

    Returns
    -------
    float
        Number of bend collision candidates per year
    """
    # Handle edge case of no bend
    if np_abs(theta) < 1e-10:
        return 0.0

    # Calculate crossing collision for self-interaction
    # Ships that fail to turn interact with ships that do turn
    # Use same ship type for both (self-interaction)
    # The crossing angle is the bend angle theta

    # Traffic that fails to turn
    Q_no_turn = Q * P_no_turn

    # Traffic that turns normally
    Q_turn = Q * (1 - P_no_turn)

    # Calculate crossing collision between non-turning and turning traffic
    # Assuming same speed for simplicity (use average speed V)
    # For self-interaction, V1 = V2 = V (speeds cancel partially in the formula)

    # Using crossing collision formula for self-interaction
    N_G_crossing = get_crossing_collision_candidates(
        Q1=Q_no_turn,
        Q2=Q_turn,
        V1=1.0,  # Normalized speed (actual speed cancels in ratio)
        V2=1.0,
        L1=L,
        L2=L,
        B1=B,
        B2=B,
        theta=theta
    )

    return N_G_crossing


# =============================================================================
# Powered Grounding Equations (Hansen 2008, Eq. 4.15-4.17)
# =============================================================================

def get_recovery_distance(position_check_minutes: float, ship_speed: float) -> float:
    """
    Calculate the characteristic recovery distance a = λ × V.

    This is the average distance a ship travels between position checks.
    Used in powered grounding Category II calculations.

    Parameters
    ----------
    position_check_minutes : float
        Position check interval λ (minutes). Typical value is ~3 minutes.
    ship_speed : float
        Ship speed V (m/s)

    Returns
    -------
    float
        Recovery distance a (meters)
    """
    # Convert minutes to seconds for consistency with m/s speed
    position_check_seconds = position_check_minutes * 60.0
    return position_check_seconds * ship_speed


def get_powered_grounding_cat1(
    Q: float,
    Pc: float,
    prob_in_obstacle: float
) -> float:
    """
    Calculate powered grounding Category I - ships sailing directly into obstacle.

    Category I covers ships that are in the traffic lane that overlaps the obstacle.
    These ships are sailing on a course that would naturally take them into the
    obstacle zone without any course deviation.

    Based on Hansen Eq. 4.15:
    N_I = P_c × Q × prob_in_obstacle

    Where prob_in_obstacle = ∫(z_min to z_max) f_i(z) dz
    (the integral of the traffic distribution over the obstacle extent)

    Parameters
    ----------
    Q : float
        Traffic volume (ships/year)
    Pc : float
        Causation factor - probability that a ship fails to take evasive action
    prob_in_obstacle : float
        Probability of being in obstacle zone. This is the integral of the
        lateral traffic distribution f(z) over the obstacle extent from z_min
        to z_max. Typically computed as Φ((z_max - μ)/σ) - Φ((z_min - μ)/σ)
        for a normal distribution.

    Returns
    -------
    float
        Expected number of Category I powered groundings per year
    """
    return Pc * Q * prob_in_obstacle


def get_powered_grounding_cat2(
    Q: float,
    Pc: float,
    prob_at_position: float,
    distance_to_obstacle: float,
    position_check_interval: float,
    ship_speed: float
) -> float:
    """
    Calculate powered grounding Category II - ships failing to turn at bend.

    Category II covers ships that fail to change course at a bend/waypoint
    and continue on their original heading towards an obstacle. The probability
    of grounding decreases exponentially with distance from the bend as ships
    have more opportunities to detect and correct their course error.

    Based on Hansen Eq. 4.16-4.17:
    N_II = P_c × Q × f(z) × exp(-d / a)

    Where:
    - d = distance from bend to obstacle (m)
    - λ = position check interval (minutes, converted to seconds)
    - a = λ × V = distance between position checks (recovery distance)
    - f(z) = probability density at the bend position

    The exponential term exp(-d/a) represents the probability that the ship
    has NOT detected its course error before reaching the obstacle.

    Parameters
    ----------
    Q : float
        Traffic volume (ships/year)
    Pc : float
        Causation factor - probability that a ship fails to take evasive action
    prob_at_position : float
        Probability density f(z) at the bend position. This represents the
        likelihood of a ship being at the lateral position where it would
        head towards the obstacle after missing the turn.
    distance_to_obstacle : float
        Distance d from the bend to the obstacle (meters)
    position_check_interval : float
        λ - Position check interval (minutes). Typical value is ~3 minutes.
        This represents how frequently the navigator checks the ship's position.
    ship_speed : float
        Ship speed V (m/s)

    Returns
    -------
    float
        Expected number of Category II powered groundings per year
    """
    # Calculate recovery distance a = λ × V
    recovery_dist = get_recovery_distance(position_check_interval, ship_speed)

    # Avoid division by zero
    if recovery_dist <= 0:
        return 0.0

    # Exponential decay: probability of not detecting course error
    # before reaching obstacle at distance d
    prob_not_recovered = exp(-distance_to_obstacle / recovery_dist)

    return Pc * Q * prob_at_position * prob_not_recovered

