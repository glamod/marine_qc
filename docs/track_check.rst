.. marine QC documentation master file

do_track_check
==============

The track check uses the location and datetime information from the reports as well as the ship speed and direction
information, if available, to determine if any of the reported locations and times are likely to be erroneous.

detailed description
++++++++++++++++++++

The aim of the track check algorithm is to identify from a set of observations sharing a common, non-generic
ID, which of those observations have misreported locations. It does this by comparing consecutive observations
with each other and comparing reported speeds and directions with actual speeds and directions.

Variables used in track check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variables are used in the track check

Difference from estimated position going forwards
"""""""""""""""""""""""""""""""""""""""""""""""""

:math:`D_{forward}` The expected position of the observation at t+1 is calculated by taking the reported speed (VS in ICOADS)
and reported direction (DS in ICOADS) at time t and calculating an increment of latitude and longitude
consistent with half the time difference between t and t+1. A second increment is calculated using the
reported speed and reported direction at time t+1. Again, the increment is that which would be expected in
half the time between the two observations. The two increments are then combined to get an estimate of where
the observation at time t+1 is expected to be.

The distance between the estimated and expected position is calculated and assigned to time t+1.

Difference from estimated position going backwards
""""""""""""""""""""""""""""""""""""""""""""""""""

:math:`D_{back}` The same test as was just described was performed on the reversed list of observations. i.e. running
backwards in time.

Difference from estimated position based on interpolating between alternate positions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

:math:`D_{mid}` The position at time t is estimated based on the latitudes and longitude reported at time t-1 and time t+1.
The estimated position is assumed to be a fraction f along this line where

:math:`f=\frac{T(t)-T(t-1)}{T(t+1)-T(t-1)}`

Again, the distance between the estimated and the reported positions is calculated and assigned to time t.

Calculating speed limits
""""""""""""""""""""""""

:math:`V_{max}` The modal speed, :math:`V_{mode}`, estimated from all consecutive pairs of observations is calculated. The
speeds
are binned into 3 knot bins and the bin with the most observations in is the modal speed. The speed
limit :math:`V_{max}` is the higher of :math:`8.50\,\mathrm{\mathrm{knots}}` or :math:`1.25*V_{mode}`.

Estimate the course of the ship
"""""""""""""""""""""""""""""""

:math:`C_{est}` The course of the ship at t is estimated based on the position at t-1 and t assuming a great-circle course
between the two points.

The individual checks
^^^^^^^^^^^^^^^^^^^^^

speed check
"""""""""""

Set the speed check total to zero and the speed check flag to pass.

If the speed estimated from the positions at t-1 and t exceeds :math:`V_{max}` and the speed estimated from the positions at time t-2 and t exceeds :math:`V_{max}`. Then add one to the speed check total.

If the speed estimated from the positions at t and t+1 exceeds :math:`V_{max}` and the speed estimated from the positions at t and t+2 exceeds :math:`V_{max}`, add two to the speed check total.

If the speed estimated from the positions at t-1 and t exceeds :math:`V_{max}` and the speed estimated from the positions at t and t+1 exceeds :math:`V_{max}`, add three to the speed check total.

If the speed check total is greater than zero, set speed check flag to fail.

Distance from estimated location check
""""""""""""""""""""""""""""""""""""""

Set the distance check flag to pass

Calculate the distance

:math:`D_{max}=\left(T(t)-T(t-1)\right)\left(V(t)+V(t-1)\right)/2`

If both :math:`D_{forward}` and :math:`D_{back}` exceed :math:`D_{max}` then set the distance check flag to fail.

Direction consistency check
"""""""""""""""""""""""""""

Set the direction consistency flag to pass

if the :math:`C_{est}` differs from the reported bearing at t-1 or t by more than :math:`60^{\circ}`, set the
direction consistency flag to fail.

Speed consistency check
"""""""""""""""""""""""

Set the speed continuity flag to pass

If the speed estimated based on the locations at t-1 and t differs from the reported speed at t by more than
10 knots and differs from the reported speed at t-1 by more than 10 knots, set the speed consistency flag to
fail.

Extreme speed check
"""""""""""""""""""

Set the extreme speed check flag to pass.

If the speed estimated from the locations at t-1 and t exceeds 40 knots, set the extreme speed
check flag to fail.

Mid point check
"""""""""""""""

Set the midpoint check flag to pass.

If :math:`D_{mid}` is greater than 150 nm set the midpoint check flag to fail.

The combined track check
^^^^^^^^^^^^^^^^^^^^^^^^

An observation fails track check if an observation fails the midpoint check, the speed check and at least
one of the other checks (extreme speed, direction consistency, speed consistency, distance from estimated
location).
