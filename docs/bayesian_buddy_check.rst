.. marine QC documentation master file

.. _bayesian_buddy_check:

Bayesian buddy check
====================

Theoretical basis
+++++++++++++++++

The probability that an observation is grossly in error :math:`P(E)`, given a particular value :math:`O` is

:math:`P(E|O)=\frac{P(O|E)P(E)}{P(O|E)P(E)+P(O|N)(1-P(E))}`

where :math:`P(O|N)` is the probability of getting the observed value, :math:`O`, given that the error in the value is normal,
i.e. roughly gaussian with a reasonable mean and standard deviation. This is similar to the approach taken by the
IQUAM background check. However, we will deviate from their method in order to derive a method that is
independent of the satellite reference field that they used and which can therefore be applied at all times. In
this case we will also deal with the quantization of the data (which are typically integer multiples
of :math:`0.1^{\circ}\mathrm{C}` or :math:`1^{\circ}\mathrm{C})` and the fact that a range limit has already been applied
to the data.

It is assumed that the data have already been passed through a basic climatology check which has rejected values
of :math:`O` outside the range :math:`R_{low}` to :math:`R_{high}` so that all values fall within an interval of
:math:`R=R_{high}-R_{low}`.

Furthermore, we assume that the data are quantized i.e, they only take values that are integer multiples of some
particular number :math:`Q` which will usually be :math:`0.1^{\circ}\mathrm{C}`.

Gross errors are assumed to fall uniformly within the allowed range such that the probability of getting a
particular observation given a particular gross error is

:math:`P(O|E)=\frac{1}{1+\frac{R}{Q}}`

i.e. the reciprocal of the number of possible quantized values in the interval :math:`R`.

The probability :math:`P(O|N)` will be a normal distribution with a specified mean, :math:`\mu` , and
standard deviation, :math:`\sigma`.
The mean might be zero if this is a climatology check (in some regions we might be able to do considerably better
than the standard :math:`\pm8^{\circ}\mathrm{C})` with the standard deviation corresponding to the climatological standard
deviation. For a background check, the mean would be the background estimate and the standard deviation its
uncertainty, augmented by the expected measurement uncertainty. For a buddy check, the mean would be the buddy
average and the standard deviation would be the uncertainty in that average combined (again) with the expected
measurement uncertainty of the observation to be tested.

Regardless, the probability will be

:math:`P(O|N)=\frac{\int_{L_{low}}^{L_{high}}\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{\left(x-\mu\right)^{2}}{2\sigma^{2}}\right)dx}{\int_{R_{low}-\frac{Q}{2}}^{R_{high}+\frac{Q}{2}}\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{\left(x-\mu\right)^{2}}{2\sigma^{2}}\right)dx}`

where

:math:`L_{high}=\min\left[o+\frac{Q}{2},\ R_{high}+\frac{Q}{2}\right]`

and

:math:`L_{low}=\max\left[o-\frac{Q}{2},\ R_{low}-\frac{Q}{2}\right]`

The integral in the top half of the fraction is the integral of the pdf between the observed value
:math:`\pm\frac{Q}{2}` which is to say over the range of actual temperatures that would round to the observed
value (unless we are close to the range limits) and the integral in the bottom half is the integral of the pdf
across the entire allowed range to normalise the probability particularly for those cases where the observation
is within a few :math:`\sigma` of the limit.

This can be rewritten in terms of error functions as

:math:`P(O|N)=\frac{erf\left(\frac{L_{high}-\mu}{\sigma\sqrt{2}}\right)-erf\left(\frac{L_{low}-\mu}{\sigma\sqrt{2}}\right)}{erf\left(\frac{R_{high}+\frac{Q}{2}-\mu}{\sigma\sqrt{2}}\right)-erf\left(\frac{R_{low}-\frac{Q}{2}-\mu}{\sigma\sqrt{2}}\right)}`

The remaining term is :math:`P(E)`, the prior probability of gross error which ought to be whatever you think the
prevailing probability of gross error is or, if you have more specific information, like you know the ship is
unreliable, then use that instead.

Combining all these terms gives a monstrously ugly formula which isn't worth writing out here.
