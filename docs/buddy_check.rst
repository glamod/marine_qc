.. marine QC documentation master file

Buddy check
===========

The buddy check compares an observation to its neighbours in order to determine whether it is grossly in error.
The basic method is as follows.

* Read in all the input reports.
* Grid all the accepted reports at 1x1xpentad resolution by taking means of anomalies of all the obs in each grid-box.
* Get the input standard deviation for each 1x1xpentad (call this :math:`\sigma`).
* For each grid-box calculate an acceptable range of data (call this :math:`R`) as follows:

  * Identify all other grid boxes within :math:`\pm2` pentads and  a distance equal to :math:`1^{\circ}` at the equator. The angular range is :math:`1/\cos\left(lat\right)` (Step 1)
  * Find the arithmetic mean of those grid box values (call this ), and the total number of obs in those grid boxes (call this :math:`n`).

    * If :math:`n>100:\ R=\mu\pm2.5\sigma` (for DPT :math:`4.0\sigma`)
    * If :math:`15<n<100:\ R=\mu\pm3.0\sigma` (for DPT 4.5)
    * If :math:`5<n<15:\ R=\mu\pm3.5\sigma` (for DPT 5.0)
    * If :math:`0<n<5:\ R=\mu\pm4.0\sigma` (for DPT 5.5)
    * If :math:`n = 0` Try again using grid boxes within :math:`\pm2` pentads and :math:`\pm2` degrees at the equator.
    * If :math:`n>0\ R=\mu\pm4.0\sigma` (for DPT 5:5).
    * If :math:`n` still :math:`=0`; Go back to step 1 but expand the time separation to 4 pentads.
    * If :math:`n` still :math:`=0` using :math:`\pm4` pentads and :math:`\pm2` degrees :math:`R` is infinite (I.e. all obs in this grid-box will pass buddy QC)

* Compare the anomaly for each observation to R for the 1x1xpentad grid box that contains the observation.
* If the range of R covers the observation anomaly, set the buddy check flag to 0. Otherwise, it has failed and the buddy flag is set to 1.
