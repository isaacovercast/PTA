=================
API Documentation
=================

This is the API documentation for ``MESS``, and provides detailed information
on the Python programming interface. See the :ref:`sec_api_tutorial` for an
introduction to using this API to run simulations.

Simulation model
****************

Region
++++++


Metacommunity
+++++++++++++


Local Community
+++++++++++++++


Inference Procedure
*******************

.. autoclass:: PTA.inference.Ensemble
    :members:

Model Selection (Classification)
++++++++++++++++++++++++++++++++

.. autoclass:: PTA.inference.Classifier
    :members:

Parameter Estimation (Regression)
+++++++++++++++++++++++++++++++++

.. autoclass:: PTA.inference.Regressor
    :members:

Classification Cross-Validation
+++++++++++++++++++++++++++++++

.. autofunction:: PTA.inference.classification_cv

Parameter Estimation Cross-Validation
+++++++++++++++++++++++++++++++++++++

.. autofunction:: PTA.inference.parameter_estimation_cv

Posterior Predictive Checks
+++++++++++++++++++++++++++

.. autofunction:: PTA.inference.posterior_predictive_check

Stats
*****

Plotting
++++++++

.. autofunction:: PTA.plotting.plot_simulations_hist

.. autofunction:: PTA.plotting.plot_simulations_pca

.. autofunction:: PTA.plotting.plots_through_time

