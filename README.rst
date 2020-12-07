UX-SAD: AI Models
=================

Developing
----------

For developing purposes, the tool checks for the existence of the environment
variable ``UXSAD_ENV``. If ``UXSAD_ENV`` is equal to "``development``" then the
tool is executed on a heavily reduced dataset. Otherwise, if ``UXSAD_ENV`` is
equal to "``production``" (the default) then the tool is executed on the full
dataset.
