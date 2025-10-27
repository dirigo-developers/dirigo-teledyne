import pyadq
from pyadq import ADQControlUnit, ADQ, ADQParameters, ADQInfoListEntry

from dirigo import units
from dirigo.hw_interfaces import digitizer  
from dirigo.sw_interfaces.acquisition import AcquisitionProduct



"""
Teledyne ADQ digitizer implementation for Dirigo.

This module provides a concrete implementation of the `digitizer.Digitizer` API
for Teledyne ADQ digitizers, using the `pyadq` library to interface with
the hardware.

Classes:
    TeledyneChannel: Configures individual input channels.
    TeledyneSampleClock: Configures the sample clock for acquisition.
    TeledyneTrigger: Manages trigger settings and operations.
    TeledyneAcquire: Handles acquisition logic and data transfer.
    TeledyneDigitizer: Combines the above components into a digitizer interface.
"""