from functools import cached_property
from typing import cast
import ctypes as ct
import time

import numpy as np
import pyadq
from pyadq import ADQControlUnit, ADQ
from pyadq.structs import (
    _ADQGen4Record, _ADQGen4RecordArray, _ADQGen4RecordHeader, 
    _ADQDataReadoutStatus
)

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


class _TeledyneParameterMixin:
    """Provides helper methods to get ADQ parameter structures"""
    _dev: "ADQ"
    
    def _get_const_params(self) -> pyadq.ADQConstantParameters:
        return cast(
            pyadq.ADQConstantParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_CONSTANT)
        )
    
    def _get_acq_params(self) -> pyadq.ADQDataAcquisitionParameters:
        return cast(
            pyadq.ADQDataAcquisitionParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_ACQUISITION)
        )
    
    def _get_transf_params(self) -> pyadq.ADQDataTransferParameters:
        return cast(
            pyadq.ADQDataTransferParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_TRANSFER)
        )
    
    def _get_afe_params(self):
        return cast(
            pyadq.ADQAnalogFrontendParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_ANALOG_FRONTEND)
        )
    
    def _get_clk_params(self):
        return cast(
            pyadq.ADQClockSystemParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_CLOCK_SYSTEM)
        )
    
    def _get_event_source_trig_params(self) -> pyadq.ADQEventSourcePortParameters:
        return cast(
            pyadq.ADQEventSourcePortParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_EVENT_SOURCE_TRIG)
        )
    
    def _get_readout_params(self) -> pyadq.ADQDataReadoutParameters:
        return cast(
            pyadq.ADQDataReadoutParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_READOUT)
        )
    
    def _get_skip_params(self) -> pyadq.ADQSampleSkipParameters:
        return cast(
            pyadq.ADQSampleSkipParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_SAMPLE_SKIP)
        )


class TeledyneChannel(digitizer.Channel, _TeledyneParameterMixin):
    """
    Configures the parameters for individual input channels on a Teledyne ADQ 
    digitizer.

    Properties:
        index (int): The index of the channel (0-based).
        coupling (...): Signal coupling mode (e.g., "AC", "DC").
        impedance (...): Input impedance setting (e.g., 50 Ohm, 1 MOhm).
        range (...): Voltage range for the channel.
        enabled (bool): Indicates whether the channel is active for acquisition.
        offset (Voltage): Analog front end DC offset voltage.
    """
    
    def __init__(self, device: ADQ, channel_index: int):
        self._dev = device
        self._index = channel_index
        super().__init__(enabled=False, inverted=False)  # Initialize parent class

        # Set fixed parameters
        self._coupling: digitizer.ChannelCoupling = digitizer.ChannelCoupling.DC
        self._impedance: units.Resistance = units.Resistance("50 ohm")
        self._range: units.VoltageRange = units.VoltageRange("-250 mV", "250 mV")
        
        # We may want to set nof_records to nonzero here to avoid parameter updates being ignored
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def enabled(self) -> bool:
        """Indicates whether the channel is enabled for acquisition."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, enable: bool):
        """Enable or disable the channel."""
        if not isinstance(enable, bool):
            raise ValueError("`enabled` must be set with a boolean")
        acq_params = self._get_acq_params()
        acq_params.channel[self.index].nof_records = 1 if enable else 0
        acq_params.channel[self.index].record_length = 2 if enable else 0
        self._dev.SetParameters(acq_params)
        self._enabled = enable

    @property
    def coupling(self) -> digitizer.ChannelCoupling:
        if self._coupling is None:
            raise RuntimeError("Coupling not initialized")
        return self._coupling
    
    @coupling.setter
    def coupling(self, coupling: digitizer.ChannelCoupling):
        if coupling not in self.coupling_options:
            raise ValueError(f"Invalid input coupling {coupling}. "
                             f"Valid options are: {self.coupling_options}")
        self._coupling = coupling

    @property
    def coupling_options(self) -> set[digitizer.ChannelCoupling]:
        # Teledyne ADQ typically supports DC coupling
        return {digitizer.ChannelCoupling.DC}
    
    @property
    def impedance(self) -> units.Resistance:
        if self._impedance is None:
            raise RuntimeError("Impedance not initialized")
        return self._impedance
    
    @impedance.setter
    def impedance(self, impedance: units.Resistance):
        if impedance not in self.impedance_options:
            raise ValueError(f"Invalid input impedance {impedance}. "
                             f"Valid options are: {self.impedance_options}")
        self._impedance = impedance

    @property
    def impedance_options(self) -> set[units.Resistance]:
        # Teledyne ADQ typically supports 50 Ohm impedance
        return {units.Resistance("50 ohm")}
    
    @property
    def range(self) -> units.VoltageRange:
        if self._range is None:
            raise RuntimeError("Range is not initialized")
        return self._range
    
    @range.setter
    def range(self, rng: units.VoltageRange):
        if rng not in self.range_options:
            raise ValueError(f"Invalid input range {rng}. "
                             f"Valid options are: {self.range_options}")
        self._range = rng
    
    @property
    def range_options(self) -> set[units.VoltageRange]:
        # ADQ32 hardware has a fixed 500 mV peak-to-peak input range
        return {
            units.VoltageRange("-250 mV", "250 mV"),  # 500 mV peak-to-peak
        }
    
    @property
    def offset(self) -> units.Voltage:
        afe_params = self._get_afe_params()
        return units.Voltage(afe_params.channel[self._index].dc_offset / 1000) # mV->V

    @offset.setter
    def offset(self, offset: units.Voltage): # returns in <4 ms
        if not self.offset_range.within_range(offset):
            raise ValueError(f"Invalid offset {offset}. "
                             f"Valid range: {self.offset_range}")
        
        # Get and set analog front end parameter struct
        afe_params = self._get_afe_params()
        afe_params.channel[self._index].dc_offset = float(offset) * 1000 # V->mV
        self._dev.SetParameters(afe_params)

    @property
    def offset_range(self) -> units.VoltageRange:
        return units.VoltageRange("-250 mV", "250 mV")


class TeledyneSampleClock(digitizer.SampleClock, _TeledyneParameterMixin):
    """
    Configures the sample clock for a Teledyne SP Devices digitizer.

    This class handles the configuration of the digitizer's sample clock, 
    including the source, rate, and edge settings.

    Properties:
        source (SampleClockSource): The source of the sample clock (e.g., Internal or External).
        rate (SampleRate): The sample clock rate.
        edge (SampleClockEdge): The clock edge to use for sampling (e.g., Rising or Falling).

    Note:
        The clock source determines which rates and ranges are valid. Internal 
        clocks use predefined rates, while external clocks can accept user-defined
        frequencies within specific limits.
    """

    def __init__(self, device: ADQ):
        self._dev = device

        # Set parameters to None to signify that they have not been initialized
        self._source: digitizer.SampleClockSource | None = None
        
        # Default clock edge, set to rising
        self._edge: digitizer.SampleClockEdge = digitizer.SampleClockEdge.RISING

        # sampling rate depends on number of channels enabled at (?) firmware level
        clk_params = self._get_clk_params()
        self._base_sampling_rate = units.SampleRate(clk_params.sampling_frequency)
    
    @property
    def source(self) -> digitizer.SampleClockSource:
        if self._source is None:
            raise RuntimeError("Source not initialized")
        return self._source
    
    @source.setter
    def source(self, source: digitizer.SampleClockSource):
        if not isinstance(source, digitizer.SampleClockSource):
            raise ValueError("Sample clock source must be set with a SampleClockSource enumeration.")
        if source not in self.source_options:
            raise ValueError(f"{source} (sample clock source) is not available")
        
        clk_params = self._get_clk_params()
        if source == digitizer.SampleClockSource.INTERNAL:
            clk_params.clock_generator = pyadq.ADQ_CLOCK_GENERATOR_INTERNAL_PLL
            clk_params.reference_source = pyadq.ADQ_REFERENCE_CLOCK_SOURCE_INTERNAL
        else:
            raise NotImplementedError("Have not finished external sample clock")
        
        self._dev.SetParameters(clk_params)

        self._source = source
    
    @property
    def source_options(self) -> set[digitizer.SampleClockSource]:
        # Teledyne ADQ supports internal and external clocks
        return {digitizer.SampleClockSource.INTERNAL, 
                digitizer.SampleClockSource.EXTERNAL}
    
    @property
    def rate(self) -> units.SampleRate:
        """
        Returns the effective sampling rate: the base rate divided by the skip 
        factor.
        """
        skip_params = self._get_skip_params()
        skip = skip_params.channel[0].skip_factor
        return units.SampleRate(self._base_sampling_rate / skip)
    
    @rate.setter
    def rate(self, rate: units.SampleRate): # note, this can take a long time to return
        if self._source is None:
            raise ValueError("`source` must be set before attempting to set `rate`")

        if self._source == digitizer.SampleClockSource.INTERNAL:
            # Check if proposed rate matches a valid rate
            valid_rates = cast(set[units.SampleRate], self.rate_options) # cast b/c we know only discrete rates are possible
            if rate not in valid_rates:
                # TODO, do we want to support rounding to nearest rate, say when the proposed rate is within 5%?
                raise ValueError(f"Invalid sample clock rate: {rate}. "
                                 f"Valid options: {valid_rates}")
            
            skip_params = self._get_skip_params()
            for i in range(2): # don't have reference to channels yet, so just set them all
                skip_params.channel[i].skip_factor = round(self._base_sampling_rate / rate)
            self._dev.SetParameters(skip_params)

        elif self._source == digitizer.SampleClockSource.EXTERNAL:
            raise NotImplementedError("External clocks not yet implemented")
            
        else:
            raise RuntimeError(f"Invalid sample clock source: {self._source}")
    
    def _valid_skip_factors(self, channels_enabled: int = 2) -> list[int]:
        # Note that this is a subset of the actual supported skip factors
        # providing a list of all the supported values (4 million) would take too long
        high_range = [20, 25, 50, 100, 125, 250, 500, 1250, 2500]
        if channels_enabled == 2:
            return [1, 2, 4, 5, 8, 9, 10] + high_range
        elif channels_enabled == 1:
            return [1, 2, 4, 5, 8, 16, 17, 18] + high_range
        else:
            raise ValueError("Unsupported channel configuration for ADQ32/33")
    
    @property
    def rate_options(self) -> set[units.SampleRate] | units.SampleRateRange:
        if self._source is None:
            raise RuntimeError("`source` must be set before attempting to access rate options")
        
        if self._source == digitizer.SampleClockSource.INTERNAL:
            return {self._base_sampling_rate / s for s in self._valid_skip_factors()}
        
        if self._source == digitizer.SampleClockSource.EXTERNAL:
            raise NotImplementedError("Haven't completed external clocking")
        
        else:
            raise RuntimeError(f"Unsupported source: {self._source}")
    
    @property
    def edge(self) -> digitizer.SampleClockEdge:
        return digitizer.SampleClockEdge.RISING
    
    @edge.setter
    def edge(self, edge: digitizer.SampleClockEdge):
        # Teledyne does not allow setting this explicitly, assume using up edge for 2-channel
        if edge not in self.edge_options:
            raise ValueError(f"Proposed clock edge: {edge} not an available "
                             f"option {self.edge_options}")
        self._edge = edge

    @property
    def edge_options(self) -> set[digitizer.SampleClockEdge]:
        # Teledyne does not allow setting this explicitly
        return {digitizer.SampleClockEdge.RISING}


class TeledyneTrigger(digitizer.Trigger, _TeledyneParameterMixin):
    """
    Configures triggering behavior for a Teledyne ADQ digitizer.

    This class manages trigger settings, including source, slope, level, and 
    external coupling. It supports both internal and external trigger sources.

    Properties:
        source (TriggerSource): The trigger source (e.g., "Channel A", "External").
        slope (TriggerSlope): The trigger slope (e.g., "Positive", "Negative").
        level (dirigo.Voltage): The trigger level in volts.
        external_coupling (ExternalTriggerCoupling): Coupling mode for the external trigger source (e.g., "DC").
        external_range (...): Voltage range for the external trigger source.
    
    Notes: With ADQ cards, it is possible to configure separate event triggers 
    for channels A & B, but this is not supported in Dirigo.
    """
    _trigger_source_mapping = {
        digitizer.TriggerSource.INTERNAL:   pyadq.ADQ_EVENT_SOURCE_SOFTWARE,
        digitizer.TriggerSource.EXTERNAL:   pyadq.ADQ_EVENT_SOURCE_TRIG,
        digitizer.TriggerSource.CHANNEL_A:  pyadq.ADQ_EVENT_SOURCE_LEVEL_CHANNEL0,
        digitizer.TriggerSource.CHANNEL_B:  pyadq.ADQ_EVENT_SOURCE_LEVEL_CHANNEL1,
        digitizer.TriggerSource.CHANNEL_C:  pyadq.ADQ_EVENT_SOURCE_LEVEL_CHANNEL2,
        digitizer.TriggerSource.CHANNEL_D:  pyadq.ADQ_EVENT_SOURCE_LEVEL_CHANNEL3,
    }
    _trigger_slope_mapping = {
        digitizer.TriggerSlope.RISING:      pyadq.ADQ_EDGE_RISING,
        digitizer.TriggerSlope.FALLING:     pyadq.ADQ_EDGE_FALLING,
    }

    def __init__(self, device: ADQ, channels: tuple[TeledyneChannel, ...]):
        self._dev = device
        self._chans = channels

        # Set parameters to None to signify that they have not been initialized
        self._slope: digitizer.TriggerSlope | None = None
        self._external_coupling: digitizer.ExternalTriggerCoupling | None = None
        self._external_range: units.VoltageRange | digitizer.ExternalTriggerRange | None = None    

    @property
    def source(self) -> digitizer.TriggerSource:
        acq_params = self._get_acq_params()
        rvs_table = {v: k for k, v in self._trigger_source_mapping.items()}
        return rvs_table[acq_params.channel[0].trigger_source]
    
    @source.setter
    def source(self, source: digitizer.TriggerSource):
        if source not in self.source_options:
            raise ValueError(f"Invalid trigger source: {source}. "
                             f"Valid options are: {self.source_options}")
        acq_params = self._get_acq_params()
        for i in range(len(self._chans)):
            acq_params.channel[i].trigger_source = self._trigger_source_mapping[source]
        self._dev.SetParameters(acq_params)
    
    @property
    def source_options(self) -> set[digitizer.TriggerSource]:
        options = {digitizer.TriggerSource.EXTERNAL}
        
        # Add channel triggers for enabled channels
        for channel in self._chans:
            if channel.enabled:
                if channel.index == 0:
                    options.add(digitizer.TriggerSource.CHANNEL_A)
                elif channel.index == 1:
                    options.add(digitizer.TriggerSource.CHANNEL_B)
                elif channel.index == 2:
                    options.add(digitizer.TriggerSource.CHANNEL_C)
                elif channel.index == 3:
                    options.add(digitizer.TriggerSource.CHANNEL_D)
        # Additional modes not (yet) supported: periodic, sync, PXIe
        return options
    
    @property
    def slope(self) -> digitizer.TriggerSlope:
        acq_params = self._get_acq_params()
        rvs_table = {v: k for k, v in self._trigger_slope_mapping.items()}
        return rvs_table[acq_params.channel[0].trigger_edge]
    
    @slope.setter
    def slope(self, slope: digitizer.TriggerSlope):
        if slope not in self.slope_options:
            raise ValueError(f"Invalid trigger slope: {slope}. "
                             f"Valid options are: {self.slope_options}")
        acq_params = self._get_acq_params()
        for i in range(len(self._chans)):
            acq_params.channel[i].trigger_edge = self._trigger_slope_mapping[slope]
        self._dev.SetParameters(acq_params)

    @property
    def slope_options(self) -> set[digitizer.TriggerSlope]:
        return {digitizer.TriggerSlope.RISING, digitizer.TriggerSlope.FALLING}

    @property
    def level(self) -> units.Voltage:
        trig_params = self._get_event_source_trig_params()
        return units.Voltage(trig_params.pin[0].threshold)
    
    @level.setter
    def level(self, level: units.Voltage):
        if not self.level_limits.within_range(level):
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        trig_params = self._get_event_source_trig_params()
        trig_params.pin[0].threshold = float(level)
        self._dev.SetParameters(trig_params)
    
    @property
    def level_limits(self) -> units.VoltageRange:
        if self.source == digitizer.TriggerSource.EXTERNAL:
            # will need to switch on TRIG port impedance
            return units.VoltageRange("0 V", "2.8 V")
        elif self.source in [digitizer.TriggerSource.CHANNEL_A, digitizer.TriggerSource.CHANNEL_B, digitizer.TriggerSource.CHANNEL_C, digitizer.TriggerSource.CHANNEL_D]:
            raise NotImplementedError("Triggering on Channels not implemented yet")
        else:
            raise RuntimeError("Invalid trigger source")
    
    # ADQ32 supports 
    @property
    def external_coupling(self) -> digitizer.ExternalTriggerCoupling:
        # Teledyne ADQ32 only supports DC coupling for external triggers
        return digitizer.ExternalTriggerCoupling.DC 
    
    @external_coupling.setter
    def external_coupling(self, external_coupling: digitizer.ExternalTriggerCoupling):
        if external_coupling not in self.external_coupling_options:
            raise ValueError(f"Unsupported external trigger coupling mode: {external_coupling}. "
                             f"Supported: {self.external_coupling_options}")
        self._external_coupling = external_coupling

    @property
    def external_coupling_options(self) -> set[digitizer.ExternalTriggerCoupling]:
        # Teledyne ADQ32 only supports DC coupling for external triggers
        return {digitizer.ExternalTriggerCoupling.DC}

    @property
    def external_range(self): # this doesn't quite make sense here
        return units.VoltageRange("0 V", "2.8 V")

    @property
    def external_range_options(self):
        return {units.VoltageRange("0 V", "2.8 V")}
        

class TeledyneAcquire(digitizer.Acquire, _TeledyneParameterMixin):
    def __init__(self, device: ADQ, channels: tuple[TeledyneChannel, ...]):
        self._dev = device
        self._channels = channels

        self._trigger_offset: int = 0
        self._record_length: int = 0

        self._records_per_buffer: int = 1 # default, but in practice usually use >>1
        self._buffers_per_acquisition: int = 1
        self._buffers_allocated: int = 1
        self._timestamps_enabled: bool = True
        self._t0: int | None = None

        self._buffers_acquired: int = 0 # the start sequence should always reset this to 0

    @property
    def trigger_offset(self) -> int:
        return self._trigger_offset

    @trigger_offset.setter
    def trigger_offset(self, offset: int):
        offset = int(offset)
        if not self.trigger_offset_range.within_range(offset):
            raise ValueError(f"Invalid trigger offset {offset}. "
                             f"Valid range: {self.trigger_offset_range}")
        if offset > 0:      # delay
            if (offset % self.trigger_delay_resolution) != 0:
                raise ValueError(f"Invalid trigger offset {offset}. "
                                 f"Must be multiple of {self.trigger_delay_resolution}")
        elif offset < 0:    # pre-trigger
            if (offset % self.pre_trigger_resolution) != 0:
                raise ValueError(f"Invalid trigger offset {offset}. "
                                 f"Must be multiple of {self.pre_trigger_resolution}")
        self._offset = offset

    @property
    def trigger_offset_range(self) -> units.IntRange:
        return units.IntRange(-16360, 34359738360) # 2**35 - 8 = 34359738360

    @property
    def pre_trigger_resolution(self) -> int:
        return 8

    @property
    def trigger_delay_resolution(self) -> int:
        return 8

    @property
    def record_length(self) -> int:
        return self._record_length

    @record_length.setter
    def record_length(self, length: int):
        # TODO should we support pyadq.ADQ_INFINITE_RECORD_LENGTH (=-1)?
        length = int(length)
        if length < self.record_length_minimum:
            raise ValueError(f"Invalid record length {length}. "
                             f"Must be greater than {self.record_length_minimum}")
        if (length % self.record_length_resolution) != 0:
            raise ValueError(f"Invalid record length {length}. "
                             f"Must be multiple of {self.record_length_resolution}")
        self._record_length = length

    @property
    def record_length_minimum(self) -> int:
        """Minimum record length."""
        return 2

    @property
    def record_length_resolution(self) -> int:
        """Resolution of the record length setting."""
        return 1
    
    @property
    def _record_size(self) -> int:
        """Data record size in bytes."""
        const_params = self._get_const_params()
        return self._record_length * const_params.channel[0].nof_bytes_per_sample
    
    @property
    def records_per_buffer(self) -> int:
        return self._records_per_buffer
    
    @records_per_buffer.setter
    def records_per_buffer(self, records: int):
        """Set the number of records per buffer."""
        records = int(records)
        if records < 1:
            raise ValueError(f"Invalid records per buffer {records}. "
                             f"Must be greater than or equal to 1.")
        self._records_per_buffer = records

    @property
    def buffers_per_acquisition(self) -> int:
        return self._buffers_per_acquisition
    
    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers: int):
        # TODO, is this a settable property in ADQ3?
        if buffers == -1:
            # -1 codes for unlimited buffers per acquisition
            pass
        else:
            if buffers < 1:
                raise ValueError(f"Attempted to set buffers per acquisition "
                                 f"{buffers}, must be ≥ 1")
        self._buffers_per_acquisition = buffers

    @property
    def buffers_allocated(self) -> int:
        return self._buffers_allocated

    @buffers_allocated.setter
    def buffers_allocated(self, buffers: int):
        buffers = int(buffers)
        if buffers > pyadq.ADQ_MAX_NOF_BUFFERS or buffers < 2:
            raise ValueError(
                f"Invalid number of buffers {buffers}. Must be an integer "
                f"between 2 and {pyadq.ADQ_MAX_NOF_BUFFERS}, inclusive.")
        self._buffers_allocated = buffers

    @property
    def timestamps_enabled(self) -> bool:
        """Enables hardware timestamps."""
        return self._timestamps_enabled
    
    @timestamps_enabled.setter
    def timestamps_enabled(self, enable: bool):
        self._timestamps_enabled = enable

    def start(self):
        # const_params = self._get_const_params()
        acq_params = self._get_acq_params()
        transf_params = self._get_transf_params()
        readout_params = self._get_readout_params()

        # Calculate buffer size using hardware-specific bytes per sample
        record_size = self._record_size
        buffer_size = self.records_per_buffer * record_size
        metadata_buffer_size = self.records_per_buffer * pyadq.SIZEOF_ADQ_GEN4_HEADER

        # TODO Limit buffer size to avoid fragmented memory alloc 

        for i in range(len(self._channels)):
            acq_params.channel[i].horizontal_offset = self._trigger_offset
            acq_params.channel[i].record_length = self._record_length
            acq_params.channel[i].nof_records = \
                self._records_per_buffer * self._buffers_per_acquisition

            transf_params.channel[i].infinite_record_length_enabled = 0
            transf_params.channel[i].dynamic_record_length_enabled = 0

            transf_params.channel[i].record_size = record_size
            transf_params.channel[i].record_buffer_size = buffer_size

            if self.timestamps_enabled:
                transf_params.channel[i].metadata_enabled = 1
                transf_params.channel[i].metadata_buffer_size = metadata_buffer_size
            # else:
            #     # there's no way to turn off metadata when in readout mode
            #     transf_params.channel[i].metadata_enabled = 0
            #     transf_params.channel[i].metadata_buffer_size = 0
            
            transf_params.channel[i].nof_buffers = self.buffers_allocated

            readout_params.channel[i].nof_record_buffers_in_array = pyadq.ADQ_FOLLOW_RECORD_TRANSFER_BUFFER

            

        self._dev.SetParameters(acq_params)
        self._dev.SetParameters(transf_params)
        self._dev.SetParameters(readout_params)
        
        self._buffers_acquired = 0
        self._t0 = None # if timestamps enabled, this will be replaced with first timestamp integer

        self._temp = np.empty((self._records_per_buffer, self._record_length), np.int16)

        result = self._dev.ADQ_StartDataAcquisition()
        if result != pyadq.ADQ_EOK:
            raise Exception(
                f"ADQ_StartDataAcquisition failed with error code {result}. See log file."
            )

    def get_next_completed_buffer(self, acq_buffer: AcquisitionProduct):
        api_buffer_array = ct.POINTER(_ADQGen4RecordArray)() 
        readout_status = _ADQDataReadoutStatus()

        chan = ct.c_int(pyadq.ADQ_ANY_CHANNEL)
        timeout_ms = 5000
        
        ret = self._dev.ADQ_WaitForRecordBuffer(
            ct.byref(chan),
            ct.cast(ct.byref(api_buffer_array), ct.POINTER(ct.c_void_p)),
            timeout_ms,
            ct.byref(readout_status),
        )

        if ret == pyadq.ADQ_EAGAIN:
            raise pyadq.Timeout("Timeout while waiting for record buffer")
        if ret < 0:
            raise pyadq.ApiError(f"ADQ_WaitForRecordBuffer failed: {ret}", error_code=ret)
        if ret == 0 and not api_buffer_array:
            # Status event (overflow etc.)
            raise pyadq.WaitForRecordBufferStatus(readout_status._to_native())
        ch = chan.value
        
        arr = api_buffer_array.contents
        assert arr.nof_records > 1

        rec0_ptr = arr.record[0]            # ADQGen4Record*
        rec0 = rec0_ptr.contents            # ADQGen4Record
        base_addr = int(ct.cast(rec0.data, ct.c_void_p).value or 0) # address of the first record in buffer
        
        total_bytes = self._records_per_buffer * self._record_size
        dst_addr = acq_buffer.data.ctypes.data  # destination base pointer
        ct.memmove(dst_addr, base_addr, total_bytes)

        if self._timestamps_enabled:
            hdr0_ptr = int(ct.cast(rec0.header, ct.c_void_p).value or 0)
            
            Raw = ct.c_ubyte * (self._records_per_buffer * pyadq.SIZEOF_ADQ_GEN4_HEADER)
            buf = Raw.from_address(hdr0_ptr)

            dt = np.dtype({
                'names'  : ['timestamp'],
                'formats': [np.uint64],
                'offsets': [8],                             # offset to timestamp
                'itemsize': pyadq.SIZEOF_ADQ_GEN4_HEADER,   # stride between headers
            })
            timestamps_raw = np.frombuffer(                     # shape (N,)
                buffer  = buf,
                dtype   = dt,
                count   = self._records_per_buffer
            )['timestamp']

            if self._t0 is None:
                self._t0 = timestamps_raw[0]
                hdr0 = ct.cast(hdr0_ptr, ct.POINTER(_ADQGen4RecordHeader)).contents
                self._time_unit = float(hdr0.time_unit)

            acq_buffer.timestamps = (timestamps_raw - self._t0) * self._time_unit
        
        res = self._dev.ADQ_ReturnRecordBuffer(ch, api_buffer_array)
        if res != pyadq.ADQ_EOK:
            raise pyadq.ApiError(f"ADQ_ReturnRecordBuffer failed: {res}", error_code=res)

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    def stop(self):
        result = self._dev.ADQ_StopDataAcquisition()
        if result not in [pyadq.ADQ_EOK, pyadq.ADQ_EINTERRUPTED]:
            raise Exception(
                f"ADQ_StartDataAcquisition failed with error code {result}. "
                f"See log file."
            )


class TeledyneDigitizer(digitizer.Digitizer):
    """
    Combines all components into a complete digitizer interface for Teledyne SP 
    Devices hardware (e.g. ADQ32).

    Args:
        device_index (int): The index of the device to use (default: 0).

    Attributes:
        channels (list[AlazarChannel]): List of configured input channels.
        sample_clock (AlazarSampleClock): Sample clock configuration.
        trigger (AlazarTrigger): Trigger configuration.
        acquire (AlazarAcquire): Acquisition settings and logic.
        aux_io (AlazarAuxiliaryIO): Auxiliary input/output configuration.

    Note:
        Ensure the digitizer hardware is correctly connected and initialized
        before creating an instance of this class.
    """
    SUPPORTED_DEVICES = {pyadq.PID_ADQ32} # Add more as we can test them

    def __init__(self, device_index: int = 0, trace_logging: bool = True, **kwargs):
        self.input_mode = digitizer.InputMode.ANALOG    # Teledyne ADQ cards are analog
        self.streaming_mode = digitizer.StreamingMode.TRIGGERED  # Only triggered modes supported

        # Create the control unit
        self._acu = ADQControlUnit()

        if trace_logging:
            self._acu.ADQControlUnit_EnableErrorTrace(pyadq.LOG_LEVEL_INFO, ".")

        # Check device list
        device_list = self._acu.ListDevices() # takes about 1-2 sec, don't call this more than once (it will add duplicate device refernces to list)
        ndevices = len(device_list)
        if ndevices < 1:
            raise RuntimeError("No digitizer devices found. At least one is required.")

        if device_index >= ndevices:
            raise RuntimeError(f"Device index {device_index} out of range. "
                               f"Found {ndevices} devices")
        
        if device_list[device_index].ProductID not in self.SUPPORTED_DEVICES:
            raise RuntimeError(f"Teledyne digitizer {device_list[device_index].ProductID} " 
                               f"no supported at this time.")
        
        self.api_revision = self._acu.api_revision
        
        self._dev = self._acu.SetupDevice(device_index) # takes about 3 sec

        chan_list = []
        for i in range(int(self._dev.ADQ_GetNofChannels())): 
            chan_list.append(TeledyneChannel(self._dev, i))
        self.channels: tuple[TeledyneChannel, ...] = tuple(chan_list)

        self.sample_clock: TeledyneSampleClock = TeledyneSampleClock(self._dev)

        self.trigger: TeledyneTrigger = TeledyneTrigger(self._dev, self.channels)

        self.acquire: TeledyneAcquire = TeledyneAcquire(self._dev, self.channels)
        
        # TODO AuxIO -- should we call this GPIO instead??
        # self.aux_io: TeledyneAuxiliaryIO = TeledyneAuxiliaryIO(self._board)

    @property
    def bit_depth(self) -> int: 
        return 12 # Note the transfer bit depth may be different

    @cached_property
    def data_range(self) -> units.IntRange:
        return units.IntRange(
            min=-2**(self.bit_depth-1),
            max=2**(self.bit_depth-1) - 1 
        )

