from functools import cached_property
from typing import cast
import ctypes as ct

import numpy as np
from scipy import signal
import pyadq
from pyadq import ADQControlUnit, ADQ
from pyadq.structs import (
    _ADQGen4RecordArray, _ADQGen4RecordHeader, _ADQDataReadoutStatus,
    _ParameterStructs
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
    """Provides helper methods to get ADQ parameter structures."""
    _dev: "ADQ"

    def _set_params(self, params: _ParameterStructs):
        self._dev.SetParameters(params)

    def _get_params(self) -> pyadq.ADQParameters:
        return cast(
            pyadq.ADQParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_TOP)
        )
    
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
    
    def _get_port_trig_params(self) -> pyadq.ADQPortParameters:
        return cast(
            pyadq.ADQPortParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_PORT_TRIG)
        )
    
    def _get_port_gpio_params(self) -> pyadq.ADQPortParameters:
        return cast(
            pyadq.ADQPortParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_PORT_GPIOA)
        )
    
    def _get_port_sync_params(self) -> pyadq.ADQPortParameters:
        return cast(
            pyadq.ADQPortParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_PORT_SYNC)
        )
    
    def _get_fir_params(self) -> pyadq.ADQFirFilterParameters:
        return cast(
            pyadq.ADQFirFilterParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_FIR_FILTER)
        )
    
    def _get_function_params(self) -> pyadq.ADQFunctionParameters:
        return cast(
            pyadq.ADQFunctionParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_FUNCTION)
        )


class TeledyneChannel(digitizer.Channel, _TeledyneParameterMixin):
    """
    Configures the parameters for individual input channels on a Teledyne ADQ 
    digitizer.

    Properties:
        index (int): The index of the channel (0-based).
        enabled (bool): Indicates whether the channel is active during acquisition.
        coupling (ChannelCoupling): Signal coupling mode (e.g., "DC").
        impedance (Resistance): Input impedance setting (e.g., 50 ohm).
        range (VoltageRange): Voltage range for the channel.
        offset (Voltage): Analog front end DC offset voltage.
    """
    
    def __init__(self, device: ADQ, channel_index: int):
        self._dev = device
        self._index = channel_index
        # Initialize parent class, needs to be done after _dev attribute set
        super().__init__(enabled=False, inverted=False)  

        # Set fixed parameters
        self._coupling: digitizer.ChannelCoupling = digitizer.ChannelCoupling.DC
        self._impedance: units.Resistance = units.Resistance("50 ohm")
        self._range: units.VoltageRange = units.VoltageRange("-250 mV", "250 mV")

        # offset parameter is allowed to change
        
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def enabled(self) -> bool:
        """Indicates whether the channel is enabled for acquisition."""
        # TODO, is it better to query nof_records? See setter
        return self._enabled
    
    
    @enabled.setter
    def enabled(self, enable: bool):
        """Enable or disable the channel."""
        if not isinstance(enable, bool):
            raise ValueError("`enabled` must be set with a boolean")
        
        # In ADQ3, channels are 'enabled' if they have nof_records > 0
        acq_params = self._get_acq_params()
        acq_params.channel[self.index].nof_records = 1 if enable else 0
        acq_params.channel[self.index].record_length = 2 if enable else 0
        self._set_params(acq_params)

        self._enabled = enable # save state internally

    @property
    def coupling(self) -> digitizer.ChannelCoupling:
        return self._coupling   # fixed DC
    
    @coupling.setter
    def coupling(self, coupling: digitizer.ChannelCoupling):
        if coupling not in self.coupling_options:
            raise ValueError(f"Invalid input coupling {coupling}. "
                             f"Valid options are: {self.coupling_options}")
        self._coupling = coupling

    @property
    def coupling_options(self) -> set[digitizer.ChannelCoupling]:
        # Teledyne ADQ32 only supports DC coupling
        return {digitizer.ChannelCoupling.DC}
    
    @property
    def impedance(self) -> units.Resistance:
        return self._impedance      # fixed at 50 ohm
    
    @impedance.setter
    def impedance(self, impedance: units.Resistance):
        if impedance not in self.impedance_options:
            raise ValueError(f"Invalid input impedance {impedance}. "
                             f"Valid options are: {self.impedance_options}")
        self._impedance = impedance

    @property
    def impedance_options(self) -> set[units.Resistance]:
        # Teledyne ADQ32 only supports 50 ohm impedance
        return {units.Resistance("50 ohm")}
    
    @property
    def input_range(self) -> units.VoltageRange:
        # Teledyne only supports 0.5 Vpp
        # TODO, should this range change depending on offset?
        return self._range
    
    @input_range.setter
    def input_range(self, rng: units.VoltageRange):
        if rng not in self.range_options:
            raise ValueError(f"Invalid input range {rng}. "
                             f"Valid options are: {self.range_options}")
        self._range = rng
    
    @property
    def range_options(self) -> set[units.VoltageRange]:
        return {
            units.VoltageRange("-250 mV", "250 mV"),  # 0.5 V peak-to-peak
        }
    
    @property
    def offset(self) -> units.Voltage:
        afe_params = self._get_afe_params()
        return units.Voltage(afe_params.channel[self._index].dc_offset / 1000) # mV -> V

    @offset.setter
    def offset(self, offset: units.Voltage): # returns in <4 ms
        if not self.offset_range.within_range(offset):
            raise ValueError(f"Invalid offset {offset}. "
                             f"Valid range: {self.offset_range}")
        
        afe_params = self._get_afe_params()
        afe_params.channel[self._index].dc_offset = float(offset) * 1000 # V -> mV
        self._set_params(afe_params)

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

        # TODO sampling rate depends on # channels enabled at firmware level
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
            raise NotImplementedError("External sample clock not yet implemented.")
        
        self._set_params(clk_params)

        self._source = source
    
    @property
    def source_options(self) -> set[digitizer.SampleClockSource]:
        # Teledyne ADQ32 supports internal and external clocks
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
            raise ValueError("Sample clock source must be set before attempting "
                             "to set sample clock rate")

        if self._source == digitizer.SampleClockSource.INTERNAL:
            # Check if proposed rate matches a valid rate
            valid_rates = cast(set[units.SampleRate], self.rate_options) # cast b/c we know only discrete rates are possible
            if rate not in valid_rates:
                # TODO, do we want to support rounding to nearest rate, say when the proposed rate is within 1%?
                raise ValueError(f"Invalid sample clock rate: {rate}. "
                                 f"Valid options: {valid_rates}")
            
            skip_params = self._get_skip_params()
            skip = round(self._base_sampling_rate / rate)
            for i in range(int(self._dev.ADQ_GetNofChannels())):
                skip_params.channel[i].skip_factor = skip
            self._set_params(skip_params)
            # self._configure_fir_for_skip(skip)  # TESTING

        elif self._source == digitizer.SampleClockSource.EXTERNAL:
            raise NotImplementedError("External sample clock not yet implemented")
            
        else:
            raise RuntimeError(f"Invalid sample clock source: {self._source}")
    
    @property
    def _valid_skip_factors(self) -> list[int]:
        # Note that this is a subset of the actual supported skip factors
        # providing a list of all the supported values (4 million) would take too long
        nchannels = self._dev.ADQ_GetNofChannels()
        high_range = [20, 25, 50, 100, 125, 250, 500, 1250, 2500, 5000]
        if nchannels == 2:
            return [1, 2, 4, 5, 8, 9, 10] + high_range
        elif nchannels == 1:
            return [1, 2, 4, 5, 8, 16, 17, 18] + high_range
        else:
            raise ValueError("Unsupported channel configuration for ADQ32")
        
    def _configure_fir_for_skip(self, skip: int):
        c = self._get_const_params().channel[0].fir_filter
        max_half = int(c.nof_coefficients)
        order = 2*(max_half - 1)    # Max full taps: N+1 = 2*max_half - 1

        # filter design
        Fs = float(self._base_sampling_rate)
        f_pass, f_stop = 0.45*(Fs/skip), 0.55*(Fs/skip)
        taps = signal.remez(
            numtaps = order + 1,
            bands   = [0, f_pass, f_stop, Fs/2],
            desired = [1, 0], 
            fs      = Fs
        )
        
        half_len = order//2 + 1
        one_side = taps[:half_len].copy()

        # apply FIR filter
        fir_params = self._get_fir_params()
        for ch in range(int(self._dev.ADQ_GetNofChannels())):
            chp = fir_params.channel[ch]
            chp.rounding_method = pyadq.ADQ_ROUNDING_METHOD_TIE_AWAY_FROM_ZERO
            chp.format = pyadq.ADQ_COEFFICIENT_FORMAT_DOUBLE
            # zero entire array, then write one-sided sequence
            for i in range(pyadq.ADQ_MAX_NOF_FILTER_COEFFICIENTS):
                chp.coefficient[i] = 0.0
            for i in range(min(half_len, pyadq.ADQ_MAX_NOF_FILTER_COEFFICIENTS)):
                chp.coefficient[i] = float(one_side[i])
        self._set_params(fir_params)
    
    @property
    def rate_options(self) -> set[units.SampleRate] | units.SampleRateRange:
        if self._source is None:
            raise RuntimeError("`source` must be set before attempting to access rate options")
        
        if self._source == digitizer.SampleClockSource.INTERNAL:
            return {self._base_sampling_rate / s for s in self._valid_skip_factors}
        
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
        level (Voltage): The trigger level in volts.
        external_coupling (ExternalTriggerCoupling): Coupling mode for the external trigger (e.g., "DC").
        external_impedance (Resistance | ImpedanceMode): Impedance for external trigger.
        external_range (VoltageRange): Voltage range for the external trigger.
    
    Notes: With ADQ3, it is possible to configure separate event triggers 
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
        self._external_impedance: units.Resistance | digitizer.ImpedanceMode | None = None
        self._external_range: units.VoltageRange | None = None    

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
            # TODO test blocking with pattern generators
            #acq_params.channel[i].trigger_blocking_source = pyadq.ADQ_FUNCTION_INVALID 
            acq_params.channel[i].trigger_blocking_source = pyadq.ADQ_FUNCTION_PATTERN_GENERATOR0 
        self._set_params(acq_params)
   
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

        # Additional ADQ modes not (yet) supported: periodic, SYNC, GPIO
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
        self._set_params(acq_params)

    @property
    def slope_options(self) -> set[digitizer.TriggerSlope]:
        return {digitizer.TriggerSlope.RISING, digitizer.TriggerSlope.FALLING}

    @property
    def level(self) -> units.Voltage:
        if self.source == digitizer.TriggerSource.EXTERNAL:
            trig_params = self._get_event_source_trig_params()
            return units.Voltage(trig_params.pin[0].threshold)
        else:
            raise NotImplementedError("Trigger sources other than external trigger "
                                      "(port TRIG) not yet supported.")
        
    @level.setter
    def level(self, level: units.Voltage):
        if self.source != digitizer.TriggerSource.EXTERNAL:
            raise NotImplementedError("Trigger sources other than external trigger "
                                      "(port TRIG) not yet supported.")
        
        if not self.level_limits.within_range(level):
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        trig_params = self._get_event_source_trig_params()
        trig_params.pin[0].threshold = float(level)
        self._set_params(trig_params)
    
    @property
    def level_limits(self) -> units.VoltageRange:
        if self.source == digitizer.TriggerSource.EXTERNAL:
            # will need to switch on TRIG port impedance
            return units.VoltageRange("0 V", "2.8 V")
        elif self.source in [digitizer.TriggerSource.CHANNEL_A, digitizer.TriggerSource.CHANNEL_B]:
            raise NotImplementedError("Triggering on Channels not implemented yet")
        else:
            raise RuntimeError("Invalid trigger source")
    
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
    def external_impedance(self) -> units.Resistance | digitizer.ImpedanceMode:
        port_trig_params = self._get_port_trig_params()
        api_imp = port_trig_params.pin[0].input_impedance
        if api_imp == pyadq.ADQ_IMPEDANCE_50_OHM:
            return units.Resistance("50 ohm")
        elif api_imp == pyadq.ADQ_IMPEDANCE_HIGH:
            return digitizer.ImpedanceMode.HIGH
        else:
            raise NotImplementedError(f"Unsupported trigger mode {api_imp}")

    @external_impedance.setter
    def external_impedance(self, imp: units.Resistance | digitizer.ImpedanceMode):
        if imp not in self.external_impedance_options:
            raise ValueError(f"Invalid external trigger impedance {imp}"
                             f"Supported: {self.external_impedance_options}")
        if imp == units.Resistance("50 ohm"):
            api_imp = pyadq.ADQ_IMPEDANCE_50_OHM
        elif imp == digitizer.ImpedanceMode.HIGH:
            api_imp = pyadq.ADQ_IMPEDANCE_HIGH
        else:
            raise ValueError(f"Unsupported external trigger impedance {imp}")
        
        port_trig_params = self._get_port_trig_params()
        port_trig_params.pin[0].input_impedance = api_imp
        self._set_params(port_trig_params)

    @property
    def external_impedance_options(self) -> set[units.Resistance | digitizer.ImpedanceMode]:
        return {units.Resistance("50 ohm"), digitizer.ImpedanceMode.HIGH}

    @property
    def external_range(self):
        # For ADQ32, a single range is available, but depends on impedance
        return units.VoltageRange("0 V", "2.8 V")

    @property
    def external_range_options(self):
        return {units.VoltageRange("0 V", "2.8 V")}
        

class TeledyneAcquire(digitizer.Acquire, _TeledyneParameterMixin):
    def __init__(self, device: ADQ, channels: tuple[TeledyneChannel, ...]):
        super().__init__()
        self._dev = device
        self._channels = channels

        self._trigger_delay: int = 0
        self._record_length: int = 0

        self._records_per_buffer: int = 1 # default, but in practice usually use >>1
        self._buffers_per_acquisition: int = 1
        self._buffers_allocated: int = 1
        self._timestamps_enabled: bool = True
        self._t0: int | None = None # (integer) timestamp of first record, to be filled when available 

        self._buffers_acquired: int = -1 # the start sequence should always reset this to 0

    @property
    def trigger_delay(self) -> int:
        return self._trigger_delay

    @trigger_delay.setter
    def trigger_delay(self, offset: int):
        offset = int(offset)
        if not self.trigger_delay_range.within_range(offset):
            raise ValueError(f"Invalid trigger offset {offset}. "
                             f"Valid range: {self.trigger_delay_range}")
        if offset > 0:      # delay
            if (offset % self.post_trigger_resolution) != 0:
                raise ValueError(f"Invalid trigger offset {offset}. "
                                 f"Must be multiple of {self.post_trigger_resolution}")
        elif offset < 0:    # pre-trigger
            if (offset % self.pre_trigger_resolution) != 0:
                raise ValueError(f"Invalid trigger offset {offset}. "
                                 f"Must be multiple of {self.pre_trigger_resolution}")
        self._offset = offset

    @property
    def trigger_delay_range(self) -> units.IntRange:
        return units.IntRange(-16360, 34359738360) # 2**35 - 8 = 34359738360

    @property
    def pre_trigger_resolution(self) -> int:
        return 8

    @property
    def post_trigger_resolution(self) -> int:
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
        """Starts acquistion using the "data readout" mode."""
        # Calculate buffer size using hardware-specific bytes per sample
        record_size = self._record_size
        buffer_size = self.records_per_buffer * record_size
        metadata_buffer_size = self.records_per_buffer * pyadq.SIZEOF_ADQ_GEN4_HEADER
        print(f"Buffer size (MB): {buffer_size/1024/1024:.2f}")

        # TODO Limit buffer size to avoid fragmented memory alloc, what is the hard limit? 

        acq_params = self._get_acq_params()
        transf_params = self._get_transf_params()
        readout_params = self._get_readout_params()
        for i in range(len(self._channels)):
            acq_params.channel[i].horizontal_offset = self._trigger_delay
            acq_params.channel[i].record_length = self._record_length
            acq_params.channel[i].nof_records = \
                max(self._records_per_buffer * self._buffers_per_acquisition, -1) # -1 codes for unlimited

            transf_params.channel[i].infinite_record_length_enabled = 0
            transf_params.channel[i].dynamic_record_length_enabled = 0

            transf_params.channel[i].record_size = record_size
            transf_params.channel[i].record_buffer_size = buffer_size

            transf_params.channel[i].metadata_enabled = 1 # there's no way to turn off metadata when in readout mode
            transf_params.channel[i].metadata_buffer_size = metadata_buffer_size
            
            transf_params.channel[i].nof_buffers = self.buffers_allocated
            
            readout_params.channel[i].nof_record_buffers_in_array = pyadq.ADQ_FOLLOW_RECORD_TRANSFER_BUFFER
            
        self._set_params(acq_params)
        self._set_params(transf_params)
        self._set_params(readout_params)
        
        self._buffers_acquired = 0
        self._t0 = None # if timestamps enabled, this will be replaced with first timestamp integer

        result = self._dev.ADQ_StartDataAcquisition()
        if result != pyadq.ADQ_EOK:
            raise Exception(
                f"ADQ_StartDataAcquisition failed with error code {result}. See log file."
            )
        self._active.set()

    def get_next_completed_buffer(self, acq_buffer: AcquisitionProduct):
        
        api_buffer_array = ct.POINTER(_ADQGen4RecordArray)() 
        readout_status = _ADQDataReadoutStatus()

        chan = ct.c_int(pyadq.ADQ_ANY_CHANNEL)
        timeout_ms = 5000 # TODO don't hardcode this
        
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
        
        self._buffers_acquired += 1

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
        self._active.clear()


class TeledyneAuxiliaryIO(digitizer.AuxiliaryIO, _TeledyneParameterMixin):
    """Configures behavior of SYNC and GPIO ports."""
    def __init__(self, device: ADQ):
        self._dev = device
        self._mode: digitizer.AuxiliaryIOMode | None = None

    def configure_mode(self, mode: digitizer.AuxiliaryIOMode, **kwargs):
        if mode == digitizer.AuxiliaryIOMode.DISABLE:
            params = self._get_params()

            # Disable SYNC
            gpio_params = params.port[pyadq.ADQ_PORT_SYNC].pin[0]
            gpio_params.function       = pyadq.ADQ_FUNCTION_INVALID
            gpio_params.direction      = pyadq.ADQ_DIRECTION_OUT

            # Disable GPIO
            gpio_params = params.port[pyadq.ADQ_PORT_GPIOA].pin[0]
            gpio_params.function       = pyadq.ADQ_FUNCTION_INVALID
            gpio_params.direction      = pyadq.ADQ_DIRECTION_OUT

            self._set_params(params)

        elif mode == digitizer.AuxiliaryIOMode.OUT_TRIGGER:
            # This invovles 2 parts:
            # 1. Buffer the trigger signal out to SYNC
            # 2. Emit a pulse on GPIO corresponding to beginning of acquisition 
            # TODO, make the pulse duration adjustable

            # Output trigger pulses in SYNC
            func_params = self._get_function_params()
            func_params.pulse_generator[0].source   = pyadq.ADQ_EVENT_SOURCE_TRIG
            func_params.pulse_generator[0].edge     = pyadq.ADQ_EDGE_RISING # not sure what this does?
            func_params.pulse_generator[0].length   = -1 # TODO, does -1 work like we expect?
            self._set_params(func_params)
 
            sync_params = self._get_port_sync_params()
            sync_params.pin[0].function       = pyadq.ADQ_FUNCTION_PULSE_GENERATOR0
            sync_params.pin[0].direction      = pyadq.ADQ_DIRECTION_OUT
            sync_params.pin[0].invert_output  = 0
            self._set_params(sync_params)

            func_params = self._get_function_params()
            pg0_params = func_params.pattern_generator[0]
            pg0_params.nof_instructions = 2

            pg0_params.instruction[0].op = pyadq.ADQ_PATTERN_GENERATOR_OPERATION_EVENT
            pg0_params.instruction[0].count = 1
            pg0_params.instruction[0].output_value = 1
            pg0_params.instruction[0].output_value_transition = 0
            pg0_params.instruction[0].source = pyadq.ADQ_EVENT_SOURCE_TRIG
            pg0_params.instruction[0].source_edge = pyadq.ADQ_EDGE_RISING
            pg0_params.instruction[0].reset_source = pyadq.ADQ_FUNCTION_INVALID

            pg0_params.instruction[1].op = pyadq.ADQ_PATTERN_GENERATOR_OPERATION_EVENT
            pg0_params.instruction[1].count = 1
            pg0_params.instruction[1].output_value = 0
            pg0_params.instruction[1].output_value_transition = 0
            pg0_params.instruction[1].source = pyadq.ADQ_FUNCTION_INVALID

            pg1_params = func_params.pattern_generator[1]
            pg1_params.nof_instructions = 3

            step = self._timer_count_step
            pg1_params.instruction[0].op = pyadq.ADQ_PATTERN_GENERATOR_OPERATION_TIMER
            pg1_params.instruction[0].count = step # will give the min valid count
            pg1_params.instruction[0].output_value = 0
            pg1_params.instruction[0].output_value_transition = 0
            pg1_params.instruction[0].source = pyadq.ADQ_EVENT_SOURCE_TRIG
            pg1_params.instruction[0].source_edge = pyadq.ADQ_EDGE_RISING

            pg1_params.instruction[1].op = pyadq.ADQ_PATTERN_GENERATOR_OPERATION_TIMER

            pg1_params.instruction[1].count = round(250_000/step) * step # 100 us @ 2.5 GHz, 50 us @ 5 GHz
            pg1_params.instruction[1].output_value = 1
            pg1_params.instruction[1].output_value_transition = 1

            pg1_params.instruction[2].op = pyadq.ADQ_PATTERN_GENERATOR_OPERATION_EVENT
            pg1_params.instruction[2].count = 1
            pg1_params.instruction[2].output_value = 0
            pg1_params.instruction[2].output_value_transition = 0
            pg1_params.instruction[2].source = pyadq.ADQ_FUNCTION_INVALID

            self._set_params(func_params)

            gpio_params = self._get_port_gpio_params()
            gpio_params.pin[0].function       = pyadq.ADQ_FUNCTION_PATTERN_GENERATOR1
            gpio_params.pin[0].direction      = pyadq.ADQ_DIRECTION_OUT
            gpio_params.pin[0].invert_output  = 0
            self._set_params(gpio_params)

        else:
            raise NotImplementedError(f"Unsupported Aux IO mode {mode}")

        self._mode = mode

    def read_input(self) -> bool:
        raise NotImplementedError

    def write_output(self, state: bool):
        raise NotImplementedError
    
    @property
    def _timer_count_step(self) -> int:
        nchannels = self._dev.ADQ_GetNofChannels()
        if nchannels == 2:
            return 8
        elif nchannels == 1:
            return 16
        else:
            raise RuntimeError(f"Invalid number of channels detected {nchannels}")


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
        
        self.aux_io: TeledyneAuxiliaryIO = TeledyneAuxiliaryIO(self._dev)

    @property
    def bit_depth(self) -> int: 
        # TODO, bit depth is really 12
        return 16 # Note the transfer bit depth may be different

    @cached_property
    def data_range(self) -> units.IntRange:
        return units.IntRange(
            min=-2**(self.bit_depth-1),
            max=2**(self.bit_depth-1) - 1 
        )

pyadq.ADQ_FUNCTION_INVALID