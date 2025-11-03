from functools import cached_property
from typing import cast
import math

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


class TeledyneChannel(digitizer.Channel):
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
        super().__init__(enabled=False, inverted=False)  # Initialize parent class
        self._dev = device
        self._index = channel_index

        # Set fixed parameters
        self._coupling: digitizer.ChannelCoupling = digitizer.ChannelCoupling.DC
        self._impedance: units.Resistance = units.Resistance("50 ohm")
        self._range: units.VoltageRange = units.VoltageRange("-250 mV", "250 mV")
        
        # We may want to set nof_records to nonzero here to avoid parameter updates being ignored
    
    @property
    def index(self) -> int:
        return self._index
    
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
    
    def _get_afe_params(self):
        return cast(
            pyadq.ADQAnalogFrontendParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_ANALOG_FRONTEND)
        )

    @property
    def offset_range(self) -> units.VoltageRange:
        return units.VoltageRange("-250 mV", "250 mV")


class TeledyneSampleClock(digitizer.SampleClock):
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
    
    def _get_clk_params(self):
        return cast(
            pyadq.ADQClockSystemParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_CLOCK_SYSTEM)
        )
    
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
        Depending on the clock source, either the internal sample clock rate, or
        the user-specified external clock rate.
        """
        clk_params = self._get_clk_params()
        return units.SampleRate(clk_params.sampling_frequency)
    
    @rate.setter
    def rate(self, rate: units.SampleRate):
        if self._source is None:
            raise ValueError("`source` must be set before attempting to set `rate`")

        if self._source == digitizer.SampleClockSource.INTERNAL:
            # Check if proposed rate is within valid range
            valid_range = cast(set[units.SampleRate], self.rate_options)
            if rate not in valid_range:
                raise ValueError(f"Invalid sample clock rate: {rate}. "
                                 f"Valid options: {valid_range}")
            clk_params = self._get_clk_params()
            clk_params.sampling_frequency = float(rate)
            self._dev.SetParameters(clk_params)

        elif self._source == digitizer.SampleClockSource.EXTERNAL:
            raise NotImplementedError("External clocks not yet implemented")
            
        else:
            raise RuntimeError(f"Invalid sample clock source: {self._source}")
    
    @property
    def rate_options(self) -> set[units.SampleRate] | units.SampleRateRange:
        if self._source is None:
            raise RuntimeError("`source` must be set before attempting to access rate options")
        
        if self._source == digitizer.SampleClockSource.INTERNAL:
            # Generally fixed, but depends on card & no. channels enabled
            # TODO, get programmatically
            return {units.SampleRate("2.5 GS/s")}
        
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


class TeledyneTrigger(digitizer.Trigger):
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
        digitizer.TriggerSlope.RISING:   pyadq.ADQ_EDGE_RISING,
        digitizer.TriggerSlope.FALLING:  pyadq.ADQ_EDGE_FALLING,
    }

    def __init__(self, device: ADQ, channels: tuple[TeledyneChannel, ...]):
        self._dev = device
        self._chans = channels

        # Set parameters to None to signify that they have not been initialized
        self._slope: digitizer.TriggerSlope | None = None
        self._external_coupling: digitizer.ExternalTriggerCoupling | None = None
        self._external_range: units.VoltageRange | digitizer.ExternalTriggerRange | None = None

    def _get_acq_params(self) -> pyadq.ADQDataAcquisitionParameters:
        return cast(
            pyadq.ADQDataAcquisitionParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_ACQUISITION)
        )
    
    def _get_event_source_trig_params(self) -> pyadq.ADQEventSourcePortParameters:
        return cast(
            pyadq.ADQEventSourcePortParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_EVENT_SOURCE_TRIG)
        )

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
            # HACK: initially both channels are disabled, meaning that record_length
            # and nof_records = 0. Changing any acqusition parameters (e.g. trigger
            # source) is ignored for disabled channels. Increment these to allow
            # new values for source to stick.
            if acq_params.channel[i].record_length <= 0:
                acq_params.channel[i].record_length = 64
            if acq_params.channel[i].nof_records <= 0:
                acq_params.channel[i].nof_records = 1
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
        

class TeledyneAcquire(digitizer.Acquire):
    def __init__(self, channels: tuple[TeledyneChannel, ...]):
        self._channels = channels

        self._records_per_buffer: int = 1 # default, but in practice usually use >>1

    def _get_acq_params(self) -> pyadq.ADQDataAcquisitionParameters:
        return cast(
            pyadq.ADQDataAcquisitionParameters,
            self._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_DATA_ACQUISITION)
        )

    @property
    def trigger_delay_samples(self) -> int:
        """
        Delay between trigger event and acquisition start, in sample clock periods.
        
        Use `trigger_delay_duration` for the same setting in terms of time.
        """
        # Arguably could be part of Trigger object, but put here because of role
        # in acquisition timing
        pass

    @trigger_delay_samples.setter
    def trigger_delay_samples(self, samples: int):
        """Set the trigger delay, in sample clock periods."""
        pass
    
    @property
    def trigger_delay_duration(self) -> units.Time:
        pass # TODO, is this really needed?

    @property
    def trigger_delay_sample_resolution(self) -> int:
        pass

    @property
    def pre_trigger_samples(self): pass

    @pre_trigger_samples.setter
    def pre_trigger_samples(self, value): pass

    @property
    def pre_trigger_resolution(self): pass

    @property
    def record_length(self) -> int:
        acq_params = self._get_acq_params(self)
        return int(acq_params.channel[0].record_length)

    @record_length.setter
    def record_length(self, length: int):
        length = int(length)
        if length < self.record_length_minimum:
            raise ValueError(f"Invalid record length {length}. "
                             f"Must be greater than {self.record_length_minimum}")
        if (length % self.record_length_resolution) != 0:
            raise ValueError(f"Invalid record length {length}. "
                             f"Must be multiple of {self.record_length_resolution}")

        acq_params = self._get_acq_params(self)
        for i in range():
            acq_params.channel[i].record_length = length
        self._dev.SetParameters(acq_params)

    @property
    def record_duration(self) -> units.Time:
        """DEPRECATE"""
        pass

    @property
    def record_length_minimum(self) -> int:
        """Minimum record length."""
        return 2

    @property
    def record_length_resolution(self) -> int:
        """Resolution of the record length setting."""
        return 1
    
    @property
    def records_per_buffer(self) -> int:
        return self._records_per_buffer

    @records_per_buffer.setter
    def records_per_buffer(self, records: int):
        """Set the number of records per buffer."""
        # TODO, add validation
        acq_params = self._get_acq_params(self)
        for i in range():
            acq_params.channel[i].buffer_size = records * record_size
        self._dev.SetParameters(acq_params)



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
    def __init__(self, device_index: int = 0, **kwargs):
        self.input_mode = digitizer.InputMode.ANALOG    # Teledyne ADQ cards are analog
        self.streaming_mode = digitizer.StreamingMode.TRIGGERED  # Only triggered modes supported

        # Create the control unit
        self._acu = ADQControlUnit()

        # TODO add option for ADQ API trace logging

        # Check device list
        device_list = self._acu.ListDevices() # takes about 1-2 sec, don't call this more than once (it will add duplicate device refernces to list)
        ndevices = len(device_list)
        if ndevices < 1:
            raise RuntimeError("No digitizer devices found. At least one is required.")

        if device_index >= ndevices:
            raise RuntimeError(f"Device index {device_index} out of range. "
                               f"Found {ndevices} devices")
        
        self.api_revision = self._acu.api_revision
        
        self._dev = self._acu.SetupDevice(device_index) # takes about 3 sec

        # TODO, check device model and confirm that it matches list of supported devices

        # Build full parameter list. TODO: not sure we want to store the top level parameters
        self._adq_params = cast(
            pyadq.ADQParameters, 
            self._dev.InitializeParameters(pyadq.ADQ_PARAMETER_ID_TOP)
        )

        chan_list = []
        for i in range(int(self._dev.ADQ_GetNofChannels())): 
            chan_list.append(TeledyneChannel(self._dev, i))
        self.channels: tuple[TeledyneChannel, ...] = tuple(chan_list)

        self.sample_clock: TeledyneSampleClock = TeledyneSampleClock(self._dev)

        self.trigger: TeledyneTrigger = TeledyneTrigger(self._dev, self.channels)
        a=1
        # self.acquire: AlazarAcquire = AlazarAcquire(self._board, self.sample_clock, self.channels)
        
        # self.aux_io: AlazarAuxiliaryIO = AlazarAuxiliaryIO(self._board)

    @property
    def bit_depth(self) -> int: 
        return self.acquire._bit_depth

    @cached_property
    def data_range(self) -> units.IntRange:
        return units.IntRange(
            min=-2**(self.bit_depth-1),
            max=2**(self.bit_depth-1) - 1 
        )



if __name__ == "__main__":
    digi = TeledyneDigitizer()
    print(digi.trigger.source)
    digi.trigger.source = digitizer.TriggerSource.EXTERNAL
    print(digi.trigger.source)

    print(digi.channels[0].offset)
    digi.channels[0].offset = units.Voltage('.11 V')
    print(digi.channels[0].offset)