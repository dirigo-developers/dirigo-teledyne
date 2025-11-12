import time

from dirigo.hw_interfaces import digitizer
from dirigo import units

from dirigo_teledyne.dirigo_teledyne import TeledyneDigitizer


# Start the res scanner
use_scanner = False
if use_scanner:
    from dirigo_ecu import ECU0ResonantScanner
    scanner = ECU0ResonantScanner(
        axis                        = "x",
        angle_limits                = {"min": "-13.0 deg", "max": "13.0 deg"},
        frequency                   = "7920 Hz",
        frequency_error             = 0.003,
        analog_control_range        = {"min": "0 V", "max": "5 V"},
        com_port                    = 3,
        amplitude_control_channel   = "Dev1/ao2",
        response_time               = "100 ms"
    )
    scanner.amplitude = units.Angle("10 deg")
    scanner.start()

try:
    digi = TeledyneDigitizer()

    digi.channels[0].enabled = True
    #digi.channels[1].enabled = True
    digi.channels[0].offset = units.Voltage('-100 mV')

    print(f"Sample clock originally {digi.sample_clock.rate}")
    digi.sample_clock.source = digitizer.SampleClockSource.INTERNAL
    digi.sample_clock.rate = units.SampleRate("100 MS/s")
    print(f"Sample clock set to {digi.sample_clock.rate}")

    digi.trigger.source = digitizer.TriggerSource.EXTERNAL
    #digi.trigger.external_impedance = digitizer.ExternalTriggerImpedance.HIGH
    digi.trigger.external_impedance = units.Resistance("50 ohm")
    digi.trigger.level = units.Voltage('1.9 V')

    digi.acquire.record_length = 1024
    digi.acquire.records_per_buffer = 1000
    digi.acquire.buffers_per_acquisition = 64
    digi.acquire.buffers_allocated = 8
    digi.acquire.trigger_delay = 0
    digi.acquire.timestamps_enabled = True

    digi.aux_io.configure_mode(digitizer.AuxiliaryIOMode.OUT_TRIGGER)

    # begin
    digi.acquire.start()
    for i in range(2 * digi.acquire.buffers_per_acquisition):
        t0 = time.perf_counter()
        digi.acquire.get_next_completed_buffer(acq_buffer=None)
        #print(f"Time spent waiting for record buffer: {time.perf_counter()-t0}")

    digi.acquire.stop()


    digi.aux_io.configure_mode(digitizer.AuxiliaryIOMode.DISABLE)


finally:
    scanner.stop()