import time

from dirigo.hw_interfaces import digitizer
from dirigo import units

from dirigo_teledyne.dirigo_teledyne import TeledyneDigitizer



digi = TeledyneDigitizer()

digi.channels[0].enabled = True
digi.channels[1].enabled = True
digi.channels[0].offset = units.Voltage('-100 mV')

print(f"Sample clock originally {digi.sample_clock.rate}")
digi.sample_clock.source = digitizer.SampleClockSource.INTERNAL
digi.sample_clock.rate = units.SampleRate("100 MS/s")
print(f"Sample clock set to {digi.sample_clock.rate}")

digi.trigger.source = digitizer.TriggerSource.EXTERNAL
digi.trigger.level = units.Voltage('250 mV')

digi.acquire.record_length = 1024
digi.acquire.records_per_buffer = 1000
digi.acquire.buffers_per_acquisition = 64
digi.acquire.buffers_allocated = 8
digi.acquire.trigger_offset = 0
digi.acquire.timestamps_enabled = True

# to use internal periodic source as trigger ...
# parameters: pyadq.ADQParameters = digi._dev.GetParameters(pyadq.ADQ_PARAMETER_ID_TOP)
# parameters.event_source.periodic.high = 0 # Reset periodic source w/ 0's
# parameters.event_source.periodic.low = 0
# parameters.event_source.periodic.period = 0
# parameters.event_source.periodic.frequency = 5e6
# parameters.acquisition.channel[0].trigger_source = pyadq.ADQ_EVENT_SOURCE_PERIODIC
# parameters.acquisition.channel[1].trigger_source = pyadq.ADQ_EVENT_SOURCE_PERIODIC
# parameters.acquisition.channel[0].trigger_edge = pyadq.ADQ_EDGE_RISING
# parameters.acquisition.channel[1].trigger_edge = pyadq.ADQ_EDGE_RISING
# digi._dev.SetParameters(parameters)

# begin
digi.acquire.start()
for i in range(2*digi.acquire.buffers_per_acquisition):
    t0 = time.perf_counter()
    digi.acquire.get_next_completed_buffer(1)
    #print(f"Time spent waiting for record buffer: {time.perf_counter()-t0}")

digi.acquire.stop()