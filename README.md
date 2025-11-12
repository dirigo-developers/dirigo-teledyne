# dirigo-teledyne
Plugin to use [Teledyne SP Devices](https://www.spdevices.com/en-us) digitizers with [Dirigo](https://github.com/dirigo-developers/dirigo).

## Installation
Install Teledyne SP Devices ADQ driver and SDK.

Install the included official Python bindings, `pyadq`. We recommend using a virtual environment (e.g. conda). For Windows using the default installation directory:

```
pip install "C:\Program Files\SP Devices\pyadq\pyadq-YYYY.X.Y-py3-none-any.whl" 
```

Replace `YYYY.X.Y` with the version number provided.


## Devices tested
- ADQ32 – 2CH and 1CH modes
