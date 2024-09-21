import re
import ast
import win32api

def quote_keys(dict_string: str):
    quoted_string = re.sub(r'(\w+):', r'"\1":', dict_string)
    return quoted_string

def as_dict(string):
    return ast.literal_eval(quote_keys(string))

def get_monitor_refresh_rate():
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return int(getattr(settings,'DisplayFrequency'))