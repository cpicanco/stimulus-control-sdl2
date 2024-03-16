from datetime import datetime, timedelta

time_format = '%H:%M:%S'
date_format = '%d/%m/%Y'

def str_to_time(time_string):
    datetime_object = datetime.strptime(time_string, time_format)
    return datetime_object.time()

def time_to_str(time_object):
    return time_object.strftime(time_format)

def str_to_date(date_string):
    return datetime.strptime(date_string, date_format).date()

def date_to_str(date_object):
    return date_object.strftime(date_format)

def str_to_timedelta(time_string):
    dt = datetime.strptime(time_string, time_format)
    return timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)

def str_to_datetime(datetime_string, format):
    return datetime.strptime(datetime_string, format)

def timedelta_to_str(td):
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def sum(secondsinput, timeinput):
    # Create a timedelta
    if isinstance(secondsinput, int):
        seconds = secondsinput

    elif isinstance(secondsinput, float):
        seconds = secondsinput

    elif isinstance(secondsinput, timedelta):
        seconds = secondsinput.seconds
    elif isinstance(secondsinput, str):
        try:
            seconds = secondsinput.replace(',', '.')
            seconds = str_to_timedelta(seconds).seconds
        except ValueError:
            seconds = float(secondsinput)

    td = timedelta(seconds=seconds)

    # Create a datetime object representing the time 08:52:21
    if isinstance(timeinput, str):
        time = datetime.strptime(timeinput, time_format)
    elif isinstance(timeinput, datetime):
        time = timeinput.time()
    elif isinstance(timeinput, timedelta):
        time = timeinput.seconds

    new_time = time + td

    return new_time.time()

class MyDate:
    def __init__(self, value, format=None):
        if isinstance(value, str):
            if format is None:
                self.value = str_to_date(value)
            else:
                self.value = datetime.strptime(value, format).date()

        elif isinstance(value, datetime):
            self.value = value.date()

    def to_string(self):
        return date_to_str(self.value)

    def __str__(self):
        return date_to_str(self.value)

class MyTime:
    def __init__(self, value):
        if isinstance(value, str):
            self.value = str_to_time(value)
        elif isinstance(value, datetime):
            self.value = value.time()

    def to_string(self):
        return time_to_str(self.value)

    def __str__(self):
        return time_to_str(self.value)

class MyDuration:
    def __init__(self, value):
        if isinstance(value, str):
            self.value = str_to_timedelta(value)
        elif isinstance(value, timedelta):
            self.value = value

    def to_string(self):
        return timedelta_to_str(self.value)

    def __str__(self):
        return timedelta_to_str(self.value)

if __name__ == '__main__':
    # print(sum(10, '08:52:21'))

    print(str_to_timedelta('08:58:43')-str_to_timedelta('08:52:21'))