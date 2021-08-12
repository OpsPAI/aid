from math import ceil, floor
import time


class TimestampAgg:
    @staticmethod
    def toSecond(ts):
        """
        change timestamp to time string
        """
        timeArray = time.localtime(ts)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return otherStyleTime

    @staticmethod
    def toMinute(ts):
        """
        change timestamp to time string
        """
        timeArray = time.localtime(ts)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M", timeArray)
        return otherStyleTime

    @staticmethod
    def toFreqMinute(ts, freq):
        """
        change timestamp to time string
        """
        seconds = freq * 60
        timeArray = time.localtime(floor(ts / float(seconds)) * seconds)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M", timeArray)
        return otherStyleTime
