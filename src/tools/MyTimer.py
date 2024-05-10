import datetime

# now = datetime.datetime.now() 返回一个表示当前日期和时间的 datetime 对象。
# now.year 获取年份，以此类推
class MyTimer:
    def __init__(self):
        self.start = datetime.datetime.now()
        self.end = datetime.datetime.now()
        pass

    def stop(self):
        self.end = datetime.datetime.now()
        pass

    def total_time(self):
        return str(self.end - self.start)
        pass
    pass
