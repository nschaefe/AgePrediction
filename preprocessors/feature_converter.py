import datetime

def date_transform(value):
    # 2012-05-25 11:20:00.0
    date = datetime.datetime.strptime(value.split()[0],  '%Y-%m-%d')
    return date.timestamp()
    
def convert(val,converter):
        if val != "null" and val != '':
            val = converter(val)
        return val

