import datetime

def date_transform(value):
    # 2012-05-25 11:20:00.0
    date = datetime.datetime.strptime(value.split()[0],  '%Y-%m-%d')
    return date.timestamp()

def convert(data, converter):
    for i in range(0, len(data)):
        val=data[i]
        if val != "null" and val != '':
            data[i] = converter(val)
    


feature_file = "../data/last_login"
text_file = open(feature_file, "r")
data = text_file.read().split('\n')

convert(data,date_transform)

write_file = open(feature_file+".converted", "w")
write_file.write('\n'.join(map(str,data)))


