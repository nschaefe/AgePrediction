import datetime
from utils import get_words
import re

na_value='NA'

def date_transformer(value):
    # 2012-05-25 11:20:00.0
    date = datetime.datetime.strptime(value.split()[0],  '%Y-%m-%d')
    return (int)(date.timestamp())


def body_to_height_weight_transform(body_field):
    """Extract height and weight from pokec body field
    parameters:
    body_field: This is a string containing the body field
    returns
    height in cm and weight in kg. if the fields are not available
    in the dataset then a string of value na_value is returned
    """
    # TODO 168cm11 -> 16811 
    # TODO 175cm, 86 -> 175, NA : This case seems too specific to solve
    # Trying to solve this might cause more problems.
    # TODO 'vyska tak nad 170cm a vaha? kolem 60kg' -> 17060, NA

    # make sure that value fits in int
    max_val=10000

    max_height, min_height = 250, 50
    max_weight, min_weight = 400, 25 
    height_program = re.compile(r"[0-9]+(\s*)cm")
    weight_program = re.compile(r"[0-9]+(\s*)kg")

    height = re.search(height_program, body_field) # Search for the pattern
    if height is None:
        height = na_value
    else:
        height = height.group(0) # take first match
        height = re.sub(r'(\s*)cm', '', height) # remove unit
        # Finally check outlier
        if int(height) >= max_height or int(height) <= min_height:
            height = na_value

    weight = re.search(weight_program, body_field)
    if weight is None:
        weight = na_value
    else:
        weight = weight.group(0)
        weight = re.sub(r'(\s*)kg', '', weight)
        if int(weight) >= max_weight or int(weight) <= min_weight:
            weight = na_value
            
    return height, weight


def int_transform(val):
    num = (int)(val)
    # TODO analyze outliers later
    if num > 0:
        return num
    else:
        return na_value

def id_transform(val):
    return val


def relable_transformer(val, keywords, no_hit_to_null=True):
    val_clean = ' '.join(get_words(val))
    for keyword, repl in keywords:
        if keyword in val_clean:
            return repl
    if no_hit_to_null:
        return na_value
    else:
        return val


def transform(val, transformer):
    if val == "null" or val == " ":
        return na_value

    return transformer(val)
