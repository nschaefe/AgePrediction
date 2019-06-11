def extract_height_weight(body_field):
    """Extract height and weight from pokec body field
    parameters:
    body_field: This is a string containing the body field
    returns
    height in cm and weight in kg. if the fields are not available
    in the dataset then a string of value 'null' is returned
    """
    tokens = body_field.split(',')
    if(len(tokens) > 0):
        if('cm' in tokens[0]):
            height = tokens[0]
            # remove units
            height = ''.join(c for c in height if c.isdigit())
        else:                             
            height = 'null'
    else:
        height = 'null'

    if(len(tokens) > 1):
        if('kg' in tokens[1]):
            weight = tokens[1]
            # remove units
            weight = ''.join(c for c in weight if c.isdigit())
        else:
            weight = 'null'
    else:
        weight = 'null'
    
    return height, weight 


