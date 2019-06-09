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
def get_body_from_profile(line):
    """
    Takes as input line representing one row of pokec profile and
    returns body feature (no 9)
    """
    split_line = line.split('\t')
    body = split_line[8]
    return body

def run_height_weight_extractor(data_path, out_path):
    """ Runs the height_weight_extractor function on the pokec profile
    data. Mostly used for debugging as this also prints the input
    data from pokec.
    example:
    run_height_weight_extractor('./data/soc-pokec-profiles.txt', './data/body_processd')

    """
    f = open(data_path)
    out_file = open(out_path, 'w')
    for line in f:
        body = get_body_from_profile(line)
        height, weight = extract_height_weight(body)
        out_file.write('processed: {}\t{}, \tinput: {}\n'.format(height, weight, body))

