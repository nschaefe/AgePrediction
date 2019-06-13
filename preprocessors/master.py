import feature_transform as trans
import sys
import getopt


# TODO
# region
# martial status
# completed lvl of education


def process(file_in, file_out):
    seperator_write = '\t'
    file_out.write(seperator_write.join(
        ["user_id", "public", "completion_percentage", "last_login", "gender", "registration",  "height", "weight", "smoking"])+'\n')

    for line in file_in:
        features = line.split('\t')

        user_id = features[0]
        public = features[1]
        completion_percentage = features[2]
        gender = features[3]
        last_login = trans.transform(features[5], trans.date_transformer)
        regist = trans.transform(features[6], trans.date_transformer)
        height, weight = trans.body_to_height_weight_transform(features[8])

        smoke_keywords = [('nefajcim', 0), ('fajcim', 1)]
        smoking = trans.transform(
            features[21], lambda v: trans.relable_transformer(v, smoke_keywords))

        out_features = [user_id, public, completion_percentage,
                        gender, last_login, regist, height, weight, smoking]
        file_out.write(seperator_write.join(map(str, out_features))+'\n')


def main(argv):
    inputfile = "../../data/soc-pokec-profiles.txt"
    outputfile = "../../data/out"

    if len(argv) != 2:
        print('master.py  <inputfile>  <outputfile>')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(argv, "h", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('master.py  <inputfile>  <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('master.py  <inputfile>  <outputfile>')
            sys.exit()
        elif opt in ("--ifile"):
            inputfile = arg
        elif opt in ("--ofile"):
            outputfile = arg

    file_in = open(inputfile)
    file_out = open(outputfile, 'w')

    process(file_in, file_out)


main(sys.argv[1:])
