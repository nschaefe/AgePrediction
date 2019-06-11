import feature_converter as conv
import extract_height_weight as exhw
import sys
import getopt


def process(file_in, file_out):
    for line in file_in:
        features = line.split('\t')

        user_id = features[0]
        last_login = conv.convert(features[5], conv.date_transform)
        regist = conv.convert(features[6], conv.date_transform)
        height, weight = exhw.extract_height_weight(features[8])

        smoke_keywords = [('nefajcim', 0), ('fajcim', 1)]
        smoking = conv.relable(features[21], smoke_keywords)

        out_features = [user_id, last_login, regist, weight, height, smoking]
        file_out.write('\t'.join(map(str, out_features))+'\n')


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
