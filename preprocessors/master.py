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
        ["user_id", "public", "completion_percentage", "gender","last_login", "registration",  "height", "weight", "comp_edu", "smoking", "martial", "age"])+'\n')

    for line in file_in:
        features = line.split('\t')

        user_id = trans.transform(features[0],trans.id_transform)
        public = trans.transform(features[1],trans.id_transform)
        completion_percentage = trans.transform(features[2],trans.id_transform)
        gender = trans.transform(features[3],trans.id_transform)
        last_login = trans.transform(features[5], trans.date_transformer)
        regist = trans.transform(features[6], trans.date_transformer)
        age = trans.transform(features[7], trans.int_transform)
        height, weight = trans.body_to_height_weight_transform(features[8])

        # stredoskolske sec school
        # zakladne basic
        # -- appear together

        # vysokoskolske academic
        # ucnovske trainee
        # studujem student
        # student student
        # pracuje working
        # bakalarske ?

        compl_edu_keywords = [('stredoskolske', 0), ('zakladne', 0), ('vysokoskolske', 1),
                              ('ucnovske', 2), ('studujem', 3), ('student', 3), ('pracuje', 4), ('bakalarske', 5)]
        comp_edu = trans.transform(
            features[19], lambda v: trans.relable_transformer(v, compl_edu_keywords))

        smoke_keywords = [('nefajcim', 0), ('fajcim', 1)]
        smoking = trans.transform(
            features[21], lambda v: trans.relable_transformer(v, smoke_keywords))

        # vztah relation
        # single no
        # slobodny no
        # zenaty married
        # vdovec witwe
        # vdova witwe
        # rozvedeny seperated
        # zadany set
        # zadana set
        martial_keywords = [('zadana', 0), ('zadany', 0), ('vztah', 0), ('single', 1), ('slobodny', 1), ('slobodna', 1),
                            ('zenaty', 2), ('vdovec', 3), ('vdova', 3), ('rozvedeny', 4)]
        martial = trans.transform(
            features[28], lambda v: trans.relable_transformer(v, martial_keywords))

        out_features = [user_id, public, completion_percentage,
                        gender, last_login, regist, height, weight, comp_edu, smoking, martial, age]
        file_out.write(seperator_write.join(map(str, out_features))+'\n')


def main(argv):
    inputfile = "../../data/soc-pokec-profiles.txt"
    outputfile = "../../data/out"

    if len(argv) != 2:
        print('master.py  <inputfile>  <outputfile>')
        sys.exit(2)

    inputfile = argv[0]
    outputfile = argv[1]

    file_in = open(inputfile)
    file_out = open(outputfile, 'w')
    process(file_in, file_out)


main(sys.argv[1:])
