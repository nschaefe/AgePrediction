def prune_graph(inpath, outpath, upto):
    f = open(inpath, 'r')
    out = []
    while True:
        line = f.readline()
        _line = line.split('\t')
        u = int(_line[0])

        v = int(_line[1].rstrip())
        if u > upto:
            break
        if v <= upto:
            out.append(line)
    outfile = open(outpath, 'w')
    outfile.writelines(out)
