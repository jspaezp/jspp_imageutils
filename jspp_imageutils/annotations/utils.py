

def parse_locations(marker_sites_file):
    locations = []

    with open(marker_sites_file, "r") as f:
        firstline = f.readline().strip("\n")
        firstline = firstline.split(",")
        index_x = firstline.index("X")
        index_y = firstline.index("Y")

        for line in f:
            linecontents = line.strip("\n").split(",")
            x_pos = int(float(linecontents[index_x]))
            y_pos = int(float(linecontents[index_y]))
            locations.append((x_pos, y_pos))

    return(locations)
