POINT3D_PATH = 'points3D.txt'
OUTPUT_PATH = 'out.ply'

colors = []
position = []
with open(POINT3D_PATH, "r") as pfile:
    line = pfile.readline() #remove comment line 
    while True:
        line = pfile.readline()
        if line == "":  
            break
        line_data = line.split(' ')
        image_id, x, y, z, r, g, b = line_data[:7]
        position.append(map(float,[x, y, z]))
        colors.append(map(int,[r,g,b]))
        
with open(OUTPUT_PATH, "w") as out:
    out.write("ply\n")
    out.write("format ascii 1.0\n")
    out.write("element vertex {:d}\n".format(len(position)))
    out.write("property float x\n")
    out.write("property float y\n")
    out.write("property float z\n")
    out.write("property uchar red\n")
    out.write("property uchar green\n")
    out.write("property uchar blue\n")
    out.write("end_header\n")
    for i in range(len(position)):
        x,y,z = position[i]
        r,g,b = colors[i]
        out.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(x,y,z,r,g,b)) 
