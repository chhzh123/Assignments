import taichi as ti


def getX():
    logo = [[
        "  ####  ",
        " ##  ## ",
        " #    # ",
        " #      ",
        "  #     ",
        "   ##   ",
        "     ## ",
        " #    # ",
        " #    # ",
        " ##  ## ",
        "  ####  "
    ], [
        " #    # ",
        " #    # ",
        "  #  #  ",
        "  #  #  ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   "
    ], [
        "  ####  ",
        " ##  ## ",
        " #    # ",
        " #      ",
        "  #     ",
        "   ##   ",
        "     ## ",
        " #    # ",
        " #    # ",
        " ##  ## ",
        "  ####  "
    ], [
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " #    # ",
        " ##  ## ",
        "  ####  "
    ]]
    x = []
    for i in range(len(logo)):
        for X in range(len(logo[i])):
            for Y in range(len(logo[i][X])):
                if logo[i][X][Y] == '#':
                    x.append([i/len(logo)+Y/len(logo[i][X])/len(logo),
                              0.5+0.5/len(logo)-X/len(logo[i])/len(logo)])
    return x


ti.cfg.arch = ti.cuda
x_logo = getX()
n_particles = len(x_logo)
dim = 2
spin = [ti.cos(0.1), ti.sin(0.1)]
x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
for i in range(n_particles):
    x[i] = x_logo[i]


@ti.func
def complexMul(a: ti.f32, b: ti.f32, c: ti.f32, d: ti.f32):  # 复数乘法，用于旋转
    return [a*c-b*d, b*c+a*d]


@ti.kernel
def substep():
    for p in x:
        x[p] = x[p]-0.5
        x[p] = complexMul(x[p][0], x[p][1], spin[0], spin[1])
        x[p] = x[p]+0.5


gui = ti.core.GUI("SYSU_CG_COURSE", ti.veci(512, 512))
canvas = gui.get_canvas()
for frame in range(19260817):
    substep()

    canvas.clear(0)
    pos = x.to_numpy(as_vector=True)
    for i in range(n_particles):
        canvas.circle(ti.vec(pos[i, 0], pos[i, 1])).radius(
            7).color(87*256+33).finish()  # 中大绿
    gui.update()
