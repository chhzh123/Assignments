from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import taichi as ti
ti.cfg.arch = ti.cuda  # Run on GPU by default


class Polyhedron:
    def __init__(self, obj=None, ply=None, off=None):
        self.v = []
        self.faces = []
        if obj != None:
            for line in open(obj, 'r').readlines():
                s = line.strip().split()
                if (s[0] == "v"):
                    self.v.append([float(s[1]), float(s[2]), float(s[3])])
                elif (s[0] == "f"):
                    # OBJ下标从1开始的，这里做转换
                    self.faces.append([int(s[1])-1, int(s[2])-1, int(s[3])-1])


model = Polyhedron(obj='cow.obj')


@ti.data_oriented
class Coord3Ds:
    def __init__(self, n):
        self.n = n
        self.v = ti.Vector(3, dt=ti.f32)

    def place(self, root):
        root.dense(ti.i, self.n).place(self.v)

    @ti.classkernel
    def assign(self, v: ti.template()):
        for i in self.v:
            self.v[i] = v[i]

    @ti.classkernel
    def output_to(self, v: ti.template()):
        for i in self.v:
            v[i] = self.v[i]

    @ti.classkernel
    def rotate(self, c: ti.template(), d: ti.template()):
        for i in self.v:
            self.v[i][0], self.v[i][2] = self.v[i][0]*c - \
                self.v[i][2] * d,  self.v[i][2]*c+self.v[i][0]*d

    @ti.classkernel
    def normal(self):  # 归一化
        ma = self.v[0][0]
        mi = self.v[0][0]
        for i in self.v:
            for j in ti.static(range(3)):
                ma = max(ma, self.v[i][j])
                mi = min(mi, self.v[i][j])
        for i in self.v:
            for j in ti.static(range(3)):
                self.v[i][j] = (self.v[i][j] - mi) / (ma - mi)*2-1


global_v = Coord3Ds(len(model.v))


@ti.layout
def place():
    # Place an object. Make sure you defined place for that obj
    ti.root.place(global_v)
    ti.root.lazy_grad()


v = ti.Vector(3, dt=ti.f32, shape=len(model.v))
for i in range(len(model.v)):
    v[i] = model.v[i]

global_v.assign(v)
global_v.normal()
global_v.output_to(v)
typeNum = 0


def keyboard(key, w, h):
    global typeNum
    if (key == b'r'):
        global_v.rotate(ti.cos(0.1), ti.sin(0.1))
        global_v.output_to(v)
    elif (key == b'e'):
        typeNum = (typeNum + 1) % 3
    else:
        return
    glutPostRedisplay()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    for i in range(len(model.faces)):
        p0 = v[model.faces[i][0]]
        p1 = v[model.faces[i][1]]
        p2 = v[model.faces[i][2]]
        if (typeNum != 1):
            glBegin(GL_LINES)
            glColor3f(1, 1, 1)
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p2[0], p2[1], p2[2])
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
            glEnd()
        if (typeNum != 2):
            glBegin(GL_TRIANGLES)
            glColor3f(0, 1, 1)
            glVertex3f(p0[0], p0[1], p0[2])
            glColor3f(1, 0, 1)
            glVertex3f(p1[0], p1[1], p1[2])
            glColor3f(1, 1, 0)
            glVertex3f(p2[0], p2[1], p2[2])
            glEnd()
    glFlush()


if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE)
    glutInitWindowSize(600, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("CG_PRJ2")
    glClearColor(0, 0, 0, 0)
    glShadeModel(GL_SMOOTH)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutMainLoop()
