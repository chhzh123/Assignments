#include <windows.h> // must be the first one to be included!
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>

#define WIN_WIDTH 600
#define WIN_HEIGHT 600

class Point
{
public:
	Point() : x(0), y(0) {}
	Point(int px, int py) {
		set(px,py);
	}
    void set(int px, int py) {
        this->x = px;
        this->y = py;
    }
    int x, y;
};

static int num_points = 0;
static Point points[4];

void init(void)
{
    glClearColor(1.0, 1.0, 1.0, 0); // set bg color -> black
    glColor3f(0.0,0.0,0.0); // drawing color -> white
    glPointSize(2.0);
    // be careful: need to set projection!
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0,WIN_WIDTH,0.0,WIN_HEIGHT,1,-1);
}

void drawPoint(Point p) {
    glBegin(GL_POINTS);
    glVertex2f(p.x, p.y);
    glEnd();
    glFlush();
}

void drawLine(Point p1, Point p2) {
    glBegin(GL_LINES);
    glVertex2f(p1.x,p1.y);
    glVertex2f(p2.x,p2.y);
    glEnd();
    glFlush();
}

Point drawBezier(Point p1, Point p2, Point p3, Point p4, double t) {
	// B(t) = P_0 (1-t)^3 + 3P_1 t(1-t)^2 + 3P_2 t^2(1-t) + P_3 t^3, t\in[0,1]
    double a1 = pow((1 - t), 3);
    double a2 = 3 * t * pow((1 - t), 2);
    double a3 = 3 * pow(t, 2) * (1 - t);
    double a4 = pow(t, 3);
    Point p(a1*p1.x + a2*p2.x + a3*p3.x + a4*p4.x,
    		a1*p1.y + a2*p2.y + a3*p3.y + a4*p4.y);
    return p;
}

void myDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
}

void mouseKicked(int button, int state, int x, int y) {
    if (state == GLUT_DOWN)
    {
    	// be careful that y increases from top to bottom
        points[num_points].set(x,WIN_HEIGHT-y);

        // draw point
        glColor3f(1.0,0.0,0.0); // red
        if (num_points == 0) // clear the previous curve
        	myDisplay();
        drawPoint(points[num_points]);

        // draw line
        glColor3f(1.0,0.0,0.0); // red
        if (num_points > 0)
        	drawLine(points[num_points-1], points[num_points]);

        // update num_points
        if (num_points == 3) {

            glColor3f(0.0,0.0,1.0); // blue

            // draw curve in small segements
            Point p_curr = points[0];
            for (double t = 0.0; t <= 1.0; t += 0.01)
            {
                Point p_new = drawBezier(points[0], points[1], points[2], points[3], t);
                drawLine(p_curr, p_new);
                p_curr = p_new;
            }

            num_points = 0;
        } else {
            num_points++;
        }
    }
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
    glutCreateWindow("Cubic interactive Bezier curve");
    printf("Please click left button of mouse to input control point of Bezier curve!\n");

    init();
    glutMouseFunc(mouseKicked);
    glutDisplayFunc(myDisplay); 
    glutMainLoop();
    return 0;
}

// g++ bezier.cpp -lglut32 -lopengl32 -o bezier.exe