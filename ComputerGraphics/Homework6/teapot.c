#include <windows.h> // must be the first one to be included!
#include <stdlib.h>
#include <GL/glut.h>

void init(void)
{
    GLfloat ambient[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat diffuse[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat specular[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat position[] = {4.5, 4.5, 3, 1.0}; // fix position by model view matrix

    GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    GLfloat local_view[] = {0.0};

    // initialize lighting model
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, local_view);

    glFrontFace(GL_CW);
    glEnable(GL_LIGHTING); // global
    glEnable(GL_LIGHT0); // each lighting
    glEnable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST); // depth buffer
    glEndList();
}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLfloat mat[4];
    glPushMatrix();
    glTranslatef(2.0, 2.0, 0.0); // x, y, z

    /*
     * material properties
     * constants reference from
     * https://www.opengl.org/archives/resources/code/samples/redbook/teapots.c
     */
    mat[0] = 0.19225; mat[1] = 0.19225; mat[2] = 0.19225; mat[3] = 1.0; // rgb
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat);
    mat[0] = 0.50754; mat[1] = 0.50754; mat[2] = 0.50754;
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
    // mat[0] = 0.508273; mat[1] = 0.508273; mat[2] = 0.508273;
    mat[0] = 1; mat[1] = 1; mat[2] = 1; // reflect white lights
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat);
    glMaterialf(GL_FRONT, GL_SHININESS, 0.2 * 128.0); // shine
    glutSolidTeapot(1.0);

    glPopMatrix();
    glFlush();
}

/* Handler for window re-size event. Called back when the window first appears and
   whenever the window is re-sized with its new width and height */
void reshape(GLsizei width, GLsizei height) {  // GLsizei for non-negative integer
    // Compute aspect ratio of the new window
    if (height == 0) height = 1;                // To prevent divide by 0
    GLfloat aspect = (GLfloat)width / (GLfloat)height;

    // Set the viewport to cover the new window
    glViewport(0, 0, width, height);

    // Set the aspect ratio of the clipping volume to match the viewport
    glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
    glLoadIdentity();             // Reset
    // void glOrtho(GLdouble left, GLdouble right,
    //     GLdouble bottom, GLdouble top,
    //     GLdouble nearVal, GLdouble farVal);
    if (width <= height)
        glOrtho(0.0, 4.0, 0.0, 4.0*(GLfloat)height/(GLfloat)width, -10.0, 10.0);
    else
        glOrtho(0.0, 4.0*(GLfloat)width/(GLfloat)height, 0.0, 4.0, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(600,600);
    glutInitWindowPosition(50,50);
    glutCreateWindow("Teapot Lighting");
    init();
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}

// gcc teapot.c -lglu32 -lglut32 -lopengl32 -o teapot.exe