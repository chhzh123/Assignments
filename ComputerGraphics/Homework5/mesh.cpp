#include <windows.h> // must be the first one to be included!
#include "mesh.h"
#include <GL/glut.h>
#include <cmath>
#include <vector>
using namespace std;

#define WIN_WIDTH 600
#define WIN_HEIGHT 600

static Mesh model;
static GLfloat angle = 0.0f;
static GLfloat pos_x = 0.0f;
static GLfloat pos_y = 0.0f;
static int mode = 1;

// init & reshape function are referenced from
// https://www.ntu.edu.sg/home/ehchua/programming/opengl/CG_Examples.html
void init(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
    glColor3f(1.0,1.0,1.0); // white
    glPointSize(2.0);
    glClearDepth(1.0f);        // Set background depth to farthest
    glEnable(GL_DEPTH_TEST);   // Enable depth testing for z-culling
    glDepthFunc(GL_LEQUAL);    // Set the type of depth-test
    glShadeModel(GL_SMOOTH);   // Enable smooth shading
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections
}

void myDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix

    // Render a color-cube consisting of 6 quads with different colors
    glLoadIdentity();                 // Reset the model-view matrix

    if (model.fmode != 3)
        glTranslatef(pos_x, pos_y, -3.0f);
    else
        glTranslatef(pos_x, -10.0f, -300.0f);
    glRotatef(angle, 0.0f, 1.0f, 0.0f);  // Rotate about (0,1,0)-axis

    for (auto face : model.faces) {
        vec3<GLfloat> v1;
        vec3<GLfloat> v2;
        vec3<GLfloat> v3;
        if (model.fmode == 1){ // obj
            v1.set(model.vertices[face.x - 1]);
            v2.set(model.vertices[face.y - 1]);
            v3.set(model.vertices[face.z - 1]);
        } else { // ply, off
            v1.set(model.vertices[face.x]);
            v2.set(model.vertices[face.y]);
            v3.set(model.vertices[face.z]);
        }
        if (mode == 1){
            glColor3f(1,1,1);
            glBegin(GL_LINES);
            glVertex3f(v1.x,v1.y,v1.z);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v1.x,v1.y,v1.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glEnd();
        }
        else if (mode == 2) {
            glBegin(GL_TRIANGLES);
            glColor3f(0.8f,0.8f,0.8f);
            glVertex3f(v1.x,v1.y,v1.z);
            glColor3f(0.7f,0.7f,0.7f);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glEnd();
        } else if (mode == 3) {
            glColor3f(1,0,0);
            glBegin(GL_LINES);
            glVertex3f(v1.x,v1.y,v1.z);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v1.x,v1.y,v1.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glEnd();
            glColor3f(1,1,1);
            glBegin(GL_TRIANGLES);
            glColor3f(0.8f,0.8f,0.8f);
            glVertex3f(v1.x,v1.y,v1.z);
            glColor3f(0.7f,0.7f,0.7f);
            glVertex3f(v2.x,v2.y,v2.z);
            glVertex3f(v3.x,v3.y,v3.z);
            glEnd();
        }
        // printf("Draw face %d %d %d\n",face.x,face.y,face.z);
    }

    glFlush();
    printf("Done display!\n");
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
    // Enable perspective projection with fovy, aspect, zNear and zFar
    if (model.fmode != 3)
        gluPerspective(45.0f, aspect, 0.1f, 100.0f);
    else
        gluPerspective(45.0f, aspect, 100.0f, 500.0f);
}

void keyPressed(unsigned char key, int x, int y)
{
    // int mod = glutGetModifiers(); // GLUT_ACTIVE_SHIFT
    printf("Pressed %c! ", key);
    
    switch (key){
        case 'r':angle -= 5.0f;printf("Rotate clockwise");break;
        case 'R':angle += 5.0f;printf("Rotate anticlockwise");break;
        case 'w':pos_y += 0.1f;printf("Move up");break;
        case 'a':pos_x -= 0.1f;printf("Move left");break;
        case 's':pos_y -= 0.1f;printf("Move down");break;
        case 'd':pos_x += 0.1f;printf("Move right");break;
        case '1':mode = 1;printf("* Change to wireframe mode");break;
        case '2':mode = 2;printf("* Change to flat mode");break;
        case '3':mode = 3;printf("* Change to flat lines");break;
    }

    printf("\n");
    myDisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);

#ifdef OBJ
    model.loadOBJ("cow.obj");
#endif
#ifdef PLY
    model.loadPLY("cactus.ply");
#endif
#ifdef OFF
    model.loadOFF("Armadillo.off");
#endif

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(50, 50);
    glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);

    glutCreateWindow("Polygon Mesh");

    glutDisplayFunc(myDisplay);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyPressed);
    init();

    // get into display
    glutMainLoop();

    return 0;
}