#include <windows.h>
#include <GL/glut.h>
#include <stdio.h>
#include <math.h>

const double PI = 2*acos(0.0);

int angRot = 0;
int angRevo = 0;
float distance = 0.8;

void myDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Big sphere
    glutWireSphere(0.4f, 20, 20);

    float posx = (float) sin((float)angRevo/180*PI) * distance;
    float posz = (float) cos((float)angRevo/180*PI) * distance;

    glPushMatrix(); // only transform smaller one
    // planet revolution
    glTranslatef(posx,0,posz);
    // planet rotation (firstly self rotate)
    glRotatef(angRot,0,1,0);
    // for visualization, the size of the sphere is changed linearly
    glutWireSphere(0.1f*(posz+2*distance)/(3*distance), 8, 8);
    glPopMatrix();

    glFlush();
}

void keyPressed(unsigned char key, int x, int y)
{
    // int mod = glutGetModifiers(); // GLUT_ACTIVE_SHIFT
    printf("Pressed %c!\n", key);
    switch (key){
        case 'd':angRot = (angRot + 10) % 360;break;
        case 'D':angRot = (angRot - 10) % 360;break;
        case 'y':angRevo = (angRevo + 10) % 360;break;
        case 'Y':angRevo = (angRevo - 10) % 360;break;
    }
    myDisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(500, 500);

    glutCreateWindow("Planet rotation");

    glutDisplayFunc(myDisplay);
    glutKeyboardFunc(keyPressed);

    // get into display
    glutMainLoop();

    return 0;
}

// gcc planet.c glut32.lib -lopengl32 -o planet.exe