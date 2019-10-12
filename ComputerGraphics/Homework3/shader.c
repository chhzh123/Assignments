#include <windows.h>
#include <stdio.h>
#include <math.h>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/glut.h>

const double PI = 2*acos(0.0);

GLuint smallSphereProgram;
GLuint bigSphereProgram;

int angRot = 0;
int angRevo = 0;
float distance = 0.8;

const char* vShaderCode = "layout (location = 0) in vec3 aPos;\n"
"uniform vec2 latCIS;\n"
"varying vec2 newPos;\n"
"void main()\n"
"{\n"
" newPos.x = aPos.x * latCIS.y - aPos.z * latCIS.x;\n"
" newPos.y = aPos.z * latCIS.y + aPos.x * latCIS.x;\n"
" gl_Position = vec4(newPos.x, aPos.y, newPos.y, 1.0f);\n"
"}\n";

const char* vFragCode = "uniform vec4 newColor;\n"
"void main(void)\n"
"{ gl_FragColor = newColor; }";

const char* bigVertCode = "void main()\n"
"{ gl_Position = ftransform(); }";

const char* bigFragCode = "void main(void)\n"
"{ gl_FragColor = vec4(1,1,1,1); }";

void myDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Big sphere
    glUseProgram(bigSphereProgram);
    glutWireSphere(0.4f, 20, 20);

    // Small sphere
    float posx = (float) sin((float)angRevo/180*PI) * distance;
    float posz = (float) cos((float)angRevo/180*PI) * distance;

    glUseProgram(smallSphereProgram); // overwrite
    int vertexColorLocation = glGetUniformLocation(smallSphereProgram,"newColor");
    glUniform4f(vertexColorLocation,1.0f,1.0f,0.0f,1.0f);
    int vertexLatCISLocation = glGetUniformLocation(smallSphereProgram,"latCIS");
    glUniform2f(vertexLatCISLocation,sin((float)angRot),cos((float)angRot));
    glutWireSphere(0.1f, 8, 8);

    // glPushMatrix(); // only transform smaller one
    // // planet revolution
    // glTranslatef(posx,0,posz);
    // // planet rotation (firstly self rotate)
    // glRotatef(angRot,0,1,0);
    // // for visualization, the size of the sphere is changed linearly
    // glutWireSphere(0.1f*(posz+2*distance)/(3*distance), 8, 8);
    // glPopMatrix();

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

    glewInit();

    // make shader for the small sphere
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vShaderCode, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader,GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(vertexShader,512,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Successfully compiled!\n");
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &vFragCode, NULL);
    glCompileShader(fragmentShader);

    smallSphereProgram = glCreateProgram();
    glAttachShader(smallSphereProgram,vertexShader);
    glAttachShader(smallSphereProgram,fragmentShader);

    glLinkProgram(smallSphereProgram);
    // glDeleteShader(vertexShader);

    // make shader for the big sphere
    GLuint vertexShader2 = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader2, 1, &bigVertCode, NULL);
    glCompileShader(vertexShader2);
    int success2;
    char infoLog2[512];
    glGetShaderiv(vertexShader2,GL_COMPILE_STATUS, &success2);
    if (!success2){
        glGetShaderInfoLog(vertexShader2,512,NULL,infoLog2);
        printf("Error: %s\n", infoLog2);
    } else
        printf("Successfully compiled!\n");
    GLuint fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader2, 1, &bigFragCode, NULL);
    glCompileShader(fragmentShader2);

    bigSphereProgram = glCreateProgram();
    glAttachShader(bigSphereProgram,vertexShader2);
    glAttachShader(bigSphereProgram,fragmentShader2);

    glLinkProgram(bigSphereProgram);

    // get into display
    glutMainLoop();

    return 0;
}

// gcc shader.c -lopengl32 -lglew32 -lglut32 -o shader.exe