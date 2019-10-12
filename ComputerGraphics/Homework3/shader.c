#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

const double PI = 2 * acos(0.0);

GLuint smallSphereProgram;
GLuint bigSphereProgram;

int angRot = 0;
int angRev = 0;
float distance = 0.8;

// CIS: \cos(\theta) + i \sin(\theta)
const char* smallVertCode = "layout (location = 0) in vec3 aPos;\n"
"uniform vec2 latCIS;\n"
"uniform vec2 revCIS;\n"
"uniform vec3 center;\n"
"varying vec3 newPos;\n"
"void main()\n"
"{\n"
"  newPos.x = aPos.x * latCIS.y - aPos.z * latCIS.x;\n"
"  newPos.z = aPos.z * latCIS.y + aPos.x * latCIS.x;\n"
"  newPos.x += center.x;\n"
"  newPos.z += center.z;\n"
"  newPos.x = (newPos.x - center.x) * revCIS.y - (newPos.z - center.z) * revCIS.x + center.x;\n"
"  newPos.z = (newPos.z - center.z) * revCIS.y + (newPos.x - center.x) * revCIS.x + center.z;\n"
"  gl_Position = vec4(newPos.x, aPos.y, newPos.z, 1.0f);\n"
"}\n";

const char* smallFragCode = "uniform vec4 newColor;\n"
"void main(void)\n"
"{ gl_FragColor = newColor; }";

const char* bigVertCode = "void main()\n"
"{ gl_Position = ftransform(); }";

const char* bigFragCode = "void main(void)\n"
"{ gl_FragColor = vec4(1,1,1,1); }";

int flag = 0;

void myDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Big sphere
    glUseProgram(bigSphereProgram);
    glutWireSphere(0.4f, 20, 20);

    // Small sphere
    glUseProgram(smallSphereProgram); // overwrite
    int vertexColorLocation = glGetUniformLocation(smallSphereProgram,"newColor");
    glUniform4f(vertexColorLocation,1.0f,1.0f,0.0f,1.0f);
    int vertexLatCISLocation = glGetUniformLocation(smallSphereProgram,"latCIS");
    glUniform2f(vertexLatCISLocation,sin((float)angRot/180*PI),cos((float)angRot/180*PI));
    int vertexRevCISLocation = glGetUniformLocation(smallSphereProgram,"revCIS");
    glUniform2f(vertexRevCISLocation,sin((float)angRev/180*PI),cos((float)angRev/180*PI));
    int vertexCenterLocation = glGetUniformLocation(smallSphereProgram,"center");
    float centerx = sin((float)angRev/180*PI) * distance;
    float centerz = cos((float)angRev/180*PI) * distance;
    glUniform3f(vertexCenterLocation,centerx,0,centerz);
    // printf("angRev: %d angRot: %d centerx: %f, centerz: %f\n", angRev, angRot, centerx, centerz);
    glutWireSphere(0.1f*(centerz+2*distance)/(3*distance),8,8);

    glFlush();
}

void keyPressed(unsigned char key, int x, int y)
{
    // int mod = glutGetModifiers(); // GLUT_ACTIVE_SHIFT
    printf("Pressed %c!\n", key);
    switch (key){
        case 'd':angRot = (angRot + 10) % 360;break;
        case 'D':angRot = (angRot - 10) % 360;break;
        case 'y':angRev = (angRev + 10) % 360;break;
        case 'Y':angRev = (angRev - 10) % 360;break;
    }
    myDisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(500, 500);

    glutCreateWindow("Planet rotation (shader)");

    glutDisplayFunc(myDisplay);
    glutKeyboardFunc(keyPressed);

    glewInit();

    int success;
    char infoLog[512];

    // make shader for the small sphere
    GLuint smallVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(smallVertexShader, 1, &smallVertCode, NULL);
    glCompileShader(smallVertexShader);

    glGetShaderiv(smallVertexShader,GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(smallVertexShader,512,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Successfully compiled!\n");
    GLuint smallFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(smallFragmentShader, 1, &smallFragCode, NULL);
    glCompileShader(smallFragmentShader);

    smallSphereProgram = glCreateProgram();
    glAttachShader(smallSphereProgram,smallVertexShader);
    glAttachShader(smallSphereProgram,smallFragmentShader);

    glLinkProgram(smallSphereProgram);

    // make shader for the big sphere
    GLuint bigVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(bigVertexShader, 1, &bigVertCode, NULL);
    glCompileShader(bigVertexShader);
    glGetShaderiv(bigVertexShader,GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(bigVertexShader,512,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Successfully compiled!\n");
    GLuint bigFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(bigFragmentShader, 1, &bigFragCode, NULL);
    glCompileShader(bigFragmentShader);

    bigSphereProgram = glCreateProgram();
    glAttachShader(bigSphereProgram,bigVertexShader);
    glAttachShader(bigSphereProgram,bigFragmentShader);

    glLinkProgram(bigSphereProgram);

    // get into display
    glutMainLoop();

    return 0;
}

// gcc shader.c -lopengl32 -lglew32 -lglut32 -o shader.exe