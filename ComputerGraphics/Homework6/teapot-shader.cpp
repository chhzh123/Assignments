#include <windows.h> // must be the first one to be included!
#include <stdio.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

GLuint teapotProgram;

const char* vertexShaderCode;
const char* fragShaderCode;
const char* loadShaderFile(const char *filename);

void init(void)
{
    GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
    GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
    GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat light_position[] = {4.5, 4.5, 3  , 1.0}; // fix position by model view matrix
    GLfloat obj_ambient[]    = {0.19225,0.19225,0.19225,1.0};
    GLfloat obj_diffuse[]    = {0.50754,0.50754,0.50754};
    GLfloat obj_specular[]   = {1,1,1};
    GLfloat obj_shininess[]  = {64.0};

    GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    GLfloat local_view[]     = {0.0};

    // initialize lighting model
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    // material properties
    glMaterialfv(GL_FRONT, GL_AMBIENT  , obj_ambient);   // rgb
    glMaterialfv(GL_FRONT, GL_DIFFUSE  , obj_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR , obj_specular);  // reflect white lights
    glMaterialfv(GL_FRONT, GL_SHININESS, obj_shininess); // shine

    glFrontFace(GL_CW);
    glEnable(GL_LIGHTING); // global
    glEnable(GL_LIGHT0); // each lighting
    glEnable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(teapotProgram);

    // glm::mat4 proj = glm::perspective(45.0f, 800.0f / 600.0f, 0.1f, 100.0f);
    GLfloat aspect = 1.0f;
    glm::mat4 proj = glm::ortho(-3.0 * aspect, 3.0 * aspect, -3.0, 3.0, -10.0, 10.0);
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, -4.5f, 2.0f), // Camera is at (4.5,4.5,3), in World Space
        glm::vec3(0.0f, 0.0f, 0.0f), // and looks at the origin
        glm::vec3(0.0f, 0.0f, 1.0f)); // (0,1,0) Head is up
    glm::mat4 model = glm::mat4(1.0f); // translate & rotate

    glm::mat4 mvp = proj * view * model;
    int lightPosLocation = glGetUniformLocation(teapotProgram,"lightPos");
    glUniform3f(lightPosLocation,4.5,4.5,3);
    int viewPosLocation = glGetUniformLocation(teapotProgram,"viewPos");
    glUniform3f(viewPosLocation,4.5,4.5,4.5);
    int lightColorLocation = glGetUniformLocation(teapotProgram,"lightColor");
    glUniform3f(lightColorLocation,1,1,1);
    int projectionLocation = glGetUniformLocation(teapotProgram,"projection");
    glUniformMatrix4fv(projectionLocation,1,GL_FALSE,&proj[0][0]);
    int modelLocation = glGetUniformLocation(teapotProgram,"model");
    glUniformMatrix4fv(modelLocation,1,GL_FALSE,&model[0][0]);
    int viewLocation = glGetUniformLocation(teapotProgram,"view");
    glUniformMatrix4fv(viewLocation,1,GL_FALSE,&view[0][0]);
    int objectColorLocation = glGetUniformLocation(teapotProgram,"objectColor");
    glUniform3f(objectColorLocation,0.50754,0.50754,0.50754);
    glutSolidTeapot(1.0);
    glFlush();
}

/* Handler for window re-size event. Called back when the window first appears and
   whenever the window is re-sized with its new width and height */
void reshape(GLsizei width, GLsizei height) {  // GLsizei for non-negative integer
    // Compute aspect ratio of the new window
    if (height == 0) height = 1;                // To prevent divide by 0
    GLfloat aspect = (GLfloat)width / (GLfloat)height;

    // Set the viewport to cover the new window
    // (x, y) is the left bottom corner
    glViewport(0, 0, width, height); // i.e. the area that can be seen

    // Set the aspect ratio of the clipping volume to match the viewport
    glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
    glLoadIdentity();             // Reset
    /*
     * The camera placed very far away, then become parallel projection
     * an object appears to be the same size regardless of the depth
     * void glOrtho(GLdouble left, GLdouble right,
     *     GLdouble bottom, GLdouble top,
     *     GLdouble nearVal, GLdouble farVal);
     */
    if (width >= height)
        glOrtho(-3.0 * aspect, 3.0 * aspect, -3.0, 3.0, -10.0, 10.0);
    else
        glOrtho(-3.0, 3.0, -3.0 / aspect, 3.0 / aspect, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
}

void keyPressed(unsigned char key, int x, int y)
{
    printf("Pressed %c!\n", key);
    display();
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(600, 600);

    glutCreateWindow("Teapot lighting (shader)");

    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyPressed);

    glewInit();

    int success;
    char infoLog[1024];

    // make shaders
    vertexShaderCode = loadShaderFile("shader.vert");
    fragShaderCode = loadShaderFile("shader.frag");
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader,GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(vertexShader,1024,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Vertex shader successfully compiled!\n");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragShaderCode, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader,GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(fragmentShader,1024,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Fragment shader successfully compiled!\n");

    teapotProgram = glCreateProgram();
    glAttachShader(teapotProgram,vertexShader);
    glAttachShader(teapotProgram,fragmentShader);

    glLinkProgram(teapotProgram);
    glGetShaderiv(teapotProgram,GL_LINK_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(fragmentShader,1024,NULL,infoLog);
        printf("Error: %s\n", infoLog);
    } else
        printf("Successfully linked!\n");

    init();

    // get into display
    glutMainLoop();

    return 0;
}

const char* loadShaderFile(const char *filename)
{
    char* text = NULL;
    
    if (filename != NULL) {
        FILE *file = fopen(filename, "rt");
        
        if (file != NULL) {
            fseek(file, 0, SEEK_END);
            int count = ftell(file);
            rewind(file);
            
            if (count > 0) {
                text = (char*)malloc(sizeof(char) * (count + 1));
                count = fread(text, sizeof(char), count, file);
                text[count] = '\0';
            }
            fclose(file);
        }
    }
    return text;
}

// g++ -Iinclude teapot-shader.cpp -lopengl32 -lglew32 -lglut32 -o teapot-shader.exe