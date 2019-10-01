//OpenGL要包含gl.h和glut.h，但glut文件中已经包含这两个文件了
#include <windows.h>
#include <GL/glut.h>

void myDisplay()
{
    //清除，GL_COLOR_BUFFER_BIT表示清除颜色，glClear函数还可以清除其它的东西。
    glClear(GL_COLOR_BUFFER_BIT);

    //画一个矩形，四个参数分别表示了位于对角线上的两个点的横、纵坐标。
    glRectf(-0.5f, -0.5f, 0.5f, 0.5f);

    //保证前面的OpenGL命令立即执行（而不是让它们在缓冲区中等待），其作用跟fflush(stdout)类似。
    glFlush();
}

int main(int argc, char *argv[])
{
    //对glut初始化，这个函数必须在其它glut使用之前调用一次。
    //其格式比较死板，一般用glutInit(&argc, argv)就可以了。
    glutInit(&argc, argv);

    //设置显示方式，其中GLUT_RGB表示使用RGB颜色，与之对应的还有GLUT_INDEX（表示使用索引颜色）。
    //GLUT_SINGLE表示使用单缓冲，与之对应的还有GLUT_DOUBLE（使用双缓冲）。
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutInitWindowPosition(100, 100);//设置窗口在屏幕中的位置
    glutInitWindowSize(400, 400);//设置窗口的大小

    //根据前面设置的信息创建窗口，参数将被作为窗口的标题。
    //注意：窗口被创建后，并不立即显示到屏幕上。需要调用glutMainLoop才能看到窗口。
    glutCreateWindow("第一个OpenGL程序");

    //设置一个函数，当需要进行画图时，这个函数就会被调用。
    //这个说法不够准确，暂时这样说吧。
    glutDisplayFunc(myDisplay);

    //进行一个消息循环。
    //只需要知道这个函数可以显示窗口，并且等待窗口关闭后才会返回，这就足够了。
    glutMainLoop();

    return 0;
}