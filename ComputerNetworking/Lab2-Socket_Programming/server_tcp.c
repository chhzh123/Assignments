// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 2: Socket programming

/* server_tcp.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <error.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#define BUF_LEN 2000

int kbhit(void);

// struct in_addr {
//     unsigned long s_addr;    // load with inet_aton()
// };

// struct sockaddr_in {
//     short sin_family;
//     u_short sin_port;        //端口号
//     struct in_addr sin_addr; //IP地址
//     char sin_zero[8];
// };

void generateMsg(char* buffer){
    printf("收到消息：%s\n", buffer);

    time_t now; /* current time           */
    time(&now);
    char* pts = (char *)ctime(&now);
    printf("收到时间：%s\n", pts);

    strcat(pts,buffer);
    strcpy(buffer,pts);
    strcat(buffer,"\n");
}

void generateEnhancedMsg(char* buffer, unsigned char *bytes, u_short port)
{
    char buf[BUF_LEN+1];
    printf("收到信息：%s\n", buffer);
    sprintf(buf, "内容：%s\n", buffer);

    time_t now; /* current time */
    time(&now);
    char* pts = (char *)ctime(&now);
    printf("收到时间：%s", pts);
    sprintf(buffer,"时间：%s", pts);
    strcat(buf, buffer);

    // inet_ntoa
    snprintf (buffer, sizeof (buf), "客户端IP地址：%d.%d.%d.%d\n",
              bytes[0], bytes[1], bytes[2], bytes[3]);
    printf("%s", buffer);
    strcat(buf,buffer);

    sprintf(buffer, "客户端端口号：%d\n", port);
    printf("%s", buffer);
    strcat(buf, buffer);

    printf("\n");
    strcpy(buffer,buf);
}

int main(int argc, char *argv[])
{
    struct  sockaddr_in fsin;               /* the from address of a client   */
    int     msock, ssock;                   /* master & slave sockets         */
    char    *service = "50500";
    char    buf[BUF_LEN+1];                 /* buffer for one line of text    */
    struct  sockaddr_in sin;                /* an Internet endpoint address   */
    int     alen;                           /* from-address length            */
    char    *pts;                           /* pointer to time string         */

    // 创建套接字，参数：因特网协议簇(family)，流套接字，TCP协议
    // 返回：要监听套接字的描述符或INVALID_SOCKET
    msock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    memset(&sin,'\0', sizeof(sin));                         // 从&sin开始的长度为sizeof(sin)的内存清0
    sin.sin_family = AF_INET;                               // 因特网地址簇(INET-Internet)
    sin.sin_addr.s_addr = INADDR_ANY;                       // 监听所有(接口的)IP地址。
    sin.sin_port = htons((u_short)atoi(service));           // 监听的端口号。atoi--把ascii转化为int，htons--主机序到网络序(host to network，s-short 16位)
    bind(msock, (struct sockaddr *)&sin, sizeof(sin));      // 绑定监听的IP地址和端口号

    listen(msock, 5);                                       // 建立长度为5的连接请求队列，并把到来的连接请求加入队列等待处理。

    printf("服务器已启动！\n\n");

    while (!kbhit()){                                        // 检测是否有按键,如果没有则进入循环体执行
        alen = sizeof(struct sockaddr);                      // 取到地址结构的长度
        // 如果在连接请求队列中有连接请求，则接受连接请求并建立连接，返回该连接的套接字
        // 否则，本语句被阻塞直到队列非空。fsin包含客户端IP地址和端口号
        ssock = accept(msock, (struct sockaddr *)&fsin, &alen);

        // 第二个参数指向缓冲区，第三个参数为缓冲区大小(字节数)，第四个参数一般设置为0
        // 返回值:(>0)接收到的字节数,(=0)对方已关闭,(<0)连接出错
        int cc = recv(ssock, buf, BUF_LEN, 0);
        if (cc <= 0)
            printf("Error!\n");                    // 出错或对方关闭(==0)。其后必须关闭套接字sock。
        else if (cc > 0) {
            buf[cc] = '\0';

            if (argc == 1){ // TCP Echo Enhancement
                generateMsg(buf);
            } else {
                generateEnhancedMsg(buf, (unsigned char *) &(fsin.sin_addr), fsin.sin_port);
            }

            cc = send(ssock, buf, strlen(buf), 0);
            if (cc <= 0)
                printf("Server send message error!\n");
        }

        close(ssock);
    }
    close(msock);
    return 0;
}

// https://cboard.cprogramming.com/c-programming/63166-kbhit-linux.html
int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;
    
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    
    ch = getchar();
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    
    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    
    return 0;
}