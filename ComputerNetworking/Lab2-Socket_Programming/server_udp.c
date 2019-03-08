// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 2: Socket programming

/* server_udp.c */

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
 
#define BUFLEN   2000                  // 缓冲区大小

int kbhit(void);

void generateEnhancedMsg(char* buffer, unsigned char *bytes, u_short port)
{
    char buf[BUFLEN+1];
    sprintf(buf, "客户端的消息：%s\n", buffer);
    printf("%s", buf);

    // inet_ntoa
    snprintf (buffer, sizeof (buf), "客户端IP地址：%d.%d.%d.%d\n",
              bytes[0], bytes[1], bytes[2], bytes[3]);
    printf("%s", buffer);
    strcat(buf, buffer);

    sprintf(buffer, "客户端端口号：%d\n", port);
    printf("%s", buffer);
    strcat(buf, buffer);

    time_t now; /* current time */
    time(&now);
    char* pts = (char *)ctime(&now);
    printf("时间：%s\n", pts);
    sprintf(buffer,"时间：%s", pts);
    strcat(buf, buffer);

    strcat(buf,"\n");
    strcpy(buffer, buf);
}

int main(int argc, char *argv[])
{
    char   *host = "127.0.0.1";        /* server IP Address to connect */
    char   *service = "50500";         /* server port to connect       */
    struct sockaddr_in sin;            /* an Internet endpoint address */
    struct sockaddr_in from;           /* sender address               */
    int    fromsize = sizeof(from);
    char   buf[BUFLEN+1];              /* buffer for one line of text  */
    int    sock;                       /* socket descriptor            */
    int    cc;                         /* recv character count         */

    sock = socket(PF_INET, SOCK_DGRAM,IPPROTO_UDP); // 创建UDP套接字, 参数：因特网协议簇(family)，数据报套接字，UDP协议号， 返回：要监听套接字的描述符或INVALID_SOCKET
    printf("服务器启动！\n\n");

    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;                     // 绑定(监听)所有的接口
    sin.sin_port = htons((u_short)atoi(service));         // 绑定指定接口。atoi--把ascii转化为int，htons -- 主机序(host)转化为网络序(network), 为short类型。 
    bind(sock, (struct sockaddr *)&sin, sizeof(sin));     // 绑定本地端口号（和本地IP地址)

    while(!kbhit()){                                      // 检测是否有按键
        // 接收客户数据。返回结果：cc为接收的字符数，from中将包含客户IP地址和端口号。
        cc = recvfrom(sock, buf, BUFLEN, 0, (struct sockaddr *)&from, &fromsize);
        if (cc < 0){
            printf("recv() failed; %d\n", cc);
            break;
        } else {
            buf[cc] = '\0';
            generateEnhancedMsg(buf, (unsigned char *) &(from.sin_addr), from.sin_port);
            cc = sendto(sock, buf, strlen(buf), 0, (struct sockaddr *)&from, fromsize);
        }
    }
    close(sock);

    printf("按回车键继续...\n");
    getchar();
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