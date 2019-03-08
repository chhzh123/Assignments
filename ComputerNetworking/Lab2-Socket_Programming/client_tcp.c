// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 2: Socket programming

/* client_tcp.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <error.h>

#define BUF_LEN 2000                        // 缓冲区大小

int main(int argc, char *argv[])
{
    char    *host = "127.0.0.1";       /* server IP to connect         */
    char    *service = "50500";        /* server port to connect       */
    struct  sockaddr_in sin;           /* an Internet endpoint address */
    char    buf[BUF_LEN+1];            /* buffer for one line of text  */
    int     sock;                      /* socket descriptor            */
    int     cc;                        /* recv character count         */

    // 创建套接字，参数：因特网协议簇(family)，流套接字，TCP协议
    // 返回：要监听套接字的描述符或INVALID_SOCKET
    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    memset(&sin, 0, sizeof(sin));                     // 从&sin开始的长度为sizeof(sin)的内存清0
    sin.sin_family = AF_INET;                         // 因特网地址簇
    sin.sin_addr.s_addr = inet_addr(host);            // 设置服务器IP地址(32位)
    sin.sin_port = htons((u_short)atoi(service));     // 设置服务器端口号
    // 连接到服务器，第二个参数指向存放服务器地址的结构，第三个参数为该结构的大小，返回值为0时表示无错误发生，
    // 返回SOCKET_ERROR表示出错，应用程序可通过WSAGetLastError()获取相应错误代码。
    int ret = connect(sock, (struct sockaddr *)&sin, sizeof(sin));

    printf("输入要发送的信息：");
    scanf("%s", buf);
    // 第二个参数指向发送缓冲区，第三个参数为要发送的字节数，第四个参数一般置0
    // 返回值为实际发送的字节数，出错或对方关闭时返回SOCKET_ERROR。
    cc = send(sock, buf, strlen(buf), 0);
    if (cc <= 0){
        printf("Error!\n");
        return 0;
    }

    printf("\n收到的信息：\n");
    if (cc <= 0)
        printf("Error!\n");                    // 出错或对方关闭(==0)。其后必须关闭套接字sock。
    else if (cc > 0) {
        cc = recv(sock, buf, BUF_LEN, 0);
        buf[cc] = '\0';
        printf("%s\n", buf);
    }

    close(sock);                                         // 关闭连接套接字

    printf("按回车键继续...\n");
    getchar();
    return 0;
}