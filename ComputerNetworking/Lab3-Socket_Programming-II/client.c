// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 3: Socket programming - II

/* client.c */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <error.h>
#include <pthread.h>

#define BUF_LEN 2000                        // 缓冲区大小

void* receive(void* arg);

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

    pthread_t pt;
    pthread_create(&pt,NULL,receive,&sock);

    while (1){
        // printf("输入要发送的信息：");
        scanf("%s", buf);
        if (strcmp(buf,"exit") == 0)
            break;
        // 第二个参数指向发送缓冲区，第三个参数为要发送的字节数，第四个参数一般置0
        // 返回值为实际发送的字节数，出错或对方关闭时返回SOCKET_ERROR。
        cc = send(sock, buf, strlen(buf), 0);
        if (cc <= 0){
            perror("Error: Server!\n");
            return 0;
        }
    }

    close(sock);                                         // 关闭连接套接字

    printf("按回车键继续...\n");
    getchar();
    return 0;
}

void* receive(void* arg)
{
    char buf[BUF_LEN+1];
    int* sock = (int*) arg;
    while (1){
        int cc = recv(*sock, buf, BUF_LEN, 0);
        if (cc <= 0){
            perror("Error: Server!\n");
            abort();
            break;
        }
        buf[cc] = '\0';
        printf("%s\n", buf);
    }
    pthread_exit(0);
}