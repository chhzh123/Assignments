// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 3: Socket programming - II

/* server.c */

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <error.h>
#include <time.h>
#include <pthread.h> // multi-thread

#define BUF_LEN 2000
#define MAX_CLIENT_NUM 10

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
    char buf[BUF_LEN+1];
    printf("消息：%s\n", buffer);
    sprintf(buf,"消息：%s\n",buffer);
    strcpy(buffer,buf);
}

void generateMsg2(char* buffer, unsigned char *bytes, u_short port)
{
    char buf[BUF_LEN+1];
    // inet_ntoa
    snprintf(buf, sizeof(buf), "客户端IP地址：%d.%d.%d.%d\n",
              bytes[0], bytes[1], bytes[2], bytes[3]);
    printf("%s", buf);

    char buf2[BUF_LEN+1];
    sprintf(buf2, "客户端端口号：%d\n", port);
    printf("%s", buf2);
    strcat(buf, buf2);

    time_t now; /* current time */
    time(&now);
    char* pts = (char *)ctime(&now);
    printf("服务器时间：%s", pts);
    sprintf(buf2,"服务器时间：%s", pts);
    strcat(buf, buf2);

    printf("收到信息：%s\n", buffer);
    sprintf(buf2, "内容：%s\n", buffer);
    strcat(buf, buf2);

    printf("\n");
    strcpy(buffer,buf);
}

void generateEnterMsg(char* buffer, int index, unsigned char *bytes, u_short port)
{
    char buf[BUF_LEN+1];
    sprintf(buf,"%d号客户端进入\n",index);
    printf("%s", buf);

    // inet_ntoa
    char buf2[BUF_LEN+1];
    snprintf(buf2, sizeof(buf2), "客户端IP地址：%d.%d.%d.%d\n",
              bytes[0], bytes[1], bytes[2], bytes[3]);
    printf("%s", buf2);
    strcat(buf,buf2);

    sprintf(buf2, "客户端端口号：%d\n", port);
    printf("%s", buf2);
    strcat(buf, buf2);

    strcpy(buffer,buf);
}

void sendToAll(int* ssock, char* buf){
    for (int i = 0; i < MAX_CLIENT_NUM; ++i)
        if (ssock[i] != -1){
            int cc = send(ssock[i], buf, strlen(buf), 0);
        }
}

int findEmptySSock(int* ssock)
{
    for (int i = 0; i < MAX_CLIENT_NUM; ++i)
        if (ssock[i] == -1)
            return i;
    perror("No ssocks available!");
    abort();
    return -1;
}

typedef struct shared_data
{
    int index;
    int* ssock; // used for communication with each other
    int mode;
    struct in_addr client_addr;
    int port;
} shared_data;

void* serve(void* arg);

int main(int argc, char *argv[])
{
    struct  sockaddr_in fsin;               /* the from address of a client   */
    int     msock;                          /* master socket                  */
    int     ssock[MAX_CLIENT_NUM];          /* slaver sockets                 */
    char    *service = "50500";             /* port number                    */
    struct  sockaddr_in sin;                /* an Internet endpoint address   */
    int     addlen;                         /* from-address length            */

    // 创建套接字，参数：因特网协议簇(family)，流套接字，TCP协议
    // 返回：要监听套接字的描述符或INVALID_SOCKET
    msock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);      // master sock

    memset(&sin,'\0', sizeof(sin));                         // 从&sin开始的长度为sizeof(sin)的内存清0
    sin.sin_family = AF_INET;                               // 因特网地址簇(INET-Internet)
    sin.sin_addr.s_addr = INADDR_ANY;                       // 监听所有(接口的)IP地址。
    sin.sin_port = htons((u_short)atoi(service));           // 监听的端口号。atoi--把ascii转化为int，htons--主机序到网络序(host to network，s-short 16位)
    bind(msock, (struct sockaddr *)&sin, sizeof(sin));      // 绑定监听的IP地址和端口号

    listen(msock, 5);                                       // 建立长度为5的连接请求队列，并把到来的连接请求加入队列等待处理。

    for (int i = 0; i < MAX_CLIENT_NUM; ++i)
        ssock[i] = -1;

    printf("服务器已启动！\n\n");

    pthread_t pthid[MAX_CLIENT_NUM];

    shared_data sdata;
    while(1) {                                               // 检测是否有按键,如果没有则进入循环体执行
        addlen = sizeof(struct sockaddr);                    // 取到地址结构的长度
        // 如果在连接请求队列中有连接请求，则接受连接请求并建立连接，返回该连接的套接字
        // 否则，本语句被阻塞直到队列非空。fsin包含客户端IP地址和端口号
        int index = findEmptySSock(ssock);
        ssock[index] = accept(msock, (struct sockaddr *)&fsin, &addlen);// slaver sock

        sdata.index = index;
        sdata.ssock = ssock;
        if (argc > 1)
            sdata.mode = atoi(argv[1]);
        else
            sdata.mode = 1;
        sdata.client_addr = fsin.sin_addr;
        sdata.port = fsin.sin_port;
        // pthread_create(&thrd1, NULL, (void *)&thread_function, (void *) &some_argument);
        pthread_create(&pthid[index],NULL,serve,(void*)& sdata);
    }
    for (int i = 0; i < MAX_CLIENT_NUM; ++i)
        pthread_join(pthid[i],NULL);
    close(msock);
    return 0;
}

void* serve(void* arg)
{
    char buf[BUF_LEN+1];
    shared_data* sdata = (shared_data*) arg;
    int index = sdata->index;
    int* ssock = sdata->ssock;
    struct in_addr client_addr = sdata->client_addr;
    int port = sdata->port;
    if (sdata->mode >= 3){
        generateEnterMsg(buf, index, (unsigned char*) &client_addr, port);
        sendToAll(ssock,buf);
    }
    while (1){
        // 第二个参数指向缓冲区，第三个参数为缓冲区大小(字节数)，第四个参数一般设置为0
        // 返回值:(>0)接收到的字节数,(=0)对方已关闭,(<0)连接出错
        int cc = recv(ssock[index], buf, BUF_LEN, 0);
        if (cc <= 0){ // error or closed
            if (sdata->mode >= 4){
                sprintf(buf,"客户端%d离开!\n",index);
                printf("%s", buf);
                sendToAll(ssock,buf);
            }
            ssock[index] = -1; // set null
            break;
        }
        else if (cc > 0) {
            buf[cc] = '\0';

            switch (sdata->mode){
                case 1: generateMsg(buf);break;
                case 2:
                case 3:
                case 4:
                default: generateMsg2(buf, (unsigned char *) &client_addr, port);break;
            }
            sendToAll(ssock,buf);
        }
    }
    close(ssock[index]);
    pthread_exit(0);
}