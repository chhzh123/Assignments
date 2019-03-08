// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 2: Socket programming

/* client_udp.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <error.h>

#define BUFLEN 2000                  // 缓冲区大小

int main(int argc, char *argv[])
{
    char   *host = "127.0.0.1";      /* server IP to connect         */
    char   *service = "50500";       /* server port to connect       */
    struct sockaddr_in toAddr;       /* an Internet endpoint address */
    int    tosize = sizeof(toAddr);
    char   buf[BUFLEN+1];            /* buffer for one line of text  */
    int    sock;                     /* socket descriptor            */
    int    cc;                       /* recv character count         */
    char   *pts;                     /* pointer to time string       */
    time_t now;                      /* current time                 */

    sock = socket(PF_INET, SOCK_DGRAM,IPPROTO_UDP);

    memset(&toAddr, 0, sizeof(toAddr));
    toAddr.sin_family = AF_INET;
    toAddr.sin_port = htons((u_short)atoi(service));    // htons：主机序(host)转化为网络序(network), s--short
    toAddr.sin_addr.s_addr = inet_addr(host);           // 如果host为域名，需要先用函数gethostbyname把域名转化为IP地址

    printf("输入消息：");
    scanf("%s", buf);
    printf("\n");

    cc = sendto(sock, buf, BUFLEN, 0, (struct sockaddr *)&toAddr, sizeof(toAddr));
    if (cc <= 0) {
        printf("发送失败，错误号：%d\n", cc);
    } else {
        cc = recvfrom(sock, buf, BUFLEN, 0, (struct sockaddr *)&toAddr, &tosize);
        if (cc < 0){
            printf("recv() failed; %d\n", cc);
        } else {
            buf[cc] = '\0';
            printf("%s", buf);
        }
    }

    close(sock);

    printf("按回车键继续...\n");
    getchar();
}