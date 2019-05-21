// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 5: File Transmission

/* client.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define BUF_LEN 100000

int main(int argc, char *argv[])
{
    char    *host = "127.0.0.1";       /* server IP to connect         */
    char    *service = "50500";        /* server port to connect       */
    struct  sockaddr_in sin;           /* an Internet endpoint address */
    char    buf[100];                  /* buffer for file name         */
    char    res[BUF_LEN];              /* buffer for file context      */
    int     sock;                      /* socket descriptor            */
    int     cc;                        /* recv character count         */

    printf("正在连接...\n");

    // create socket
    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = inet_addr(host);
    sin.sin_port = htons((u_short)atoi(service));
    int ret = connect(sock, (struct sockaddr *)&sin, sizeof(sin));
    printf("连接成功！\n\n");

    while (1){
        printf("输入文件名：");
        scanf("%s", buf);
        if (strcmp(buf,"exit") == 0)
            break;
        // split string
        char path[100], *file_name;
        strcpy(path,buf);
        char* str = buf;
        char* p = strsep(&str,"/"); // must in Linux!
        while (p != NULL)
        {
            file_name = p;
            p = strsep(&str,"/");
        }

        cc = send(sock, file_name, strlen(file_name), 0);
        assert(cc > 0);

        printf("正在传送...\n");
        FILE* fin = fopen(path,"rb");
        assert(fin != NULL);
        // get file size
        fseek(fin,0,SEEK_END);
        long size = ftell(fin);
        rewind(fin);

        fread(res,size,1,fin);
        cc = send(sock, res, strlen(res), 0);

        fclose(fin);
        printf("传送结束！\n\n");
    }

    close(sock);
    printf("\n程序结束！\n按回车键继续...\n");
    getchar();
    return 0;
}