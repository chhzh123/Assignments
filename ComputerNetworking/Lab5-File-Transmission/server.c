// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 5: File Transmission

/* server.c */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BUF_LEN 100000

int main(int argc, char *argv[])
{
    struct  sockaddr_in fsin;               /* the from address of a client   */
    int     msock, ssock;                   /* master & slave sockets         */
    char    *service = "50500";
    char    buf[100];                       /* buffer for file name           */
    char    res[BUF_LEN];                   /* buffer for file context        */
    struct  sockaddr_in sin;                /* an Internet endpoint address   */
    int     alen;                           /* from-address length            */
    char    *pts;                           /* pointer to time string         */

    char path[100];
    printf("输入接收文件夹：");
    scanf("%s",path);

    printf("\n等待连接...\n");
    // create socket
    msock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    memset(&sin,'\0', sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons((u_short)atoi(service));
    bind(msock, (struct sockaddr *)&sin, sizeof(sin));

    listen(msock, 5); // length-5 request queue
    alen = sizeof(struct sockaddr);
    ssock = accept(msock, (struct sockaddr *)&fsin, &alen);
    printf("连接成功！\n\n");

    while (1){
        memset(buf,sizeof(buf),0);
        memset(res,sizeof(res),0);
        char tmppath[100];
        strcpy(tmppath,path);

        // receive file name
        int cc = recv(ssock, buf, BUF_LEN, 0);
        if (cc <= 0) // client closed
            break;
        else if (cc > 0) {
            buf[cc] = '\0';

            printf("正接收文件%s...\n", buf);
            int cc = recv(ssock, res, BUF_LEN, 0);
            
            // test if file exists
            int cnt = 1;
            strcat(tmppath,"/"); // must in Linux!
            strcat(tmppath,buf);
            char file_name[100];
            while (1){
                strcpy(file_name,tmppath);
                if (cnt != 1) {
                    char* p = strchr(file_name,'.');
                    if (p != 0) { // not found
                        char suffix[10];
                        strcpy(suffix,p);
                        *p = '\0';
                        strcat(file_name,"(");
                        char numstr[10];
                        sprintf(numstr,"%d",cnt);
                        strcat(file_name,numstr);
                        strcat(file_name,")");
                        strcat(file_name,suffix);
                    } else {
                        strcat(file_name,"(");
                        char numstr[10];
                        sprintf(numstr,"%d",cnt);
                        strcat(file_name,numstr);
                        strcat(file_name,")");
                    }
                }
                if (access(file_name, F_OK ) == -1)
                    break;
                cnt++;
            }
            FILE* fout = fopen(file_name,"wb");
            fwrite(res,strlen(res),1,fout);
            fclose(fout);
            printf("接收完毕！\n\n");
        }
    }
    close(ssock);
    close(msock);
    printf("程序结束！\n按任意键继续...\n");
    getchar();
    return 0;
}