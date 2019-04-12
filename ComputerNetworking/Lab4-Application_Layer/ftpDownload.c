// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 4: FTP Client

/* ftpDownload.c */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <error.h>
#include <pthread.h>

#define BUF_LEN 100000

void* receive(void* arg);

int main(int argc, char *argv[])
{
    /* check command line arguments */
    if (argc != 4) {
       fprintf(stderr,"usage: %s <hostname> <filename> <dstname>\n", argv[0]);
       exit(0);
    }
    struct hostent *server;
    char* hostname = argv[1];
    int port = 21; // ftp
    printf("Host: %s %d\n", hostname, port);
    char* filename = argv[2];
    char* dstname = argv[3];
    FILE *fp = fopen(dstname,"w");

    /* gethostbyname: get the server's DNS entry */
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr,"Error: no such host as %s\n", hostname);
        exit(0);
    }

    struct  sockaddr_in sin;           /* an Internet endpoint address */
    char    buf[BUF_LEN+1];            /* buffer for one line of text  */
    char    res[BUF_LEN+1];
    int     sock;                      /* socket descriptor            */
    int     cc;                        /* recv character count         */

    // create socket
    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) 
        perror("Error: opening socket\n");

    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    // sin.sin_addr.s_addr = inet_addr(host);
    bcopy((char *)server->h_addr,(char *)&sin.sin_addr.s_addr, server->h_length);
    sin.sin_port = htons((u_short)port);
    printf("Connecting to server...\n");
    int ret = connect(sock, (struct sockaddr *)&sin, sizeof(sin));
    if (ret == 0)
        printf("Connected!\n\n");
    else {
        perror("Error: Fail!\n");
        abort();
    }

    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("%s", buf);

    memset(res, 0, sizeof(sin));
    strcpy(buf,"user abc\r\n");
    cc = send(sock,buf,strlen(buf),0);
    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("user abc\r\n");
    printf("%s", buf);

    strcpy(buf,"pass 123666\r\n");
    cc = send(sock,buf,strlen(buf),0);
    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("pass 123666\r\n");
    printf("%s", buf);

    strcpy(buf,"pasv\r\n");
    cc = send(sock,buf,strlen(buf),0);
    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("pasv\r\n");
    printf("%s", buf);

    int addr_ftp[6];
    sscanf(buf, "%*[^(](%d,%d,%d,%d,%d,%d)",&addr_ftp[0],&addr_ftp[1],&addr_ftp[2],&addr_ftp[3],&addr_ftp[4],&addr_ftp[5]);

    struct  sockaddr_in r_sin;           /* an Internet endpoint address */
    int     r_sock;                      /* socket descriptor            */
    memset(&r_sin, 0, sizeof(r_sin));
    r_sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    r_sin.sin_family = AF_INET;
    // sin.sin_addr.s_addr = inet_addr(host);
    bcopy((char *)server->h_addr,(char *)&r_sin.sin_addr.s_addr, server->h_length);
    r_sin.sin_port = htons((u_short)(addr_ftp[4]*256 + addr_ftp[5]));
    printf("Connecting to data link %d.%d.%d.%d %d...\n",addr_ftp[0],addr_ftp[1],addr_ftp[2],addr_ftp[3],r_sin.sin_port);
    ret = connect(r_sock, (struct sockaddr *)&r_sin, sizeof(r_sin));
    if (ret == 0)
        printf("Connected!\n\n");
    else {
        perror("Error: Fail!\n");
        abort();
    }

    strcpy(buf,"retr ");
    strcat(buf,filename);
    strcat(buf,"\r\n");
    cc = send(sock,buf,strlen(buf),0);
    printf("%s",buf);
    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("%s", buf);

    pthread_t pt;
    pthread_create(&pt,NULL,receive,&sock);

    printf("Begin downloading...\n");
    while (1){
        int cc = recv(r_sock, buf, BUF_LEN, 0);
        if (cc <= 0){
            // perror("Error: Server!\n");
            break;
        }
        buf[cc] = '\0';
        fprintf(fp, "%s", buf);
    }
    fclose(fp);
    printf("Finish downloading.\n");

    strcpy(buf,"quit\r\n");
    cc = send(sock,buf,strlen(buf),0);
    cc = recv(sock,buf,BUF_LEN, 0);
    buf[cc] = '\0';
    printf("quit\r\n");
    printf("%s", buf);

    close(sock);

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
            // perror("Error: Server!\n");
            break;
        }
        buf[cc] = '\0';
        printf("%s", buf);
    }
    pthread_exit(0);
}