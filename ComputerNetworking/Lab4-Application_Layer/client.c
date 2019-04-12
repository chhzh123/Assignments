// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 4: TCP Client

/* client.c */

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
    if (argc != 3) {
       fprintf(stderr,"usage: %s <hostname> <port>\n", argv[0]);
       exit(0);
    }
    struct hostent *server;
    char* hostname = argv[1];
    int port = atoi(argv[2]);
    printf("Host: %s %d\n", hostname, port);

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
        printf("Connected！\n\n");
    else {
        perror("Error: Fail！\n");
        abort();
    }

    pthread_t pt;
    pthread_create(&pt,NULL,receive,&sock);

    memset(res, 0, sizeof(sin));
    while (1){
        // fgets(buf,BUF_LEN,stdin);
        gets(buf);
        // if (strlen(buf) == 0 || strcmp(buf,"\n") == 0 || strcmp(buf,"\r") == 0){
        //     strcat(buf,"\r\n");
        //     cc = send(sock,buf,strlen(buf),0);
        // } else if (strcmp(buf,"exit") == 0)
        //     break;
        strcat(buf,"\r\n");
        cc = send(sock,buf,strlen(buf),0);
    }

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
            perror("Error: Server!\n");
            abort();
            break;
        }
        buf[cc] = '\0';
        printf("%s\n", buf);
    }
    pthread_exit(0);
    abort();
}