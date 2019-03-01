// Chen Hongzheng 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 1: Data Representation - Structure store & copy

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUF_LEN 100
#define USER_NAME_LEN 20
#define EMAIL_LEN 80
#define TIME_BUF_LEN 30
#define MAX_PEOPLE 100

typedef unsigned long DWORD;

typedef struct Person {
    char username[USER_NAME_LEN];
    int level;
    char email[EMAIL_LEN];
    DWORD sendtime;
    time_t regtime;
} Person;

int main()
{
	Person people[MAX_PEOPLE];
	FILE* pfile;
	int i;
	// Input
	pfile = fopen("./Persons.txt","wb");
	printf("----- USER INPUT -----\n");
	for (i = 0; i < MAX_PEOPLE; ++i){
		fflush(stdin);
		char name[USER_NAME_LEN];
		printf("username: ");
		gets(name);
		if (name[0] == '\0')
			break;
		strcpy(people[i].username,name);
		fprintf(pfile, "%s\n", name);

		int l;
		printf("level: ");
		scanf("%d",&l);
		people[i].level = l;
		fprintf(pfile, "%d\n", l);

		char email[EMAIL_LEN];
		printf("email: ");
		scanf("%s",&email);
		strcpy(people[i].email,email);
		fprintf(pfile, "%s\n", email);

		time_t now; // current time
		time (&now);  // get now time
		struct tm* lt = localtime (&now);
		people[i].sendtime = (DWORD)now;
		people[i].regtime = now;
		fprintf(pfile, "%ld\n", people[i].sendtime);
		fprintf(pfile, "%ld\n", people[i].regtime);

		printf("\n");
	}
	fclose(pfile);
	printf("----- USER INPUT OVER -----\n\n");

	// Read the file
	pfile = fopen("Persons.txt","r");
	if (pfile == NULL)
		exit(EXIT_FAILURE);
	printf("----- PRINT RESULTS -----\n");
	for (i = 0; i < MAX_PEOPLE; ++i){
		char name[USER_NAME_LEN];
		if (fscanf(pfile,"%s",name) != 1)
			break;
		printf("username: %s\n", name);

		int l;
		fscanf(pfile,"%d",&l);
		printf("level: %d\n", l);

		char email[EMAIL_LEN];
		fscanf(pfile,"%s",email);
		printf("email: %s\n",email);

		char buf[TIME_BUF_LEN];
		time_t sendtime;
		fscanf(pfile,"%ld",&sendtime);
		struct tm* lt = localtime (&sendtime);
		// Www Mmm dd hh:mm:ss yyyy\n
		strftime(buf,TIME_BUF_LEN,"%a %b %d %H:%m:%S %Y",lt);
		printf("sendtime: %s\n", buf);
		
		time_t regtime;
		fscanf(pfile,"%ld",&regtime);
		lt = localtime (&regtime);
		strftime(buf,TIME_BUF_LEN,"%a %b %d %H:%m:%S %Y",lt);
		printf("regtime: %s\n", buf);

		printf("\n");
	}
	printf("----- PRINT RESULTS OVER ----\n");
	fclose(pfile);
	return 0;
}