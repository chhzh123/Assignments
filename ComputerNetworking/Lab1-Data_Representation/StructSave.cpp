// 陈鸿峥 17341015
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
#define MAX_INT 0x3f3f3f3f

typedef unsigned long DWORD;

typedef struct Person {
    char username[USER_NAME_LEN];
    int level;
    char email[EMAIL_LEN];
    DWORD sendtime;
    time_t regtime;
} Person;

int inputOnePerson(FILE* pfile)
{
	Person person;
	
	fflush(stdin);
	char name[USER_NAME_LEN];
	printf("username: ");
	gets(name);
	if (strcmp(name,"exit") == 0)
		return 0;
	strcpy(person.username,name);
	fprintf(pfile, "%s\n", name);

	int l;
	printf("level: ");
	scanf("%d",&l);
	person.level = l;
	fprintf(pfile, "%d\n", l);

	char email[EMAIL_LEN];
	printf("email: ");
	scanf("%s",&email);
	strcpy(person.email,email);
	fprintf(pfile, "%s\n", email);

	time_t now; // current time
	time (&now);  // get now time
	struct tm* lt = localtime (&now);
	person.sendtime = (DWORD)now;
	person.regtime = now;
	fprintf(pfile, "%ld\n", person.sendtime);
	fprintf(pfile, "%ld\n", person.regtime);

	printf("\n");
	return 1;
}

int main()
{
	FILE* pfile;
	int i;
	// Input
	pfile = fopen("./Persons.stru","wb");
	printf("----- USER INPUT -----\n");
	for (i = 0; i < MAX_INT; ++i)
		if (!inputOnePerson(pfile))
			break;
	fclose(pfile);
	printf("----- USER INPUT OVER -----\n\n");

	// Read the file
	pfile = fopen("Persons.stru","r");
	if (pfile == NULL)
		exit(EXIT_FAILURE);
	printf("----- PRINT RESULTS -----\n");
	for (i = 0; i < MAX_INT; ++i){
		char name[USER_NAME_LEN];
		if (fscanf(pfile,"%s",name) != 1)
			break;
		printf("姓名: %s ", name);

		int l;
		fscanf(pfile,"%d",&l);
		printf("级别: %d ", l);

		char email[EMAIL_LEN];
		fscanf(pfile,"%s",email);
		printf("电子邮件: %s\n",email);

		char buf[TIME_BUF_LEN];
		time_t sendtime;
		fscanf(pfile,"%ld",&sendtime);
		struct tm* lt = localtime (&sendtime);
		// Www Mmm dd hh:mm:ss yyyy\n
		strftime(buf,TIME_BUF_LEN,"%a %b %d %H:%m:%S %Y",lt);
		printf("发送时间: %s\n", buf);
		
		time_t regtime;
		fscanf(pfile,"%ld",&regtime);
		lt = localtime (&regtime);
		strftime(buf,TIME_BUF_LEN,"%a %b %d %H:%m:%S %Y",lt);
		printf("注册时间: %s\n", buf);

		printf("\n");
	}
	printf("----- PRINT RESULTS OVER ----\n");
	fclose(pfile);
	return 0;
}