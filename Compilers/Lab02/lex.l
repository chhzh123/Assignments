/*** Definition Section has one variable 
which can be accessed inside yylex()  
and main() ***/
%{ 
    int cnt_int = 0;
    int cnt_float = 0; 
%}

%% 
(\/\*([^*]|[\r\n]|(\*+([^*\/]|[\r\n])))*\*+\/)|(\/\/.*) ; /* firstly ignore the comments */
[a-zA-Z_][a-zA-Z0-9]* ; /* variable names */
[+-]?[0-9]+ {
    printf("Int: %s\n", yytext);
    cnt_int++;
}
[+-]?[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)? {
    printf("Float: %s\n", yytext);
    cnt_float++;
} 
.           ;
[ \t\r\n] ;  /* skip whitespace which is not part of a string */
%% 
  
/*** Code Section prints the number of 
capital letter present in the given input***/
int yywrap(){} 
int main(){ 
  
// Explanation: 
// yywrap() - wraps the above rule section 
/* yyin - takes the file pointer  
          which contains the input*/
/* yylex() - this is the main flex function 
          which runs the Rule Section*/
// yytext is the text in the buffer 

    FILE *fp = fopen("lex_test.cpp","r"); 
    yyin = fp; 
      
    yylex(); 
    printf("\n# of Integers: %d"
           "\n# of Floats: %d\n", cnt_int, cnt_float);
     
    return 0; 
}