#include <stdio.h>


int main(){	
	char str[1000], ch;
	int freq, n;
	
	printf("Enter a string: ");
	gets_s(str,1001);
	
	printf("Enter a symbol: ");
	n = scanf("%c", &ch);
	if (n!=1)
	printf("Error");
	
	while(*str){
		if(ch == *str)
		++freq;
		}
	
	printf("Total freq: %d\n", freq);
	
	return 0;
}

