#include <stdio.h>
#include <stdlib.h>
int main() {
    int *p = malloc(10);  // Allocating 10 bytes instead of 10 integers
    if(p == NULL) return 1;
    free(p);
    return 0;
}
