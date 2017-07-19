#include <stdio.h>
#include <math.h>
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image.h"
//#include "stb_image_write.h"

#define SEPERATION 0.4
#define HALFOFSEP  0.2

float* smooth(int *input, unsigned int inpX, unsigned int inpY) {
    return 0;
}

void bilinearFilter(float *input, unsigned int inpX, unsigned int inpY, float *out, unsigned int outX, unsigned int outY) {
    unsigned int size = outX * outY;
    unsigned int upX, upY, loX, loY;
    unsigned int xVal, yVal;
    float relX = ((float)inpX - 1 - SEPERATION) / ((float)outX - 1);
    float relY = ((float)inpY - 1 - SEPERATION) / ((float)outY - 1);
    float x, y;
    float r1, r2;
    for (unsigned int i = 0; i < size; i++) {
        xVal = i % (outX);
        yVal = i / (outX);
        //printf("xVal %u yVal %u", xVal, yVal);
        x = relX *  xVal + HALFOFSEP;
        y = relY *  yVal + HALFOFSEP;
        //printf("x %f und y %f\n", x, y);
        upX = (unsigned int)floor(x);
        upY = (unsigned int)floor(y);
        loX = upX + 1;
        loY = upY + 1;
        //printf("upX %u, upY %u, loX %u, loY %u\n", upX, upY, loX, loY);
        //if(loX == inpX || loY == inpY) printf("bingo");
        //printf("up-x %f und ceil(x) -x = %f und input %f \n", (upX-x), ceil(x) -x, input[loX + loY * inpX]);
        r1 = (input[loX + loY * inpX] * (upX - x) + input[upX + loY * inpX] * (x - loX));
        r2 = (input[loX + upY * inpX] * (upX - x) + input[upX + upY * inpX] * (x - loX));

        out[i] = (r1 * (upY - y) + r2 * (y - loY));

        //printf("r1: %f und r2: %f und out: %f\n", r1, r2, out[i]);
    }

}
