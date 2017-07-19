#include <iostream>
#include <string.h>
#include <stdint.h>
//#include <dir.h>
#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
//#include "stb_image_write.h"
//#include "stb_image_resize.h"
#include "BLF.cpp"

using namespace std;

const char* CATDOG[] = { "cat", "dog" };
const int labels = 100;
const int testLabels[] = {
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
    1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
    0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0, 1, 1, 0, 0
};

// maintain indices to compute next file names
int currentPet = 0;
int numberOfTrainings = 0;
int numberOfTests = 0;

// write downscaled image to out folder for debug purposes
// bool write_png(char* file_name, unsigned char* data, int width, int channels) {
//     char folder[20] = "out\\";
//     strcat(folder, file_name);
//     printf("Writing %s\n", folder);

//     if(!stbi_write_png(folder, width, width, channels, data, 0)) {
//         printf("Could not write %s\n", folder);
//         return false;
//     }

//     return true;
// }

// read jpg from file name into byte array data
bool read_jpg(char* file_name, float*** data, int width, int channels) {
    //printf("Reading %s\n", file_name);

    int origWidth, origHeight, bpp;
    float* image = stbi_loadf(file_name, &origWidth, &origHeight, &bpp, channels);
    if (image == NULL) {
        printf("Could not read %s\n", file_name);
        currentPet = 0;
        numberOfTrainings = 0;
        return false;
    }

    float* channel = (float*)malloc(origWidth * origHeight * sizeof(float));
    float* resized = (float*)malloc(width * width * sizeof(float));
    for (int c = 0; c < channels; c++) {

        // separate colour channel
        for (int i = 0; i < origHeight * origWidth; i++) {
            channel[i] = image[i * channels + c];
        }

        // scale image down to width * width
        bilinearFilter(channel, origWidth, origHeight, resized, width, width);

        // copy values to 3D array data
        for (int h = 0; h < width; h++) {
            for (int w = 0; w < width; w++) {
                data[c][h][w] = resized[h * width + w];
                //printf("%f\n", data[c][h][w]);
            }
        }
    }

    stbi_image_free(image);
    free(channel);
    free(resized);
    return true;
}

// flip downscaled image horizontally
void flipHorizontally(unsigned char* data, int width, int channels) {
    for (int x = 0; x < width / 2; x++) {
        for (int y = 0; y < width; y++) {
            for (int c = 0; c < channels; c++) {
                int index = y * width * channels + x * channels + c;
                int newIndex = y * width * channels + (width - 1 - x) * channels + c;

                int temp = data[index];
                data[index] = data[newIndex];
                data[newIndex] = temp;
            }
        }
    }
}

// get next training image
bool getTrainingImage(int* label, float*** data, int width, int channels) {

    // get file path of next image, should be train\[cat,dog].[index].jpg
    char file_name[100] = "train\\";
    strcat(file_name, CATDOG[currentPet]);
    strcat(file_name, ".");
    char buffer[20];
    sprintf(buffer, "%i", numberOfTrainings / 2);
    strcat(file_name, buffer);
    strcat(file_name, ".jpg");

    *label = currentPet;
    currentPet = 1 - currentPet;
    numberOfTrainings++;

    return read_jpg(file_name, data, width, channels);
}

// get images by numbers
bool getBatch(int* label, float**** data, int width, int channels, int batchSize, int* numbers) {

    // get file path of next image, should be train\[width]\[number]_[label].jpg
    char file_name_cat[100] = "train\\";
    char file_name_dog[100] = "train\\";
    char buffer[20];
    sprintf(buffer, "%i", width);
    strcat(file_name_cat, buffer);
    strcat(file_name_dog, buffer);
    strcat(file_name_cat, "\\");
    strcat(file_name_dog, "\\");

    // read images
    float** image = (float**)malloc(batchSize * sizeof(float*));
    for (int b = 0; b < batchSize; b++) {

        char file_name[100];
        sprintf(file_name, file_name_cat);
        sprintf(buffer, "%i", numbers[b]);
        strcat(file_name, buffer);
        strcat(file_name, "_0.jpg");

        int origWidth, origHeight, bpp;
        image[b] = stbi_loadf(file_name, &origWidth, &origHeight, &bpp, channels);
        label[b] = 0;
        if (image[b] == NULL) {

            sprintf(file_name, file_name_dog);
            sprintf(buffer, "%i", numbers[b]);
            strcat(file_name, buffer);
            strcat(file_name, "_1.jpg");

            image[b] = stbi_loadf(file_name, &origWidth, &origHeight, &bpp, channels);
            label[b] = 1;
            if (image[b] == NULL) {
                printf("Could not read %s\n", file_name);
                return false;
            }
        }
    }

    for (int b = 0; b < batchSize; b++) {

        // copy values to 3D array data
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < width; h++) {
                for (int w = 0; w < width; w++) {
                    data[b][c][h][w] = image[b][h * width + w];
                    //printf("%f\n", data[c][h][w]);
                }
            }
        }
        stbi_image_free(image[b]);
    }
    free(image);

    return true;
}

// get single image by number
bool getImage(int* label, float*** data, int width, int channels, int number) {

    // get file path of next image, should be train\[width]\[number]_[label].jpg
    char file_name_cat[100] = "train\\";
    char file_name_dog[100] = "train\\";
    char buffer[20];
    sprintf(buffer, "%i", width);
    strcat(file_name_cat, buffer);
    strcat(file_name_dog, buffer);
    strcat(file_name_cat, "\\");
    strcat(file_name_dog, "\\");
    sprintf(buffer, "%i", number);
    strcat(file_name_cat, buffer);
    strcat(file_name_dog, buffer);
    strcat(file_name_cat, "_0.jpg");
    strcat(file_name_dog, "_1.jpg");

    // read image
    int origWidth, origHeight, bpp;
    float* image = stbi_loadf(file_name_cat, &origWidth, &origHeight, &bpp, channels);
    *label = 0;
    if (image == NULL) {
        image = stbi_loadf(file_name_dog, &origWidth, &origHeight, &bpp, channels);
        *label = 1;
        if (image == NULL) {
            printf("Could not read %s\n", file_name_dog);
            return false;
        }
    }

    // copy values to 3D array data
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < width; h++) {
            for (int w = 0; w < width; w++) {
                data[c][h][w] = image[h * width + w];
                //printf("%f\n", data[c][h][w]);
            }
        }
    }

    stbi_image_free(image);
    return true;
}

// get next test image
bool getTestImage(int* label, float*** data, int width, int channels) {

    if (numberOfTests >= labels) {
        return false;
    }

    // get file path of next image, should be test\[index].jpg
    char file_name[20] = "test\\";
    char buffer[20];
    sprintf(buffer, "%i", numberOfTests + 1);
    strcat(file_name, buffer);
    strcat(file_name, ".jpg");

    *label = testLabels[numberOfTests];
    numberOfTests++;

    return read_jpg(file_name, data, width, channels);
}
