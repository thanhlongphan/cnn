#include "CNN_batch.cpp"
#include "filereader.cpp"
#include <algorithm>
#include <ctime>

/*****************************************************************************************************************************************
 *                                  MAIN METHOD
 *****************************************************************************************************************************************/

// cats vs dogs
int main(int argc, char **argv) {

    const int width = 96;
    const int channels = 3;
    const int batchSize = 1;
    const int trainings = 1;
    const int trainingImages = 49000;
    const int testImages = 1000;

    double read_time = 0;
    double forward_time = 0;
    double backward_time = 0;
    double overall_time = 0;
    time_t begin;
    time_t end;
    time_t begin_overall;
    time(&begin_overall);

    uint conlayers = 1;
    uint inputSize[] = { width, channels };
    uint filterSize[] = { 3 };
    uint filterCount[] = { 4 };
    uint stride[] = { 3 };
    uint padding[] = { 0 };
    uint pool[] = { 2 };
    char activConv[] = { 'r' };
    char lossConv[] = { 'r' };
    uint fclayers = 1;
    uint dimensions[] = { 500 };
    char activFC[] = { 'r' };
    char lossFC[] = { 'r' };
    uint softmax = 2;
    float learningRate = 0.001;
    float momentum = 0;
    float dropout = 1;
    network *nn = makeCNN(conlayers, inputSize, filterSize, padding, filterCount, stride, pool, activConv, lossConv, fclayers, dimensions, activFC, lossFC, softmax, learningRate, momentum, dropout);
    printWeights(nn);
    //printOutputs(nn, NULL);

    int* label = (int*)malloc(batchSize * sizeof(int));
    float**** data = (float****)malloc(batchSize * channels * width * width * sizeof(float***));
    for (int b = 0; b < batchSize; b++) {
        data[b] = (float***)malloc(channels * width * width * sizeof(float**));
        for (int c = 0; c < channels; c++) {
            data[b][c] = (float**)malloc(width * width * sizeof(float*));
            for (int h = 0; h < width; h++) {
                data[b][c][h] = (float*)malloc(width * sizeof(float));
            }
        }
    }

    int* number = (int*)malloc(trainingImages * sizeof(int));
    for (int i = 0; i < trainingImages; i++) {
        number[i] = i;
    }
    random_shuffle(&number[0], &number[trainingImages]);

    // train
    for (int training = 0; training < trainings; training++) {
        printf("Training %i of %i...\n", training + 1, trainings);
        read_time += difftime(end, begin);
        for (int batch = 0; batch < trainingImages / batchSize; batch++) {
            time(&begin);
            bool success = getBatch(label, data, width, channels, batchSize, &number[batch * batchSize]);
            time(&end);
            read_time += difftime(end, begin);

            float loss = 0;
            if (success) {

                // train batch
                for (int b = 0; b < batchSize; b++) {
                    time(&begin);
                    computeNN(nn, data[b]);
                    time(&end);
                    forward_time += difftime(end, begin);
                    float correctVal[] = { 1 - (float)label[b], (float)label[b] };
                    time(&begin);
                    loss += backPropNN(nn, data[b], correctVal, b == batchSize - 1);
                    time(&end);
                    backward_time += difftime(end, begin);
                }
                //printf("Output: \tCat %.6f \tDog %.6f\tLoss %.6f\n", nn->fcl[nn->fclayers-1]->out[0], nn->fcl[fclayers-1]->out[1], loss);
                printf("Output: \tCat %.6f \tDog %.6f\tLoss %.6f\n", nn->sml->out[0], nn->sml->out[1], loss);
            }
        }
        nn->learnR = nn->learnR * 0.9;
        //break;
    }

    // reset dropout to 1 for testing
    nn->dropout = 1;

    printOutputs(nn, NULL);
    printDeltas(nn);
    printWeights(nn);

    time(&end);
    overall_time = difftime(end, begin_overall);
    printf("Read time: %f %%\nForward time %f %%\nBackward time %f %%\n", read_time / overall_time * 100, forward_time / overall_time * 100, backward_time / overall_time * 100);

    // test
    const char* catDog[] = { "Cat", "Dog" };
    int correct = 0;
    for (int test = 0; test < testImages; test++) {
        int label;
        bool success = getImage(&label, data[0], width, channels, trainingImages + test);

        computeNN(nn, data[0]);
        //printf("Label: %s \tPrediction: %3.f %% Cat \t%3.f %% Dog\n", catDog[label], nn->fcl[nn->fclayers-1]->out[0] * 100, nn->fcl[nn->fclayers-1]->out[1] * 100);
        //if(label == 0 && nn->fcl[nn->fclayers-1]->out[0] < nn->fcl[nn->fclayers-1]->out[1] || label == 1 && nn->fcl[nn->fclayers-1]->out[0] > nn->fcl[nn->fclayers-1]->out[1]) {
        //   correct++;
        //}
        printf("Label: %s \tPrediction: %3.f %% Cat \t%3.f %% Dog\n", catDog[label], nn->sml->out[0] * 100, nn->sml->out[1] * 100);
        if (label == 0 && nn->sml->out[0] > nn->sml->out[1] || label == 1 && nn->sml->out[0] < nn->sml->out[1]) {
            correct++;
        }
    }
    printf("Correct Tests: %.2f %%\n", (float)correct / testImages * 100);

    // test training data
    correct = 0;
    for (int test = 0; test < testImages; test++) {
        int label;
        bool success = getImage(&label, data[0], width, channels, trainingImages - testImages + test);

        computeNN(nn, data[0]);
        //if(label == 0 && nn->fcl[nn->fclayers-1]->out[0] < nn->fcl[nn->fclayers-1]->out[1] || label == 1 && nn->fcl[nn->fclayers-1]->out[0] > nn->fcl[nn->fclayers-1]->out[1]) {
        //   correct++;
        //}
        if (label == 0 && nn->sml->out[0] > nn->sml->out[1] || label == 1 && nn->sml->out[0] < nn->sml->out[1]) {
            correct++;
        }
    }
    printf("Correct Tests on Training Data: %.2f %%\n", (float)correct / testImages * 100);
}
