#include "CNN.cpp"
#include "filereader.cpp"

/*****************************************************************************************************************************************
 *                                  MAIN METHOD
 *****************************************************************************************************************************************/

// cats vs dogs
int main(int argc, char **argv) {

    const int width = 143;
    const int channels = 3;
    const int trainings = 1;
    const int trainingImages = 49000;
    const int testImages = 1000;

    uint conlayers = 3;
    uint inputSize[] = { width, channels };
    uint filterSize[] = { 5, 3, 3 };
    uint filterCount[] = { 30, 30, 30 };
    uint stride[] = { 2, 1, 1 };
    uint padding[] = { 0, 0, 0 };
    uint pooling[] = { 2, 2, 2 };
    char activConv[] = { 'r', 'r', 'r' };
    char lossConv[] = { 'r', 'r', 'r' };
    uint fclayers = 3;
    uint dimensions[] = { 1000, 500, 2 };
    char activFC[] = { 'r', 'r', 'r' };
    char lossFC[] = { 'r', 'r', 'r' };
    uint softmax = 0;
    float learningRate = 0.001;
    float momentum = 0;
    float dropout = 1;
    network *nn = makeCNN(conlayers, inputSize, filterSize, padding, filterCount, stride, pooling, activConv, lossConv, fclayers, dimensions, activFC, lossFC, softmax, learningRate, momentum, dropout);
    printWeights(nn);
    //printOutputs(nn, NULL);

    int label;
    float*** data = (float***)malloc(channels * width * width * sizeof(float***));
    for (int c = 0; c < channels; c++) {
        data[c] = (float**)malloc(width * width * sizeof(float**));
        for (int h = 0; h < width; h++) {
            data[c][h] = (float*)malloc(width * sizeof(float*));
        }
    }

    bool exit;
    for (int training = 0; training < trainings; training++) {
        printf("Training %i\n", training);
        for (int i = 0; i < trainingImages; i++) {
            bool success = getTrainingImage(&label, data, width, channels);

            float loss = 0;
            if (success) {

                // train batch
                computeNN(nn, data);
                float correctVal[] = { 1 - (float)label, (float)label };
                loss += backPropNN(nn, data, correctVal);

                printf("Output: \tCat %.6f \tDog %.6f\tLoss %.6f\n", nn->fcl[nn->fclayers - 1]->out[0], nn->fcl[nn->fclayers - 1]->out[1], loss);
            }
            //nn->learnR = nn->learnR * 0.9999;
        }
        //break;
    }

    printOutputs(nn, data);
    printDeltas(nn);
    printWeights(nn);

    int count = 0;
    for (exit = getTestImage(&label, data, width, channels); exit; exit = getTestImage(&label, data, width, channels)) {

        computeNN(nn, data);
        printf("Label: %i, Prediction: %.2f %.2f\n", label, nn->fcl[nn->fclayers - 1]->out[0], nn->fcl[nn->fclayers - 1]->out[1]);
        if (label == 0 && nn->fcl[nn->fclayers - 1]->out[0] > nn->fcl[nn->fclayers - 1]->out[1] || label == 1 && nn->fcl[nn->fclayers - 1]->out[0] <= nn->fcl[nn->fclayers - 1]->out[1]) {
            count++;
        }
    }
    printf("Correct: %i %%\n", count);

    count = 0;
    for (int i = 0; i < testImages; i++) {
        bool success = getTrainingImage(&label, data, width, channels);

        float loss = 0;
        if (success) {

            // train batch
            computeNN(nn, data);
            if (label == 0 && nn->fcl[nn->fclayers - 1]->out[0] > nn->fcl[nn->fclayers - 1]->out[1] || label == 1 && nn->fcl[nn->fclayers - 1]->out[0] <= nn->fcl[nn->fclayers - 1]->out[1]) {
                count++;
            }
        }
    }
    printf("Correct: %f %%\n", (float)count / testImages * 100);
}
