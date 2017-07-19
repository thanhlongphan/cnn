#include <stdio.h>
#include <cstdlib>
#include <random>
#include <time.h>
//#include <math.h>

using namespace std;

//typedef unsigned char     lqimg;
//typedef float             hqimg;
//typedef char              weunit;
//typedef unsigned int      uint;
//typedef int               accum;

typedef float           lqimg;
typedef float           hqimg;
typedef float           weunit;
typedef unsigned int    uint;
typedef float           accum;

#define RANDOM_NORMAL_STDDEV 1.0
#define DEFAULT_BIAS 0.1
#define UNSIGNEDCHAR_FULL 255
#define SIGNEDCHAR_NEGMIN -128
#define SIGNEDCHAR_POSMAX 127

default_random_engine generator;
normal_distribution<float> distribution(0.0, RANDOM_NORMAL_STDDEV);

/*************************************************************************************************************************************
 *                                  STRUCT-DEFINITIONS
 **************************************************************************************************************************************/

typedef struct {
    // input and output size
    uint inwid;
    uint outwid;

    // weights,bias output and delta
    // neuron, inputneuron
    weunit** weig;
    // neuron
    weunit* bias;
    lqimg* out;
    float* delta;
    float* cumDelta;

    // store old deviations for momentum
    float** devWeig;
    float* devBias;

    // activationsFunctions and derivative
    lqimg(*actF)(accum);
    weunit(*actD)(weunit);
    // lossFunctions and derivative
    float(*lossF)(lqimg, int);
    float(*lossD)(lqimg, int);
} fclayer;

typedef struct {
    // input(depth, width)
    uint indep;
    uint inwid;

    // output(depth, width)
    uint outdep;
    uint outwid;

    // nextlayerpadding
    // uint pad;
    uint nextpadding;

    // size of max pool
    uint pool;
    // max indices
    uint*** switchH;
    uint*** switchW;

    // outputdep, outpudwid, outputwid
    lqimg*** out;
    // outputdep, outpudwid, outputwid
    float*** delta;
    float*** cumDelta;

    //matches previous layer
    weunit(*actD)(weunit);
} poollayer;

typedef struct {
    // input(depth, width)
    uint indep;
    uint inwid;

    // output(depth, width)
    uint outdep;
    uint outwid;

    // nextlayerpading, stride and filter(width)
    // uint pad;
    uint nextpadding;
    uint stride;
    uint filtwid;

    // outputdep, inputdep, filterwid, filterwid
    weunit**** filter;
    // outputdep
    weunit* bias;
    // outputdep, outpudwid, outputwid
    lqimg*** out;
    // outputdep, outpudwid, outputwid
    float*** delta;
    float*** cumDelta;

    // store old deviations for momentum
    float**** devFilter;
    float* devBias;

    lqimg(*actF)(accum);
    weunit(*actD)(weunit);
    float(*lossF)(lqimg, int);
    float(*lossD)(lqimg, int);
} conlayer;

typedef struct {
    uint conlayers;
    uint fclayers;

    conlayer **convl;
    poollayer **pl;
    fclayer  **fcl;
    fclayer *sml;

    // float *fclinp;

    // size of pool filters
    uint* pool;

    // (1-dropout) is probability of ignoring an output
    float dropout;
    float learnR;
    float momentum;

} network;

/*************************************************************************************************************************************
 *                                  FUNCTION-DEFINITIONS
 ***************************************************************************************************************************************/

inline lqimg id(accum z) {
    return (lqimg)z;
}
inline lqimg idD(accum z) {
    return 1.0;
}

// without math.h
inline lqimg sigmoidalt(accum z) {
    if (z < 0) {
        if (z > -4) return ((z / 4 + 1) * (z / 4 + 1)) / 2;
        else     return 0;
    }
    else {
        if (z < 4) return 1 - ((z / 4 - 1) * (z / 4 - 1)) / 2;
        else    return 1;
    }
}

// sigmoid function
inline float sigmoid(float z) {
    return 1 / (1 + exp(-z));
}

inline lqimg relu(accum z) {
    if (z < 0) return 0;
    if (z > SIGNEDCHAR_POSMAX) return SIGNEDCHAR_POSMAX;
    else return (weunit)z;
}

inline weunit sigmoidD(weunit z) {
    return (1 - z) * z;
}

inline weunit reluD(weunit z) {
    return (z > 0) ? 1.0 : (0.0);
}

inline float rmse(lqimg output, int desiredOut) {
    return (pow(desiredOut - output, 2)) / 2;
}

inline float rmseD(lqimg output, int desiredOut) {
    return (output - desiredOut);
}

inline float crossEntropy(lqimg output, int desiredOut) {
    return -(desiredOut * log(output) + (1 - desiredOut) * log(1 - output));
}

inline float crossEntropyD(lqimg output, int desiredOut) {
    return (output - desiredOut) / ((1 - output) * output);
}

inline int bernoulli(float p) {
    if ((float)rand() / RAND_MAX < p) {
        return 1;
    }
    else {
        return 0;
    }
}

/************************************************************************************************************************************
                                                MAKE AND BREAK THE NETWORK
************************************************************************************************************************************/

inline weunit castRandom(float x) {
    if (x < SIGNEDCHAR_NEGMIN) return SIGNEDCHAR_NEGMIN / 2;
    else if (x > SIGNEDCHAR_POSMAX) return SIGNEDCHAR_POSMAX / 2;
    else return  (SIGNEDCHAR_POSMAX * x / 2);
}

conlayer* makeConLayer(uint inputwidth, uint inputdepth, uint filterwidth, uint filtercount,
                       uint nextLayerPadding, uint stride, char actChoice, char lossChoice) {
    conlayer *res = (conlayer*)malloc(sizeof(conlayer));
    res->inwid = inputwidth;
    res->indep = inputdepth;

    res->filtwid = filterwidth;
    res->outdep = filtercount;
    res->stride = stride;

    // get padding from next layer
    res->outwid = (res->inwid - res->filtwid) / res->stride + 1;
    res->nextpadding = nextLayerPadding;

    // initialize filter randomly
    res->filter = (weunit****)malloc(sizeof(weunit***) *res->outdep);

    for (uint i = 0; i < res->outdep; i++) {
        if (i == 0) {
            res->filter[0] = (weunit***)malloc(sizeof(weunit**) * res->outdep * res->indep);
        }
        else {
            res->filter[i] = &(res->filter[0][i * res->indep]);
        }
        for (uint j = 0; j < res->indep; j++) {
            if (i == 0 && j == 0) {
                res->filter[0][0] = (weunit**)malloc(sizeof(weunit*) *  res->outdep * res->indep * res->filtwid);
            }
            else {
                res->filter[i][j] = &(res->filter[0][0][i * res->indep * res->filtwid + j * res->filtwid]);
            }
            for (uint k = 0; k < res->filtwid; k++) {
                // use a single coherent array
                if (i == 0 && j == 0 && k == 0) {
                    res->filter[0][0][0] = (weunit*)malloc(sizeof(weunit)  *  res->outdep * res->indep * res->filtwid * res->filtwid);
                }
                // point to beginnings
                else {
                    res->filter[i][j][k] = &(res->filter[0][0][0][i * res->indep * res->filtwid * res->filtwid +
                                             j * res->filtwid * res->filtwid + k * res->filtwid]);
                }
                for (uint l = 0; l < res->filtwid; l++) {
                    res->filter[i][j][k][l] = distribution(generator) / sqrt(inputdepth * filterwidth * filterwidth); //castRandom( distribution(generator) );
                    //printf("random: %f\n", res->filter[i][j][k][l]);
                }
            }
        }
    }

    // output
    res->out = (lqimg***)malloc(sizeof(lqimg**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->out[0] = (lqimg**)malloc(sizeof(lqimg*) * res->outdep * (res->outwid + 2 * res->nextpadding));
        }
        else {
            res->out[fmap] = &(res->out[0][fmap * (res->outwid + 2 * res->nextpadding)]);
        }
        for (uint j = 0; j < res->outwid + 2 * res->nextpadding; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->out[0][0] = (lqimg*)calloc(res->outdep * (res->outwid + 2 * res->nextpadding) *
                                                (res->outwid + 2 * res->nextpadding), sizeof(lqimg));
            }
            // point to beginnings
            else {
                res->out[fmap][j] = &(res->out[0][0][fmap * (res->outwid + 2 * res->nextpadding) * (res->outwid + 2 * res->nextpadding)
                                                     + j * (res->outwid + 2 * res->nextpadding)]);
            }
        }
    }

    // bias
    res->bias = (weunit*)calloc(res->outdep, sizeof(weunit));
    for (int i = 0; i < res->outdep; i++) {
        res->bias[i] = DEFAULT_BIAS;
    }

    // deltas
    res->delta = (float***)malloc(sizeof(float**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->delta[0] = (float**)malloc(sizeof(float*) * res->outdep * res->outwid);
        }
        else {
            res->delta[fmap] = &(res->delta[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->delta[0][0] = (float*)calloc(res->outdep * res->outwid * res->outwid, sizeof(float));
            }
            // point to beginnings
            else {
                res->delta[fmap][j] = &(res->delta[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }

    // cumulative deltas
    res->cumDelta = (float***)malloc(sizeof(float**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->cumDelta[0] = (float**)malloc(sizeof(float*) * res->outdep * res->outwid);
        }
        else {
            res->cumDelta[fmap] = &(res->cumDelta[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->cumDelta[0][0] = (float*)calloc(res->outdep * res->outwid * res->outwid, sizeof(float));
            }
            // point to beginnings
            else {
                res->cumDelta[fmap][j] = &(res->cumDelta[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }

    // old deviations
    res->devFilter = (float****)malloc(sizeof(float***) *res->outdep);

    for (uint i = 0; i < res->outdep; i++) {
        if (i == 0) {
            res->devFilter[0] = (float***)malloc(sizeof(float**) * res->outdep * res->indep);
        }
        else {
            res->devFilter[i] = &(res->devFilter[0][i * res->indep]);
        }
        for (uint j = 0; j < res->indep; j++) {
            if (i == 0 && j == 0) {
                res->devFilter[0][0] = (float**)malloc(sizeof(float*) *  res->outdep * res->indep * res->filtwid);
            }
            else {
                res->devFilter[i][j] = &(res->devFilter[0][0][i * res->indep * res->filtwid + j * res->filtwid]);
            }
            for (uint k = 0; k < res->filtwid; k++) {
                // use a single coherent array
                if (i == 0 && j == 0 && k == 0) {
                    res->devFilter[0][0][0] = (float*)malloc(sizeof(float)  *  res->outdep * res->indep * res->filtwid * res->filtwid);
                }
                // point to beginnings
                else {
                    res->devFilter[i][j][k] = &(res->devFilter[0][0][0][i * res->indep * res->filtwid * res->filtwid +
                                                j * res->filtwid * res->filtwid + k * res->filtwid]);
                }
                for (uint l = 0; l < res->filtwid; l++) {
                    res->devFilter[i][j][k][l] = 0;
                }
            }
        }
    }
    res->devBias = (float*)calloc(res->outdep, sizeof(float));

    // activation functions
    if (actChoice == 's') {
        res->actF = sigmoid;
        res->actD = sigmoidD;
    }
    else if (actChoice == 'r') {
        res->actF = relu;
        res->actD = reluD;
    }
    else {
        res->actF = id;
        res->actD = idD;
    }

    // loss functions
    if (lossChoice == 'r') {
        res->lossF = rmse;
        res->lossD = rmseD;
    }
    else {
        res->lossF = rmse;
        res->lossD = rmseD;
    }

    return res;
}

poollayer* makePoolLayer(conlayer* conlayer, uint pool) {
    poollayer *res = (poollayer*)malloc(sizeof(poollayer));
    res->inwid = conlayer->outwid;
    res->indep = conlayer->outdep;

    // get padding from next layer
    res->outwid = conlayer->outwid / pool;
    res->outdep = conlayer->outdep;
    res->nextpadding = conlayer->nextpadding;
    res->pool = pool;
    res->actD = conlayer->actD;

    //output
    res->out = (lqimg***)malloc(sizeof(lqimg**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->out[0] = (lqimg**)malloc(sizeof(lqimg*) * res->outdep * (res->outwid + 2 * res->nextpadding));
        }
        else {
            res->out[fmap] = &(res->out[0][fmap * (res->outwid + 2 * res->nextpadding)]);
        }
        for (uint j = 0; j < res->outwid + 2 * res->nextpadding; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->out[0][0] = (lqimg*)calloc(res->outdep * (res->outwid + 2 * res->nextpadding) *
                                                (res->outwid + 2 * res->nextpadding), sizeof(lqimg));
            }
            // point to beginnings
            else {
                res->out[fmap][j] = &(res->out[0][0][fmap * (res->outwid + 2 * res->nextpadding) * (res->outwid + 2 * res->nextpadding)
                                                     + j * (res->outwid + 2 * res->nextpadding)]);
            }
        }
    }

    // deltas
    res->delta = (float***)malloc(sizeof(float**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->delta[0] = (float**)malloc(sizeof(float*) * res->outdep * res->outwid);
        }
        else {
            res->delta[fmap] = &(res->delta[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->delta[0][0] = (float*)calloc(res->outdep * res->outwid * res->outwid, sizeof(float));
            }
            // point to beginnings
            else {
                res->delta[fmap][j] = &(res->delta[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }

    // cumulative deltas
    res->cumDelta = (float***)malloc(sizeof(float**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->cumDelta[0] = (float**)malloc(sizeof(float*) * res->outdep * res->outwid);
        }
        else {
            res->cumDelta[fmap] = &(res->cumDelta[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->cumDelta[0][0] = (float*)calloc(res->outdep * res->outwid * res->outwid, sizeof(float));
            }
            // point to beginnings
            else {
                res->cumDelta[fmap][j] = &(res->cumDelta[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }

    // switches
    res->switchH = (uint***)malloc(sizeof(uint**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->switchH[0] = (uint**)malloc(sizeof(uint*) * res->outdep * res->outwid);
        }
        else {
            res->switchH[fmap] = &(res->switchH[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->switchH[0][0] = (uint*)calloc(res->outdep * res->outwid * res->outwid, sizeof(uint));
            }
            // point to beginnings
            else {
                res->switchH[fmap][j] = &(res->switchH[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }
    res->switchW = (uint***)malloc(sizeof(uint**) * res->outdep);
    for (uint fmap = 0; fmap < res->outdep; fmap++) {
        if (fmap == 0) {
            res->switchW[0] = (uint**)malloc(sizeof(uint*) * res->outdep * res->outwid);
        }
        else {
            res->switchW[fmap] = &(res->switchW[0][fmap * res->outwid]);
        }
        for (uint j = 0; j < res->outwid; j++) {
            // use a single coherent array
            if (fmap == 0 && j == 0) {

                res->switchW[0][0] = (uint*)calloc(res->outdep * res->outwid * res->outwid, sizeof(uint));
            }
            // point to beginnings
            else {
                res->switchW[fmap][j] = &(res->switchW[0][0][fmap * res->outwid * res->outwid + j * res->outwid]);
            }
        }
    }

    return res;
}

fclayer* makeFClayer(uint inputsize, uint outputsize, char actChoice, char lossChoice) {
    fclayer *res = (fclayer*)malloc(sizeof(fclayer));
    res->inwid = inputsize;
    res->outwid = outputsize;

    // weights
    res->weig = (weunit**)malloc(sizeof(weunit*) * res->outwid);
    for (uint i = 0; i < res->outwid; i++) {
        if (i == 0) {
            res->weig[0] = (weunit*)malloc(sizeof(weunit) * res->outwid * res->inwid);
        }
        else {
            res->weig[i] = &(res->weig[0][i * res->inwid]);
        }
        for (uint j = 0; j < res->inwid; j++) {
            res->weig[i][j] = distribution(generator) / sqrt(inputsize); //castRandom( distribution(generator) );
            //printf("random: %f\n", res->weig[i][j]);
        }
    }

    // bias
    res->bias = (weunit*)calloc(res->outwid, sizeof(weunit));
    for (int i = 0; i < res->outwid; i++) {
        res->bias[i] = DEFAULT_BIAS;
    }

    // output
    res->out = (lqimg*)malloc(sizeof(lqimg) * res->outwid);

    // deltas
    res->delta = (float*)malloc(sizeof(float) * res->outwid);

    // cumulative deltas
    res->cumDelta = (float*)calloc(res->outwid, sizeof(float));

    // old deviations
    res->devWeig = (float**)malloc(sizeof(float*) * res->outwid);
    for (uint i = 0; i < res->outwid; i++) {
        if (i == 0) {
            res->devWeig[0] = (float*)malloc(sizeof(float) * res->outwid * res->inwid);
        }
        else {
            res->devWeig[i] = &(res->devWeig[0][i * res->inwid]);
        }
        for (uint j = 0; j < res->inwid; j++) {
            res->devWeig[i][j] = 0;
        }
    }
    res->devBias = (float*)calloc(res->outwid, sizeof(float));

    // activation functions
    if (actChoice == 's') {
        res->actF = sigmoid;
        res->actD = sigmoidD;
    }
    else if (actChoice == 'r') {
        res->actF = relu;
        res->actD = reluD;
    }
    else {
        res->actF = id;
        res->actD = idD;
    }

    // loss functions
    if (lossChoice == 'r') {
        res->lossF = rmse;
        res->lossD = rmseD;
    }
    else {
        res->lossF = rmse;
        res->lossD = rmseD;
    }

    return res;
}

/**
 * conlayer: total count of convolution layers
 * inputSize[0]: width of input image (quadractic only), inputSize[1]: depth (how many) input images
 * filtersize[i]: width (quadratic) of filter in i-th layer
 * padding[i]: padding in i-th layer but padding[0] is set to zero regardless
 * filterCount[i]: count of filters (hence output featuremaps) in i-th layer
 * stride[i]: step size of filter in i-th layer
 * pool[i]: size of max pool filter after i-th convolutional layer, 1 = no pooling
 * activConv[i]: choice of activation function in i-th conv layer
 * lossConv[i]: choice of activation function in i-th conv layer
 * fclayers: total count of fully connected layers
 * dimension[i] neurons in i-th layer
 * activFC[i]: choice of activation Function in i-th fc layer
 * lossFC[i]: choice of activation Function in i-th fc layer
 * softmax: width of additional softmax output layer, uses log-likelihood loss function, 0 = no softmax layer
 * learningRate: step size of gradient descent
 * momentum: adds fraction of previous deviation, 0 = no momentum
 * dropout: probability of using a neuron, forces redundancy, 1 = no dropout
 */
network* makeCNN(uint conlayers, uint *inputSize, uint *filterSize, uint *padding, uint *filterCount, uint *stride, uint* pool,
                 char *activConv, char *lossConv, uint fclayers, uint *dimensions, char *activFC, char *lossFC, uint softmax,
                 float learningRate, float momentum, float dropout) {
    network *cnn = (network*)malloc(sizeof(network));
    cnn->conlayers = conlayers;
    cnn->fclayers = fclayers;
    cnn->learnR = learningRate;
    cnn->momentum = momentum;
    cnn->dropout = dropout;
    cnn->pool = pool;


    cnn->convl = (conlayer**)malloc(sizeof(conlayer*) * conlayers);
    cnn->fcl = (fclayer**)malloc(sizeof(fclayer*) * fclayers);
    cnn->pl = (poollayer**)malloc(sizeof(poollayer*) * conlayers);

    for (int i = 0; i < conlayers; i++) {
        cnn->pl[i] = NULL;
    }

    // check for wrong choice of padding
    if (padding[0] != 0) {
        printf("padding=0 im ersten layer nicht moeglich. Rechne weiter mit padding= 0\n");
        padding[0] = 0;
    }
    // check for wrong combination of dimensions
    int test = 0;
    int input[conlayers + 1];
    input[0] = inputSize[0];

    for (uint i = 0; i < conlayers; i++) {
        if ((input[i] - filterSize[i] + 2 * padding[i]) % stride[i] != 0) {
            test = i + 1;
        }
        input[i + 1] = (input[i] - filterSize[i] + 2 * padding[i]) / stride[i];
    }
    if (test != 0) {
        printf("fehler beim erstellen des conNN in layer %u\n", test);
    }

    // make convLayers and poolLayers
    if (conlayers == 1) {

        //printf("inputsize[0,1]: %u , %u, filtersize: %u, filterCount: %u, stride: %u, activCon: %c \n",
        //      inputSize[0], inputSize[1], filterSize[0], filterCount[0], 0, stride[0], activConv[0], lossConv[0] );
        cnn->convl[0] = makeConLayer(inputSize[0], inputSize[1], filterSize[0], filterCount[0], 0, stride[0], activConv[0], lossConv[0]);
    }
    else if (conlayers > 1) {
        cnn->convl[0] = makeConLayer(inputSize[0], inputSize[1], filterSize[0], filterCount[0], padding[1], stride[0], activConv[0], lossConv[0]);
        for (uint i = 1; i < conlayers - 1; i++) {
            if (pool[i - 1] > 1) {
                cnn->pl[i - 1] = makePoolLayer(cnn->convl[i - 1], pool[i - 1]);
                cnn->convl[i] = makeConLayer(cnn->pl[i - 1]->outwid, cnn->pl[i - 1]->outdep, filterSize[i], filterCount[i], padding[i + 1], stride[i], activConv[i], lossConv[i]);
            }
            else {
                cnn->convl[i] = makeConLayer(cnn->convl[i - 1]->outwid, cnn->convl[i - 1]->outdep, filterSize[i], filterCount[i], padding[i + 1], stride[i], activConv[i], lossConv[i]);
            }
        }
        if (pool[conlayers - 2] > 1) {
            cnn->pl[conlayers - 2] = makePoolLayer(cnn->convl[conlayers - 2], pool[conlayers - 2]);
            cnn->convl[conlayers - 1] = makeConLayer(cnn->pl[conlayers - 2]->outwid, cnn->pl[conlayers - 2]->outdep, filterSize[conlayers - 1],
                                        filterCount[conlayers - 1], 0, stride[conlayers - 1], activConv[conlayers - 1], lossConv[conlayers - 1]);
        }
        else {
            cnn->convl[conlayers - 1] = makeConLayer(cnn->convl[conlayers - 2]->outwid, cnn->convl[conlayers - 2]->outdep, filterSize[conlayers - 1],
                                        filterCount[conlayers - 1], 0, stride[conlayers - 1], activConv[conlayers - 1], lossConv[conlayers - 1]);
        }
    }
    uint inputs;
    if (pool[conlayers - 1] > 1) {
        cnn->pl[conlayers - 1] = makePoolLayer(cnn->convl[conlayers - 1], pool[conlayers - 1]);
        inputs = cnn->pl[conlayers - 1]->outwid * cnn->pl[conlayers - 1]->outwid * cnn->pl[conlayers - 1]->outdep;
    }
    else {
        inputs = cnn->convl[conlayers - 1]->outwid * cnn->convl[conlayers - 1]->outwid * cnn->convl[conlayers - 1]->outdep;
    }

    // make fcLayers
    printf("inputs in fully connected: %u\n", inputs);
    //cnn->fclinp = (float*) malloc(sizeof(float) * inputs);
    cnn->fcl[0] = makeFClayer(inputs, dimensions[0], activFC[0], lossFC[0]);
    for (uint i = 1; i < fclayers; i++) {
        cnn->fcl[i] = makeFClayer(cnn->fcl[i - 1]->outwid, dimensions[i], activFC[i], lossFC[i]);
    }

    // softmax layer
    if (softmax > 0) {
        cnn->sml = makeFClayer(cnn->fcl[fclayers - 1]->outwid, softmax, 0, 0);
    }
    else {
        cnn->sml = NULL;
    }

    return cnn;
}

// makes only fullyconnected layer
network* makeFCNN(uint fclayers, uint input, uint *dimensions, char *activ, char *loss, float learningRate) {
    network *res = (network*)malloc(sizeof(network));
    res->fcl = (fclayer**)malloc(sizeof(fclayer*) * fclayers);
    res->conlayers = 0;
    res->fclayers = fclayers;
    res->learnR = learningRate;
    res->fcl[0] = makeFClayer(input, dimensions[0], activ[0], loss[0]);
    for (uint i = 1; i < fclayers; i++) {
        res->fcl[i] = makeFClayer(res->fcl[i - 1]->outwid, dimensions[i], activ[i], loss[i]);
    }

    return res;
}

/*****************************************************************************************************************************************
 *                                  FORWARD PROPAGATION
 ****************************************************************************************************************************************/

void convoluteLayer(float dropout, lqimg ***image, conlayer *cl) {
    uint stride = cl->stride;

    for (uint dep = 0; dep < cl->outdep; dep++) {
        for (uint hei = 0; hei < cl->outwid; hei++) {
            for (uint wid = 0; wid < cl->outwid; wid++) {

                if (dropout < 1 && bernoulli(dropout)) {
                    cl->out[dep][hei][wid] = 0;

                }
                else {
                    accum sum = 0;

                    for (uint filterDep = 0; filterDep < cl->indep; filterDep++) {
                        for (uint filterHei = 0; filterHei < cl->filtwid; filterHei++) {
                            for (uint filterWid = 0; filterWid < cl->filtwid; filterWid++) {
                                //printf("accessing image[%u][%u][%u]", filterDep, hei + filterHei, wid + filterWid);
                                //printf("and filter[%u][%u][%u][%u]\n", dep, filterDep, filterHei, filterWid);
                                sum += cl->filter[dep][filterDep][filterHei][filterWid] * image[filterDep][hei * stride + filterHei][wid * stride + filterWid];
                            }
                        }
                    }
                    //printf("aggregated: %i ", sum + cl->bias[dep]);
                    //printf("downcasting :%u\n",  cl->actF(sum + cl->bias[dep]));
                    cl->out[dep][hei][wid] = cl->actF(sum + cl->bias[dep]) / dropout;
                }
            }
        }
    }
}

void pool(lqimg ***input, poollayer *pl) {
    uint pool = pl->pool;

    for (uint dep = 0; dep < pl->outdep; dep++) {
        for (uint hei = 0; hei < pl->outwid; hei++) {
            for (uint wid = 0; wid < pl->outwid; wid++) {
                lqimg max = 0;

                for (int i = hei * pool; i < (hei + 1) * pool; i++) {
                    for (int j = wid * pool; j < (wid + 1) * pool; j++) {
                        lqimg value = input[dep][i][j];
                        if (value > max) {
                            max = value;
                            pl->switchH[dep][hei][wid] = i;
                            pl->switchW[dep][hei][wid] = j;
                        }
                    }
                }
                pl->out[dep][hei][wid] = max;
            }
        }
    }
}

void computeLayer(float dropout, lqimg *input, fclayer *theLayer) {
    for (uint perc = 0; perc < theLayer->outwid; perc++) {

        if (dropout < 1 && bernoulli(dropout)) {
            theLayer->out[perc] = 0;

        }
        else {
            accum sum = 0;

            for (uint inc = 0; inc < theLayer->inwid; inc++) {
                sum += theLayer->weig[perc][inc] * input[inc];
            }
            sum += theLayer->bias[perc];
            theLayer->out[perc] = theLayer->actF(sum) / dropout;
            //printf("output of %i: %f\n", perc, sum);
        }
    }
}

// no dropout, no activation function
void computeSoftMaxLayer(lqimg *input, fclayer *theLayer) {
    for (uint perc = 0; perc < theLayer->outwid; perc++) {
        accum sum = 0;
        for (uint inc = 0; inc < theLayer->inwid; inc++) {
            sum += theLayer->weig[perc][inc] * input[inc];
        }
        theLayer->out[perc] = sum + theLayer->bias[perc];
    }
}

void computeNN(network *theNetwork, lqimg ***image) {
    float dropout = theNetwork->dropout;

    // conv layers
    convoluteLayer(dropout, image, theNetwork->convl[0]);
    for (uint layer = 1; layer < theNetwork->conlayers; layer++) {
        if (theNetwork->pl[layer - 1] != NULL) {
            pool(theNetwork->convl[layer - 1]->out, theNetwork->pl[layer - 1]);
            convoluteLayer(dropout, theNetwork->pl[layer - 1]->out, theNetwork->convl[layer]);
        }
        else {
            convoluteLayer(dropout, theNetwork->convl[layer - 1]->out, theNetwork->convl[layer]);
        }
    }

    // first fc layer
    if (theNetwork->pl[theNetwork->conlayers - 1] != NULL) {
        pool(theNetwork->convl[theNetwork->conlayers - 1]->out, theNetwork->pl[theNetwork->conlayers - 1]);
        computeLayer(dropout, &(theNetwork->pl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }
    else {
        computeLayer(dropout, &(theNetwork->convl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }

    // fc layers
    for (uint layer = 1; layer < theNetwork->fclayers; layer++) {
        computeLayer(dropout, theNetwork->fcl[layer - 1]->out, theNetwork->fcl[layer]);
    }

    // softmax layer
    fclayer* softmax = theNetwork->sml;
    if (softmax != NULL) {
        computeSoftMaxLayer(theNetwork->fcl[theNetwork->fclayers - 1]->out, softmax);

        // apply activation function, numerically stable
        lqimg max = 0;
        for (int i = 0; i < softmax->outwid; i++) {
            if (softmax->out[i] > max) {
                max = softmax->out[i];
            }
        }
        lqimg sum = 0;
        for (int i = 0; i < softmax->outwid; i++) {
            softmax->out[i] = exp(softmax->out[i] - max);
            sum += softmax->out[i];
        }
        for (int i = 0; i < softmax->outwid; i++) {
            if (sum != 0) {
                softmax->out[i] = softmax->out[i] / sum;
            }
        }
    }
}

/*******************************************************************************************
                                BACK PROPAGATION
 *******************************************************************************************/

void computeDeltaOfConLayer(conlayer *current, conlayer *upLayer) {
    uint stride = upLayer->stride;

    for (uint dep = 0; dep < current->outdep; dep++) {
        for (uint hei = 0; hei < current->outwid; hei++) {
            for (uint wid = 0; wid < current->outwid; wid++) {

                current->delta[dep][hei][wid] = current->actD(current->out[dep][hei][wid]);
                if (current->delta[dep][hei][wid] != 0) {
                    float delta = 0;

                    for (uint filterNum = 0; filterNum < upLayer->outdep; filterNum++) {
                        for (uint filterHei = 0; filterHei <= (upLayer->filtwid - 1) / stride; filterHei++) {
                            for (uint filterWid = 0; filterWid <= (upLayer->filtwid - 1) / stride; filterWid++) {
                                uint filterH = hei % stride + filterHei * stride;
                                uint filterW = wid % stride + filterWid * stride;
                                uint deltaH = hei / stride - filterHei;
                                uint deltaW = wid / stride - filterWid;

                                // zero padding: delta = 0 when index out of bounds
                                if (filterH >= 0 && filterH < upLayer->filtwid && filterW >= 0 && filterW < upLayer->filtwid && deltaH >= 0 && deltaH < upLayer->outwid && deltaW >= 0 && deltaW < upLayer->outwid) {

                                    delta += upLayer->filter[filterNum][dep][filterH][filterW] * upLayer->delta[filterNum][deltaH][deltaW];
                                }
                            }
                        }
                    }
                    current->delta[dep][hei][wid] *= delta;
                    current->cumDelta[dep][hei][wid] += current->delta[dep][hei][wid];
                }
            }
        }
    }
}

void computeDeltaOfPoolLayer(poollayer *current, conlayer *upLayer) {
    uint stride = upLayer->stride;

    for (uint dep = 0; dep < current->outdep; dep++) {
        for (uint hei = 0; hei < current->outwid; hei++) {
            for (uint wid = 0; wid < current->outwid; wid++) {

                current->delta[dep][hei][wid] = current->actD(current->out[dep][hei][wid]);
                if (current->delta[dep][hei][wid] != 0) {
                    float delta = 0;

                    for (uint filterNum = 0; filterNum < upLayer->outdep; filterNum++) {
                        for (uint filterHei = 0; filterHei <= (upLayer->filtwid - 1) / stride; filterHei++) {
                            for (uint filterWid = 0; filterWid <= (upLayer->filtwid - 1) / stride; filterWid++) {
                                uint filterH = hei % stride + filterHei * stride;
                                uint filterW = wid % stride + filterWid * stride;
                                uint deltaH = hei / stride - filterHei;
                                uint deltaW = wid / stride - filterWid;

                                // zero padding: delta = 0 when index out of bounds
                                if (filterH >= 0 && filterH < upLayer->filtwid && filterW >= 0 && filterW < upLayer->filtwid && deltaH >= 0 && deltaH < upLayer->outwid && deltaW >= 0 && deltaW < upLayer->outwid) {

                                    delta += upLayer->filter[filterNum][dep][filterH][filterW] * upLayer->delta[filterNum][deltaH][deltaW];
                                }
                            }
                        }
                    }
                    current->delta[dep][hei][wid] *= delta;
                    current->cumDelta[dep][hei][wid] += current->delta[dep][hei][wid];
                }
            }
        }
    }
}

void feedBackDeltaOfPoolLayer(conlayer *current, poollayer *upLayer) {

    // reset deltas
    for (uint dep = 0; dep < current->outdep; dep++) {
        for (uint hei = 0; hei < current->outwid; hei++) {
            for (uint wid = 0; wid < current->outwid; wid++) {
                current->delta[dep][hei][wid] = 0;
            }
        }
    }

    // feed back deltas to max indizes
    for (uint dep = 0; dep < upLayer->outdep; dep++) {
        for (uint hei = 0; hei < upLayer->outwid; hei++) {
            for (uint wid = 0; wid < upLayer->outwid; wid++) {
                uint maxH = upLayer->switchH[dep][hei][wid];
                uint maxW = upLayer->switchW[dep][hei][wid];
                current->delta[dep][maxH][maxW] = upLayer->delta[dep][hei][wid];
                current->cumDelta[dep][maxH][maxW] += current->delta[dep][maxH][maxW];
            }
        }
    }
}

void computeDeltaOfFCLayer(fclayer *current, fclayer *upLayer) {
    for (uint perc = 0; perc < current->outwid; perc++) {

        current->delta[perc] = current->actD(current->out[perc]);
        if (current->delta[perc] != 0) {
            float delta = 0;
            for (uint out = 0; out < upLayer->outwid; out++) {
                delta += upLayer->delta[out] * upLayer->weig[out][perc];
            }
            current->delta[perc] *= delta;
            current->cumDelta[perc] += current->delta[perc];
        }
    }
}

float computeDeltas(network *theNetwork, float *correctVal) {
    float loss = 0;

    // compute deltas in upmost layer
    if (theNetwork->sml == NULL) {
        fclayer *current = theNetwork->fcl[theNetwork->fclayers - 1];
        for (uint perc = 0; perc < current->outwid; perc++) {
            float result = current->lossF(current->out[perc], correctVal[perc]);
            loss += result * result;
            current->delta[perc] = current->lossD(current->out[perc], correctVal[perc]) * current->actD(current->out[perc]);
            current->cumDelta[perc] += current->delta[perc];
        }
    }
    else {
        for (uint perc = 0; perc < theNetwork->sml->outwid; perc++) {
            // using cross-entropy loss function
            loss += crossEntropy(theNetwork->sml->out[perc], correctVal[perc]);
            theNetwork->sml->delta[perc] = theNetwork->sml->out[perc] - correctVal[perc];
            theNetwork->sml->cumDelta[perc] += theNetwork->sml->delta[perc];
        }
        computeDeltaOfFCLayer(theNetwork->fcl[theNetwork->fclayers - 1], theNetwork->sml);
    }

    // compute deltas in all lower layers
    for (uint layer = theNetwork->fclayers - 1; layer > 0; layer--) {
        computeDeltaOfFCLayer(theNetwork->fcl[layer - 1], theNetwork->fcl[layer]);
    }

    // compute deltas in last convolutional layer
    if (theNetwork->pl[theNetwork->conlayers - 1] != NULL) {
        poollayer* lastPoolLayer = theNetwork->pl[theNetwork->conlayers - 1];
        fclayer* firstFCLayer = theNetwork->fcl[0];
        uint width = lastPoolLayer->outwid;
        for (uint dep = 0; dep < lastPoolLayer->outdep; dep++) {
            for (uint hei = 0; hei < width; hei++) {
                for (uint wid = 0; wid < width; wid++) {

                    lastPoolLayer->delta[dep][hei][wid] = lastPoolLayer->actD(lastPoolLayer->out[dep][hei][wid]);
                    if (lastPoolLayer->delta[dep][hei][wid] != 0) {
                        float delta = 0;
                        uint perc = dep * width * width + hei * width + wid;
                        //printf("perceptron: %u\n", perc);
                        for (uint out = 0; out < firstFCLayer->outwid; out++) {
                            delta += firstFCLayer->delta[out] * firstFCLayer->weig[out][perc];
                        }
                        lastPoolLayer->delta[dep][hei][wid] *= delta;
                        lastPoolLayer->cumDelta[dep][hei][wid] += lastPoolLayer->delta[dep][hei][wid];
                    }
                }
            }
        }
        feedBackDeltaOfPoolLayer(theNetwork->convl[theNetwork->conlayers - 1], lastPoolLayer);

    }
    else {
        conlayer* lastConLayer = theNetwork->convl[theNetwork->conlayers - 1];
        fclayer* firstFCLayer = theNetwork->fcl[0];
        uint width = lastConLayer->outwid;
        for (uint dep = 0; dep < lastConLayer->outdep; dep++) {
            for (uint hei = 0; hei < width; hei++) {
                for (uint wid = 0; wid < width; wid++) {

                    lastConLayer->delta[dep][hei][wid] = lastConLayer->actD(lastConLayer->out[dep][hei][wid]);
                    if (lastConLayer->delta[dep][hei][wid] != 0) {
                        float delta = 0;
                        uint perc = dep * width * width + hei * width + wid;
                        //printf("perceptron: %u\n", perc);
                        for (uint out = 0; out < firstFCLayer->outwid; out++) {
                            delta += firstFCLayer->delta[out] * firstFCLayer->weig[out][perc];
                        }
                        lastConLayer->delta[dep][hei][wid] *= delta;
                        lastConLayer->cumDelta[dep][hei][wid] += lastConLayer->delta[dep][hei][wid];
                    }
                }
            }
        }
    }

    // compute deltas in all lower convolutional layers
    for (uint layer = theNetwork->conlayers - 1; layer > 0; layer--) {
        if (theNetwork->pl[layer - 1] != NULL) {
            computeDeltaOfPoolLayer(theNetwork->pl[layer - 1], theNetwork->convl[layer]);
            feedBackDeltaOfPoolLayer(theNetwork->convl[layer - 1], theNetwork->pl[layer - 1]);
        }
        else {
            computeDeltaOfConLayer(theNetwork->convl[layer - 1], theNetwork->convl[layer]);
        }
    }

    return loss;
}

void correctConLayerWeights(float learning, float momentum, lqimg ***input, conlayer *current) {
    uint stride = current->stride;

    for (uint filterNum = 0; filterNum < current->outdep; filterNum++) {

        // weights
        for (uint dep = 0; dep < current->indep; dep++) {
            for (uint hei = 0; hei < current->filtwid; hei++) {
                for (uint wid = 0; wid < current->filtwid; wid++) {
                    float sum = 0;

                    for (int outHei = 0; outHei < current->outwid; outHei++) {
                        for (int outWid = 0; outWid < current->outwid; outWid++) {
                            sum += current->delta[filterNum][outHei][outWid] * input[dep][hei + outHei * stride][wid + outWid * stride];
                        }
                    }

                    // momentum
                    float deviation = learning * sum + momentum * current->devFilter[filterNum][dep][hei][wid];
                    current->devFilter[filterNum][dep][hei][wid] = deviation;
                    current->filter[filterNum][dep][hei][wid] -= deviation;
                }
            }
        }

        // bias
        float sum = 0;
        for (int outHei = 0; outHei < current->outwid; outHei++) {
            for (int outWid = 0; outWid < current->outwid; outWid++) {
                sum += current->delta[filterNum][outHei][outWid];
            }
        }

        // momentum
        float deviation = learning * sum + momentum * current->devBias[filterNum];
        current->devBias[filterNum] = deviation;
        current->bias[filterNum] -= deviation;
    }
}

void correctConLayerCumWeights(float learning, float momentum, lqimg ***input, conlayer *current) {
    uint stride = current->stride;

    for (uint filterNum = 0; filterNum < current->outdep; filterNum++) {

        // weights
        for (uint dep = 0; dep < current->indep; dep++) {
            for (uint hei = 0; hei < current->filtwid; hei++) {
                for (uint wid = 0; wid < current->filtwid; wid++) {
                    float sum = 0;

                    for (int outHei = 0; outHei < current->outwid; outHei++) {
                        for (int outWid = 0; outWid < current->outwid; outWid++) {
                            sum += current->cumDelta[filterNum][outHei][outWid] * input[dep][hei + outHei * stride][wid + outWid * stride];
                        }
                    }

                    // momentum
                    float deviation = learning * sum + momentum * current->devFilter[filterNum][dep][hei][wid];
                    current->devFilter[filterNum][dep][hei][wid] = deviation;
                    current->filter[filterNum][dep][hei][wid] -= deviation;
                }
            }
        }

        // bias
        float sum = 0;
        for (int outHei = 0; outHei < current->outwid; outHei++) {
            for (int outWid = 0; outWid < current->outwid; outWid++) {
                sum += current->cumDelta[filterNum][outHei][outWid];
            }
        }

        // momentum
        float deviation = learning * sum + momentum * current->devBias[filterNum];
        current->devBias[filterNum] = deviation;
        current->bias[filterNum] -= deviation;
    }
}

void correctFCLayerWeights(float learning, float momentum, lqimg *input, fclayer *current) {
    for (uint perceptron = 0; perceptron < current->outwid; perceptron++) {
        for (uint in = 0; in < current->inwid; in++) {
            float deviation = learning * current->delta[perceptron] * input[in] + momentum * current->devWeig[perceptron][in];
            current->devWeig[perceptron][in] = deviation;
            current->weig[perceptron][in] -= deviation;
        }
        // bias
        float deviation = learning * current->delta[perceptron] + momentum * current->devBias[perceptron];
        current->devBias[perceptron] = deviation;
        current->bias[perceptron] -= deviation;
    }
}

void correctFCLayerCumWeights(float learning, float momentum, lqimg *input, fclayer *current) {
    for (uint perceptron = 0; perceptron < current->outwid; perceptron++) {
        for (uint in = 0; in < current->inwid; in++) {
            float deviation = learning * current->cumDelta[perceptron] * input[in] + momentum * current->devWeig[perceptron][in];
            current->devWeig[perceptron][in] = deviation;
            current->weig[perceptron][in] -= deviation;
        }
        // bias
        float deviation = learning * current->cumDelta[perceptron] + momentum * current->devBias[perceptron];
        current->devBias[perceptron] = deviation;
        current->bias[perceptron] -= deviation;
    }
}

void correctWeights(network *theNetwork, lqimg ***input) {
    float learning = theNetwork->learnR;
    float momentum = theNetwork->momentum;

    // first convolutional layer
    correctConLayerWeights(learning, momentum, input, theNetwork->convl[0]);

    // other convolutional layers
    for (uint layer = 1; layer < theNetwork->conlayers; layer++) {
        if (theNetwork->pl[layer - 1] != NULL) {
            correctConLayerWeights(learning, momentum, theNetwork->pl[layer - 1]->out, theNetwork->convl[layer]);
        }
        else {
            correctConLayerWeights(learning, momentum, theNetwork->convl[layer - 1]->out, theNetwork->convl[layer]);
        }
    }

    // first layer
    if (theNetwork->pl[theNetwork->conlayers - 1] != NULL) {
        correctFCLayerWeights(learning, momentum, &(theNetwork->pl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }
    else {
        correctFCLayerWeights(learning, momentum, &(theNetwork->convl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }

    // all other layers
    for (uint layer = 1; layer < theNetwork->fclayers; layer++) {
        correctFCLayerWeights(learning, momentum, theNetwork->fcl[layer - 1]->out, theNetwork->fcl[layer]);
    }

    // softmax layer
    if (theNetwork->sml != NULL) {
        correctFCLayerWeights(learning, momentum, theNetwork->fcl[theNetwork->fclayers - 1]->out, theNetwork->sml);
    }
}

void correctCumWeights(network *theNetwork, lqimg ***input) {
    float learning = theNetwork->learnR;
    float momentum = theNetwork->momentum;

    // first convolutional layer
    correctConLayerCumWeights(learning, momentum, input, theNetwork->convl[0]);

    // other convolutional layers
    for (uint layer = 1; layer < theNetwork->conlayers; layer++) {
        if (theNetwork->pl[layer - 1] != NULL) {
            correctConLayerCumWeights(learning, momentum, theNetwork->pl[layer - 1]->out, theNetwork->convl[layer]);
        }
        else {
            correctConLayerCumWeights(learning, momentum, theNetwork->convl[layer - 1]->out, theNetwork->convl[layer]);
        }
    }

    // first layer
    if (theNetwork->pl[theNetwork->conlayers - 1] != NULL) {
        correctFCLayerCumWeights(learning, momentum, &(theNetwork->pl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }
    else {
        correctFCLayerCumWeights(learning, momentum, &(theNetwork->convl[theNetwork->conlayers - 1]->out[0][0][0]), theNetwork->fcl[0]);
    }

    // all other layers
    for (uint layer = 1; layer < theNetwork->fclayers; layer++) {
        correctFCLayerCumWeights(learning, momentum, theNetwork->fcl[layer - 1]->out, theNetwork->fcl[layer]);
    }

    // softmax layer
    if (theNetwork->sml != NULL) {
        correctFCLayerCumWeights(learning, momentum, theNetwork->fcl[theNetwork->fclayers - 1]->out, theNetwork->sml);
    }
}

void resetCumDeltas(network *theNetwork) {
    for (int layer = 0; layer < theNetwork->conlayers; layer++) {
        for (int dep = 0; dep < theNetwork->convl[layer]->outdep; dep++) {
            for (int hei = 0; hei < theNetwork->convl[layer]->outwid; hei++) {
                for (int wid = 0; wid < theNetwork->convl[layer]->outwid; wid++) {
                    theNetwork->convl[layer]->cumDelta[dep][hei][wid] = 0;
                }
            }
        }
        if (theNetwork->pl[layer] != NULL) {
            for (int dep = 0; dep < theNetwork->pl[layer]->outdep; dep++) {
                for (int hei = 0; hei < theNetwork->pl[layer]->outwid; hei++) {
                    for (int wid = 0; wid < theNetwork->pl[layer]->outwid; wid++) {
                        theNetwork->pl[layer]->cumDelta[dep][hei][wid] = 0;
                    }
                }
            }
        }
    }
    for (int layer = 0; layer < theNetwork->fclayers; layer++) {
        for (int perc = 0; perc < theNetwork->fcl[layer]->outwid; perc++) {
            theNetwork->fcl[layer]->cumDelta[perc] = 0;
        }
    }
}

float backPropNN(network *theNetwork, lqimg ***input, float *correctVal, bool learn) {
    float loss = computeDeltas(theNetwork, correctVal);
    if (learn) {
        correctCumWeights(theNetwork, input);
        resetCumDeltas(theNetwork);
    }
    return loss;
}

/*******************************************************************************************
                                    PRINT METHODS
*******************************************************************************************/

void printOutputs(network *theNN, lqimg ***input) {
    int BOUND = 100;

    // print label
    printf("OUTPUTS\n");
    printf("input");
    for (int i = 0; i < theNN->conlayers; i++) {
        printf("\t\tconvL");
        if (theNN->pl[1] != NULL) {
            printf("\t\tpoolL");
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        printf("\t\tfcL");
    }
    if (theNN->sml != NULL) {
        printf("\t\tsmL");
    }
    printf("(output)\n");

    // print layersizes
    uint max = theNN->convl[0]->inwid * theNN->convl[0]->inwid * theNN->convl[0]->indep;
    printf("%i\t\t", max);
    for (int i = 0; i < theNN->conlayers; i++) {
        uint size = theNN->convl[i]->outwid * theNN->convl[i]->outwid * theNN->convl[i]->outdep;
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
        if (theNN->pl[i] != NULL) {
            printf("%i\t\t", theNN->pl[i]->outwid * theNN->pl[i]->outwid * theNN->pl[i]->outdep);
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        uint size = theNN->fcl[i]->outwid;
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
    }
    if (theNN->sml != NULL) {
        uint size = theNN->sml->outwid;
        if (size > max) {
            max = size;
        }
        printf("%i", size);
    }
    if (BOUND < max) {
        max = BOUND;
    }
    printf("\n\n");

    // print values
    for (int i = 0; i < max; i++) {
        if (input != NULL && theNN->convl[0]->inwid * theNN->convl[0]->inwid * theNN->convl[0]->indep > i) {
            uint dep = i / (theNN->convl[0]->inwid * theNN->convl[0]->inwid);
            uint hei = (i % (theNN->convl[0]->inwid * theNN->convl[0]->inwid)) / theNN->convl[0]->inwid;
            uint wid = i % theNN->convl[0]->inwid;
            printf("%.6f\t", input[dep][hei][wid]);
        }
        else {
            printf("\t\t");
        }
        for (int j = 0; j < theNN->conlayers; j++) {
            if (theNN->convl[j]->outwid * theNN->convl[j]->outwid * theNN->convl[j]->outdep > i) {
                uint dep = i / (theNN->convl[j]->outwid * theNN->convl[j]->outwid);
                uint hei = (i % (theNN->convl[j]->outwid * theNN->convl[j]->outwid)) / theNN->convl[j]->outwid;
                uint wid = i % theNN->convl[j]->outwid;
                printf("%.6f\t", theNN->convl[j]->out[dep][hei][wid]);
            }
            else {
                printf("\t\t");
            }
            if (theNN->pl[j] != NULL) {
                if (theNN->pl[j]->outwid * theNN->pl[j]->outwid * theNN->pl[j]->outdep > i) {
                    uint dep = i / (theNN->pl[j]->outwid * theNN->pl[j]->outwid);
                    uint hei = (i % (theNN->pl[j]->outwid * theNN->pl[j]->outwid)) / theNN->pl[j]->outwid;
                    uint wid = i % theNN->pl[j]->outwid;
                    printf("%.6f\t", theNN->pl[j]->out[dep][hei][wid]);
                }
                else {
                    printf("\t\t");
                }
            }
        }
        for (int j = 0; j < theNN->fclayers; j++) {
            if (theNN->fcl[j]->outwid > i) {
                printf("%.6f\t", theNN->fcl[j]->out[i]);
            }
            else {
                printf("\t\t");
            }
        }
        if (theNN->sml != NULL) {
            if (theNN->sml->outwid > i) {
                printf("%.6f", theNN->sml->out[i]);
            }
        }
        printf("\n");
    }
    printf("\n");

}

void printDeltas(network *theNN) {
    int BOUND = 100;

    // print label
    printf("DELTAS\n");
    printf("input");
    for (int i = 0; i < theNN->conlayers; i++) {
        printf("\t\tconvL");
        if (theNN->pl[i] != NULL) {
            printf("\t\tpoolL");
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        printf("\t\tfcL");
    }
    if (theNN->sml != NULL) {
        printf("\t\tsmL");
    }
    printf("(output)\n");

    // print layersizes
    uint max = 0;
    printf("\t\t");
    for (int i = 0; i < theNN->conlayers; i++) {
        uint size = theNN->convl[i]->outwid * theNN->convl[i]->outwid * theNN->convl[i]->outdep;
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
        if (theNN->pl[i] != NULL) {
            printf("%i\t\t", theNN->pl[i]->outwid * theNN->pl[i]->outwid * theNN->pl[i]->outdep);
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        uint size = theNN->fcl[i]->outwid;
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
    }
    if (theNN->sml != NULL) {
        uint size = theNN->sml->outwid;
        if (size > max) {
            max = size;
        }
        printf("%i", size);
    }
    if (BOUND < max) {
        max = BOUND;
    }
    printf("\n\n");

    // print values
    for (int i = 0; i < max; i++) {
        printf(" \t\t");
        for (int j = 0; j < theNN->conlayers; j++) {
            if (theNN->convl[j]->outwid * theNN->convl[j]->outwid * theNN->convl[j]->outdep > i) {
                uint dep = i / (theNN->convl[j]->outwid * theNN->convl[j]->outwid);
                uint hei = (i % (theNN->convl[j]->outwid * theNN->convl[j]->outwid)) / theNN->convl[j]->outwid;
                uint wid = i % theNN->convl[j]->outwid;
                printf("%.6f\t", theNN->convl[j]->delta[dep][hei][wid]);
            }
            else {
                printf("\t\t");
            }
            if (theNN->pl[j] != NULL) {
                if (theNN->pl[j]->outwid * theNN->pl[j]->outwid * theNN->pl[j]->outdep > i) {
                    uint dep = i / (theNN->pl[j]->outwid * theNN->pl[j]->outwid);
                    uint hei = (i % (theNN->pl[j]->outwid * theNN->pl[j]->outwid)) / theNN->pl[j]->outwid;
                    uint wid = i % theNN->pl[j]->outwid;
                    printf("%.6f\t", theNN->pl[j]->delta[dep][hei][wid]);
                }
                else {
                    printf("\t\t");
                }
            }
        }
        for (int j = 0; j < theNN->fclayers; j++) {
            if (theNN->fcl[j]->outwid > i) {
                printf("%.6f\t", theNN->fcl[j]->delta[i]);
            }
            else {
                printf("\t\t");
            }
        }
        if (theNN->sml != NULL) {
            if (theNN->sml->outwid > i) {
                printf("%.6f", theNN->sml->delta[i]);
            }
        }
        printf("\n");
    }
    printf("\n");

}

void printWeights(network *theNN) {
    int BOUND = 100;

    //print label
    printf("WEIGHTS\n");
    printf("input");
    for (int i = 0; i < theNN->conlayers; i++) {
        printf("\t\tconvL");
        if (theNN->pl[i] != NULL) {
            printf("\t\tpoolL");
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        printf("\t\tfcL");
    }
    if (theNN->sml != NULL) {
        printf("\t\tsmL");
    }
    printf("(output)\n");

    // print number of weights
    uint max = 0;
    printf("\t\t");
    for (int i = 0; i < theNN->conlayers; i++) {
        uint size = theNN->convl[i]->outdep * (theNN->convl[i]->indep * theNN->convl[i]->filtwid * theNN->convl[i]->filtwid + 1);
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
        if (theNN->pl[i] != NULL) {
            printf("\t\t");
        }
    }
    for (int i = 0; i < theNN->fclayers; i++) {
        uint size = theNN->fcl[i]->outwid * (theNN->fcl[i]->inwid + 1);
        if (size > max) {
            max = size;
        }
        printf("%i\t\t", size);
    }
    if (theNN->sml != NULL) {
        uint size = theNN->sml->outwid * (theNN->sml->inwid + 1);
        if (size > max) {
            max = size;
        }
        printf("%i", size);
    }
    if (BOUND < max) {
        max = BOUND;
    }
    printf("\n\n");

    // print values
    for (int i = 0; i < max; i++) {
        printf(" \t\t");
        for (int j = 0; j < theNN->conlayers; j++) {
            if (theNN->convl[j]->outdep * (theNN->convl[j]->indep * theNN->convl[j]->filtwid * theNN->convl[j]->filtwid + 1) > i) {
                uint filterNum = i / (theNN->convl[j]->indep * theNN->convl[j]->filtwid * theNN->convl[j]->filtwid + 1);
                uint index = i % (theNN->convl[j]->indep * theNN->convl[j]->filtwid * theNN->convl[j]->filtwid + 1);
                if (index < theNN->convl[j]->indep * theNN->convl[j]->filtwid * theNN->convl[j]->filtwid) {
                    uint dep = index / (theNN->convl[j]->filtwid * theNN->convl[j]->filtwid);
                    index %= theNN->convl[j]->filtwid * theNN->convl[j]->filtwid;
                    uint hei = index / theNN->convl[j]->filtwid;
                    uint wid = index % theNN->convl[j]->filtwid;
                    printf("%.6f\t", theNN->convl[j]->filter[filterNum][dep][hei][wid]);
                }
                else {
                    printf("%.6f (b)\t", theNN->convl[j]->bias[filterNum]);
                }
            }
            else {
                printf("\t\t");
            }
            if (theNN->pl[j] != NULL) {
                printf("\t\t");
            }
        }
        for (int j = 0; j < theNN->fclayers; j++) {
            if (theNN->fcl[j]->outwid * (theNN->fcl[j]->inwid + 1) > i) {
                uint perc = i / (theNN->fcl[j]->inwid + 1);
                uint in = i % (theNN->fcl[j]->inwid + 1);
                if (in < theNN->fcl[j]->inwid) {
                    printf("%.6f\t", theNN->fcl[j]->weig[perc][in]);
                }
                else {
                    printf("%.6f (b)\t", theNN->fcl[j]->bias[perc]);
                }
            }
            else {
                printf("\t\t");
            }
        }
        if (theNN->sml != NULL) {
            if (theNN->sml->outwid > i) {
                uint perc = i / (theNN->sml->inwid + 1);
                uint in = i % (theNN->sml->inwid + 1);
                if (in < theNN->sml->inwid) {
                    printf("%.6f\t", theNN->sml->weig[perc][in]);
                }
                else {
                    printf("%.6f (b)\t", theNN->sml->bias[perc]);
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}
