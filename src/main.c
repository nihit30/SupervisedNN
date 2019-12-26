
// Dataset Used : Iris
//  main.c


// g(z) = z/2(1+abs(z))+0.5                       sigmoid with 2 divides and no multiplication

// CAUTION :  HIDDEN LAYER NEURONS CAN NEVER EXCEED INPUT LAYER NEURONS

// Hidden Layer 0 = Input Layer
// Hidden Layer [MAX_LAYERS - 1] = Output Layer



#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define NUMBER_OF_EXAMPLES 150                    // Total number of examples in dataset or input text file
#define NUMBER_OF_FEATURES 4                // Corresponds to total columns in dataset or input text file
#define MAX_LAYERS 4                        // 0 = input layer ; (MAX_LAYERS - 1) = output layer
#define NUMBER_OF_OUTPUTS   3               // Number of output neurons
#define HIGH 0.99
#define LOW 0.01
#define generateRandom() ((double)rand()/((double)RAND_MAX+1))


typedef uint8_t BYTE;
typedef uint32_t WORD;
typedef double DWORD;
typedef char IO;

DWORD inputFeatureMatrix[NUMBER_OF_EXAMPLES][NUMBER_OF_FEATURES];              // Matrix for features from Input file

IO* inputAddress = "/home/nihit/Eclipse/EclipseWorkspace/SupervisedNN/src/irisFeatures.txt";        // Input file address
IO* outputAddress = "/home/nihit/Eclipse/EclipseWorkspace/SupervisedNN/src/irisOutputs.txt";       // Output file address

static BYTE currentTrainingExample=0;
const BYTE Neurons[MAX_LAYERS] = {5, 4, 3, NUMBER_OF_OUTPUTS};             // Number of neurons in each layer

bool isAllocMem = false;
bool isReadFiles = false;


/*  ALIGN STRUCTURES TO DWORD BOUNDARY  */

//# pragma pack(8)

typedef struct
{
    uint8_t                  Neurons;         // Neurons
    volatile DWORD*         inputVector;           // Incoming weighted sum to each neuron
    volatile DWORD*         outputVector;          // Outgoing activated signal from each neuron
    volatile DWORD*         deltaVector;           // Error associated with each neuron
    volatile DWORD**        Theta;           // Weights
    volatile DWORD**        changeTheta;     // Weight change
} LAYER;


typedef struct {
    LAYER**            pLayer;         //   layers of the net
    BYTE* outputFeatureMatrix;         // output class vector
    DWORD* desiredOutput;             // Desired output of the Net, changes every iteration according to output class
    DWORD* squaredError;                 // Individual squared error corresponding to each neuron
    DWORD netError;                      // total error of the net
    DWORD gain;                          // Gain used in sigmoid function
    DWORD momentum;                // momentum to control weight update steps, usually around 0.95
    DWORD eta;                            // learning rate
}NN;


/*   INITIALIZE RANDOMS FOR DISTRIBUTION    */
void initRandoms()
{
    srand(3000);
}

DWORD sigmoid(DWORD in)
{
    if(in < -DBL_MAX)
        return 0;
    else if( in > DBL_MAX)
        return 1;
    else
        return (1.0/(1.0 + exp(-in)));
}

DWORD sigmoidDerivative(DWORD in)
{

    return (in*(1-in));


}


DWORD tanH(DWORD in)
{
    return ((2/1.0+exp(-2*in)) - 1);


}

DWORD tanHDerivative(DWORD in)
{

    return (1 - (in*in));

}

DWORD SQUAREDCOST(DWORD error)
{

     return (-0.5*(error*error)) ;

}


/*  INITIALIZE WEIGHTS    */
void distributeWeights(LAYER* higherLayer, LAYER* lowerLayer)
{
    BYTE row,col;
    for(row=0; row< lowerLayer->Neurons ; row++)
    {
        for(col=0; col< higherLayer->Neurons; col++)
        {
            higherLayer->Theta[col][row] =  2*(generateRandom()-0.5)*0.6;
            printf("%f  \t", higherLayer->Theta[col][row]);
        }

        printf("\n");
    }

}

void initializeWeights(NN* nn)
{
    BYTE currentLayer;
    for(currentLayer=1; currentLayer < MAX_LAYERS; currentLayer++)
    {
        printf("Theta %d Matrix\n", currentLayer);
        LAYER* higherLayer = nn->pLayer[currentLayer];
        LAYER* lowerLayer = nn->pLayer[currentLayer-1];
        distributeWeights(higherLayer,lowerLayer);

        printf("\n");                                         // FOR DEBUGGING
    }

}


/*  ALLOCATE MEMORY TO WHOLE STRUCTURE  */
void allocateMem(NN* nn)
{
    if(isAllocMem==true)
    {
        return;
    }

    BYTE currentLayer,previousLayer,rows;
    nn->pLayer =  (LAYER**)malloc((MAX_LAYERS-1)*sizeof(LAYER)+1);
    nn->outputFeatureMatrix = (BYTE*)calloc(NUMBER_OF_EXAMPLES, sizeof(BYTE)+1);
    nn->desiredOutput = (DWORD*)calloc(NUMBER_OF_OUTPUTS, sizeof(DWORD)+1);
    nn->squaredError = (DWORD*)calloc((MAX_LAYERS-1), sizeof(DWORD));



    for(currentLayer=0; currentLayer < MAX_LAYERS; currentLayer++)
    {
        nn->pLayer[currentLayer] =          (LAYER*)malloc(sizeof(LAYER)+1);
        nn->pLayer[currentLayer]->Neurons = Neurons[currentLayer];
        nn->pLayer[currentLayer]->outputVector =  (double*)calloc(Neurons[currentLayer]+1, sizeof(double));


        if(currentLayer!=0)            // except input
        {
            nn->pLayer[currentLayer]->inputVector =   (volatile DWORD*)calloc(Neurons[currentLayer]+1, sizeof(DWORD));
            nn->pLayer[currentLayer]->deltaVector =   (volatile DWORD*)calloc(Neurons[currentLayer]+1, sizeof(DWORD));
            nn->pLayer[currentLayer]->Theta = (volatile DWORD**)calloc(Neurons[currentLayer]+1, sizeof(DWORD*));
            nn->pLayer[currentLayer]->changeTheta = (volatile DWORD**)calloc(Neurons[currentLayer]+1, sizeof(DWORD*));
            previousLayer = currentLayer - 1;

            for(rows=0; rows<Neurons[previousLayer]; rows++)
            {
                nn->pLayer[currentLayer]->Theta[rows] = (volatile DWORD*)calloc(Neurons[previousLayer]+1,sizeof(DWORD));
                nn->pLayer[currentLayer]->changeTheta[rows] = (volatile DWORD*)calloc(Neurons[previousLayer]+1, sizeof(DWORD));
            }

        }

    }

    isAllocMem = true;
}

/*  INITIALIZE INPUT AND OUTPUT FILES  */
void readFeaturesFromFiles(NN* nn)
{
    if(isReadFiles==true)
    {
        return;
    }

    BYTE currentRow;
    BYTE currentCol;

    FILE *INPUTFILE;
    FILE *OUTPUTFILE;
    INPUTFILE = fopen(inputAddress, "r");
    OUTPUTFILE = fopen(outputAddress, "r");

    printf("INPUT FILE \n");
    for(currentRow = 0; currentRow < NUMBER_OF_EXAMPLES; currentRow++)
    {
        for(currentCol = 0; currentCol < NUMBER_OF_FEATURES; currentCol++)
        {
            if (!fscanf(INPUTFILE, " %lf%*c", &inputFeatureMatrix[currentRow][currentCol]))
                break;
            //printf("%lf\t",inputFeatureMatrix[currentRow][currentCol]);
        }
        if(!fscanf(OUTPUTFILE, "%hhu", &nn->outputFeatureMatrix[currentRow]))            // %hhu = uint8_t, %hu = uint16_t, %zu = uint long
            break;
        //printf("\t%d\n",outputFeatureMatrix[currentRow]);

    }
    fclose(INPUTFILE);
    fclose(OUTPUTFILE);
    isReadFiles = true;

}


void initTrainingExample(NN* nn)
{

    WORD currentFeature;
    for(; currentTrainingExample<150; )
    {
        for(currentFeature=0; currentFeature < NUMBER_OF_FEATURES; currentFeature++)
        {
            nn->pLayer[0]->outputVector[currentFeature] = inputFeatureMatrix[currentTrainingExample][currentFeature];

            printf("%f\n", nn->pLayer[0]->outputVector[currentFeature]);
            if(currentFeature==3)                                                    // number of features = 4
            {
                nn->desiredOutput[nn->outputFeatureMatrix[currentTrainingExample]-1] = HIGH;
                printf("%d\n\n",nn->outputFeatureMatrix[currentTrainingExample]);
                currentTrainingExample++;
                break;
            }
        }
        break;
    }

}






/*   UPDATE WEIGHTS   */
void changeWeights(NN* nn, LAYER* higherLayer, LAYER* lowerLayer)
{
    BYTE row,col;
    for(row =0; row < lowerLayer->Neurons; row++)
    {
        //printf("\n\n\n");
        for(col=0; col < higherLayer->Neurons; col++)
        {
            higherLayer->changeTheta[col][row] =  higherLayer->Theta[col][row] - nn->eta * higherLayer->deltaVector[col] ;    // add learning rate to delta
            higherLayer->Theta[col][row] =   higherLayer->changeTheta[col][row];                             // add momentum terM
            //  printf(" %f\t", higherLayer->changeTheta[col][row]);
        }
        //printf("\n");
    }
}

void updateWeights(NN* nn)
{
    uint8_t currentWeightMatrix;

    for(currentWeightMatrix = MAX_LAYERS - 1; currentWeightMatrix > 0; currentWeightMatrix--)
    {
        LAYER* higherLayer = nn->pLayer[currentWeightMatrix];          // initializing pointer to
        LAYER* lowerLayer = nn->pLayer[currentWeightMatrix - 1];
        // printf("\n\nLAYER %d CHANGE WEIGHTS\n",currentWeightMatrix);
        changeWeights(nn, higherLayer, lowerLayer);
    }
}


/*   CALCULATE GRADIENTS    */
void calculateDeltas(NN* nn, LAYER* higherLayer, LAYER* lowerLayer)
{

    DWORD sumAtLowerLayerNeuron;
    BYTE row,col;

    for(row=0; row < lowerLayer->Neurons; row++)
    {
        printf("\n");
        sumAtLowerLayerNeuron = 0.0;
        for( col=0; col < higherLayer->Neurons; col++)
        {
            sumAtLowerLayerNeuron += higherLayer->deltaVector[col] * higherLayer->Theta[col][row];// matrix inversion by accessing opposite indices ( eg : coloumn instead of row
            printf("\t%f", higherLayer->deltaVector[col]);
            printf("\t%f\n", higherLayer->Theta[col][row]);
        }
        lowerLayer->deltaVector[row] = sigmoidDerivative(lowerLayer->outputVector[row]) * sumAtLowerLayerNeuron;           // sigmoid gradient and sum of errors
        printf("\nLOWER LAYER DELTA : %f\n", lowerLayer->deltaVector[row]);


    }

}


/*     CALCULATE OUTPUT ERRORS     */
void outputDelta(NN *nn)
{
    BYTE row;
    DWORD output = 0;
    nn->netError = 0.0;

    for(row=0; row < nn->pLayer[MAX_LAYERS - 1]->Neurons ; row++)
    {
        double error = nn->pLayer[MAX_LAYERS - 1]->outputVector[row] - nn->desiredOutput[row];
        nn->squaredError[row] = SQUAREDCOST(error);                             // CONVEX OPTIMIZATION error
        output = nn->pLayer[MAX_LAYERS - 1]->outputVector[row];
        printf("\n NODE %d ERROR : %f\n",row+1, nn->squaredError[row]);
        nn->pLayer[MAX_LAYERS - 1]->deltaVector[row] = nn->squaredError[row] * (output * (1 - output));
        printf(" NODE %d DELTA : %f\n\n",row+1, nn->pLayer[MAX_LAYERS - 1]->deltaVector[row]);
        nn->netError += nn->squaredError[row];                                                  // total summed output error
    }
    printf("TOTAL ERROR : %f ", nn->netError);
    printf("\n");

}


/*   BACKWARD PROPAGATION    */
void backprop(NN* nn)
{
    BYTE currentLayer;

    for(currentLayer=MAX_LAYERS - 1; currentLayer>1; currentLayer--)
    {
        LAYER* higherLayer = nn->pLayer[currentLayer];
        LAYER* lowerLayer = nn->pLayer[currentLayer - 1];
        calculateDeltas(nn, higherLayer, lowerLayer);
    }

}

void backwardPass(NN* nn)
{
    outputDelta(nn);
    backprop(nn);
}




void train()
{



}

void crossValidate()
{




}

void test()
{



}

/*   CALCULATE WEIGHTED SUMS AND ACTIVATIONS    */
void calculateActivation(NN* nn, LAYER* higherLayer, LAYER* lowerLayer)
{

    DWORD sum = 0.0;
    BYTE col,row;
    for(col=0; col < higherLayer->Neurons; col++)
    {
        sum = 0.0;
        for(row=0; row < lowerLayer->Neurons; row++)
        {
            sum += lowerLayer->outputVector[row] * higherLayer->Theta[col][row];
            printf("\t%f", lowerLayer->outputVector[row]);
            printf("\t%f\n", higherLayer->Theta[col][row]);
        }
        higherLayer->inputVector[col] = sum;
        higherLayer->outputVector[col] = sigmoid(sum);                                     // Activation function = sigmoid
        printf("\t higher layer output : %f\n\n", higherLayer->outputVector[col]);
    }
}




/*   FORWARD PROPAGATION     */
void forwardPass(NN* nn)
{

    BYTE currentLayer;

    for(currentLayer=1; currentLayer < MAX_LAYERS; currentLayer++)
    {

        LAYER* higherLayer = nn->pLayer[currentLayer];
        LAYER* lowerLayer = nn->pLayer[currentLayer-1];
        calculateActivation(nn,higherLayer,lowerLayer);
    }


    // printf("\n");
}





/*    INITIALIZING NETWORK ONLY ONCE    */
void initializeNetwork(NN* nn)                // Functions that run only once in the program
{

    initRandoms();
    allocateMem(nn);                   // Allocate memory for NN
    initializeWeights(nn);             // Initialize inputs and randomize weights

}

void freeMemory(NN* nn)
{
    BYTE i;
    for( i=0; i < MAX_LAYERS - 1; i++)
    {

        free(nn->pLayer[i]);

    }
    free(nn);

}


/*   MAIN    */
int main()
{
    // Application specific initialization
    WORD epoch,iterationOverDataset;
    NN* nn = malloc(sizeof(NN));

    initializeNetwork(nn);

    readFeaturesFromFiles(nn);

    nn->eta = 0.01;


    for(epoch = 0; epoch < 100; epoch++)
    {
        for(iterationOverDataset = 0; iterationOverDataset< NUMBER_OF_EXAMPLES ; iterationOverDataset++)
        {
            initTrainingExample(nn);
            forwardPass(nn);
            backwardPass(nn);
            updateWeights(nn);
        }
         printf("\n TOTAL ERROR : %f",-nn->netError);

    }
    printf("\n\n");
    freeMemory(nn);
    return 0;
}

