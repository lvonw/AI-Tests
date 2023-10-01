import numpy
import os

# === FILE IO ===
CURRENT_DIR = os.path.dirname(__file__)
TRAINING_LABELS = "Data/Training/train-labels.idx1-ubyte"
TRAINING_IMAGES = "Data/Training/train-images.idx3-ubyte"
TEST_LABELS = "Data/Test/t10k-labels.idx1-ubyte"
TEST_IMAGES = "Data/Test/t10k-images.idx3-ubyte"

def getPath (relativePath):
    return os.path.join(CURRENT_DIR, relativePath)

def readLabels(path):
    with open(path, 'rb') as file:
        magicNumber = int.from_bytes(file.read(4), byteorder='big', signed=False)
        items       = int.from_bytes(file.read(4), byteorder='big', signed=False)

        data = numpy.frombuffer(file.read(), dtype=numpy.uint8)
        return data

def readImages(path):
    with open(path, 'rb') as file:
        magicNumber = int.from_bytes(file.read(4), byteorder='big', signed=False)
        images      = int.from_bytes(file.read(4), byteorder='big', signed=False)
        rows        = int.from_bytes(file.read(4), byteorder='big', signed=False)
        cols        = int.from_bytes(file.read(4), byteorder='big', signed=False)

        data = numpy.frombuffer(file.read(), dtype=numpy.uint8) / 255
        return data.reshape(images, rows*cols)

# === ACTIVATION ===
def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))  

def reLU(x):
    return numpy.maximum(0,x)

def dReLU(x):
    return numpy.where(x > 0, 1, 0)

def softMax(output):
    ex = numpy.exp(output)
    return ex / numpy.sum(ex)

def dSoftMax(output, i, j):
    if i == j:
        return output[i] * (1 - output[i]) 
    return (-output[i]) * output[j] 

# === COST ===
def oneHot(expected):
    res = numpy.zeros(10)
    res[expected] = 1
    return res

def squareError(output, expected):
    return numpy.square(output - oneHot(expected))  

def dSquareError(output, expected):
    return 2 * (output - oneHot(expected)) 

def logError(output, expected):
    return -(numpy.log(output) * oneHot(expected)) 

def dLogError(output, expected):
    return  output - oneHot(expected)


# === MODEL ===
class MultiLayerPerceptron:
    weights = []
    biases = []
    layers = 0

    def print(self):
        print (self.weights)
        print (self.biases)
    
    def __init__ (self, dims):
        self.layers = len(dims)
        for i in range (0, self.layers - 1):
            self.weights.append(numpy.random.uniform(-0.5, 0.5, size=(dims[i+1], dims[i])))
            self.biases.append(numpy.random.uniform(-0.5, 0.5, size=dims[i+1])) 

    def forwardPropagation(self, input):
        # feed forward
        resActivations = []
        resZs = []
        layerRes = input
        for i in range (0, self.layers-2):
            # Add results of weights and biases
            layerRes = numpy.dot(self.weights[i], layerRes) + self.biases[i]
            resZs.append(layerRes)
            # Add Results of non-linearity
            layerRes = reLU(layerRes)
            resActivations.append(layerRes)
        
        i = self.layers-2    
        resOutput = softMax(numpy.dot(self.weights[i], layerRes) + self.biases[i])

        return (resZs, resActivations, resOutput)
    
    def train(self, data):
        for d in data:
            output = self.forwardPropagation(d)[2]   

    def backwardPropagation(self, output, expected, zs, activations):
        resW = []
        resB = []

        # dC / dA
        dz = dLogError(output, expected)
        dB = dz
        # dA / dW * dC / dA
        dW = activations[len(activations)].dot(dz)

        resW.append(dW)
        resB.append(dB)

        for i in range(self.layers-3, -1, -1):
            dr = dReLU (zs[i])
            dz = self.weights[i+1].dot(dz) * dr
            dW = activations[i].dot(dz)
            dB = dz
            resW.append(dW)
            resB.append(dB)

        return (reversed(resW), reversed(resB))

# === CLASSIFICATION ===
def classify (nn, img):
    output = nn.forwardPropagation(img)[2]
    # determine digit
    res = 0 
    temp = 0
    for i in range (0, len(output)):
        if output[i] > temp:
            temp = output[i]
            res = i

    return res

def test(nn):
    hits    = 0.0
    misses  = 0.0

    tImages = readImages(getPath(TEST_IMAGES))
    tLabels = readLabels(getPath(TEST_LABELS))
    size = tImages.shape[0]

    for i in range (0, size):
        if (classify (nn, tImages[i]) == tLabels[i]):
            hits += 1
        else:
            misses += 1
    ratio = hits / size

    print (f"Hits: {hits}; Misses: {misses}; Ratio: {ratio}")

def main():
    nn = MultiLayerPerceptron([784,16,10])
    test(nn)


if __name__ == "__main__":
    main()