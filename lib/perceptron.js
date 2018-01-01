
const {heaviside} = require('./activation');
const Network = require('./network');
const FullyConnectedLayer = require('./fullyConnectedLayer');
const InputLayer = require('./inputLayer');

class Perceptron extends Network {
  constructor(inputSize) {
    super('Perceptron', [new InputLayer(inputSize), new FullyConnectedLayer(inputSize, 1, heaviside)]);
  }

  get neuron() {
    return this.outputLayer.neurons[0];
  }

  output(x) {
    return super.output(x)[0];
  }

  error(sample) {
    return sample.y[0] - this.output(sample.x);
  }

  loss(sample) { 
    return Math.abs(this.error(sample));
  }

  totalLoss(samples) {
    return samples.reduce((err, sample) => err + this.loss(sample), 0.0) / samples.length;
  }

  adjust(x, error) {
    for(let i=0; i<this.neuron.size; ++i) {
      this.neuron.w[i] += x[i] * error;
    }
    this.neuron.b += error;
  }

  learn(samples, minLoss, maxIterations) {
    minLoss = minLoss || 0.001;
    maxIterations = maxIterations || 1000;
    let counter = 0;
    while((this.totalLoss(samples) > minLoss) && (++counter < maxIterations)) {
      samples.forEach(sample => this.adjust(sample.x, this.error(sample)));
    }
    return (counter < maxIterations);  
  }
}

module.exports = Perceptron;