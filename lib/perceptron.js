
const Neuron = require('./neuron');
const {heaviside} = require('./activation');

class Perceptron {

  constructor(inputSize) {
    this.type = 'Perceptron';
    this.neuron = new Neuron(inputSize, heaviside);
  }

  output(x) {
    return this.neuron.output(x);
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

  learn(samples, minLoss, maxIterations) {
    minLoss = minLoss || 0.001;
    maxIterations = maxIterations || 1000;
    let counter = 0;
    while((this.totalLoss(samples) > minLoss) && (++counter < maxIterations)) {
      samples.forEach(sample => this.neuron.adjust(sample.x, this.error(sample), 1, 0));
    }
    return (counter < maxIterations);  
  }
}

module.exports = Perceptron;