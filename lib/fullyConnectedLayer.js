const Layer = require('./layer');
const Neuron = require('./neuron');

class FullyConnectedLayer extends Layer {

  constructor(inputSize, neuronCount, act) {
    super(neuronCount);
    this.type = 'FullyConnectedLayer';
    this.inputSize = inputSize;
    for(let i=0; i<neuronCount; ++i) {
      this.neurons[i] = new Neuron(inputSize, act);
    }
  }

  backpropagateError(error) {
    let backError = new Array(this.inputSize).fill(0);
    for(let i=0; i<this.inputSize; ++i) {
      for(let j=0; j<this.size; ++j) {
        backError[i] += error[j] * this.neurons[j].w[i];
      }
    }
    return backError;
  }
}

module.exports = FullyConnectedLayer;