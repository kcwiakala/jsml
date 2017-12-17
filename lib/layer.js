
const Neuron = require('./neuron');

class Layer {
  constructor(size) {
    this.size = size;
    this.neurons = new Array(size);
  }

  feed(x) {
    let y = new Array(this.size);
    for(let i in this.neurons) {
      y[i] = this.neurons[i].out(x);
    }
    return y;
  }
};

class PerceptonLayer extends Layer {
  constructor(inputSize, outputSize, activation) {
    super(outputSize);
    for(let i=0; i<outputSize; ++i) {
      this.neurons[i] = new Neuron(inputSize, activation);
    }
  }
}