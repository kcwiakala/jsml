
const Neuron = require('./neuron');
const transfer = require('./transfer');
const initializer = require('./initializer');

class Perceptron extends Neuron {
  constructor(inputSize) {
    super(inputSize, transfer.heaviside, initializer.uniform(-1, 1));
  }

  learn(sample) {
    const e = sample.y - this.out(sample.x);
    for(let i in this.w) {
      this.w[i] += e * sample.x[i];
    }
    this.b += e;
  }

  error(L) {
    let e = 0.0;
    for(let j in L) {
      const y = this.out(L[j].x);
      e += Math.abs(y - L[j].y);
    }
    return e / L.length;
  }
}

module.exports = Perceptron;