
const Neuron = require('./neuron');
const {heaviside} = require('./activation');

class Perceptron extends Neuron { // Should extend network ...
  constructor(inputSize) {
    super(inputSize, heaviside);
  }

  learn(s) {
    this.adjust(s.x, this.de(s));
  }

  error(s) {
    return Math.abs(s.y - this.feed(s.x));
  }

  de(s) {
    return s.y - this.feed(s.x);
  }

  totalError(ls) {
    let e = 0.0;
    for(let i in ls) {
      e += this.error(ls[i]);
    }
    return e / ls.length;
  }
}

module.exports = Perceptron;