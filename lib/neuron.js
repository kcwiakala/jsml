
const heaviside = require('./transfer').heaviside;
const initializer = require('./initializer');

class Neuron {
  constructor(inputSize, transfer, initializer) {
    this.transfer = transfer || heaviside;
    this.w = new Array(inputSize);
    this.init(initializer || initializer.constant(0.0));
  }

  init(initializer) {
    for(let i=0; i<this.w.length; ++i) {
      this.w[i] = initializer();
    }
    this.b = initializer();
  }

  out(x) {
    let s = this.b;
    for(let i in x) {
      s += this.w[i] * x[i];
    }
    return this.transfer(s);
  }
}

module.exports = Neuron;