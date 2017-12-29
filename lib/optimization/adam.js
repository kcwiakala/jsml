
const StochasticGradientDescent = require('./stochasticGradientDescent');

class Adam extends StochasticGradientDescent {
  constructor(rate, b1, b2, epsilon) {
    super(rate);
    this.b1 = b1 || 0.9;
    this.b2 = b2 || 0.999;
    this.epsilon = epsilon || 0.000001;
    this.bias1 = 0;
    this.bias2 = 0;
    this.t = 0;
  }

  prepareNeuron(neuron) {
    neuron._mw = new Array(neuron.w.length);
    neuron._mw.fill(0);
    neuron._mb = 0;
    neuron._vw = neuron._mw.slice();
    neuron._vb = 0;
  }

  cleanNeuron(neuron) {
    delete neuron._mw;
    delete neuron._mb;
    delete neuron._vw;
    delete neuron._vb;
  }

  prepareNetwork(network) {
    super.prepareNetwork(network);
    this.t = 1;
  }

  learnSample(network, sample) {
    this.bias1 = 1 - Math.pow(this.b1, this.t);
    this.bias2 = 1 - Math.pow(this.b2, this.t);
    this.t += 1;
    super.learnSample(network, sample);
  }

  /** Adjust weights on single neuron using SGD Adam algorithm.
   * 
   * @param {Neuron} neuron 
   * Neuron to be updated
   * @param {Array} x
   * Array of neuron inputs causing measured error 
   * @param {Number} de 
   * Error measured on neuron output
   */
  adjustNeuron(neuron, x, de) {
    for(let i=0; i < neuron.size; ++i) {
      const gradient = de * x[i];
      neuron._mw[i] = this.b1 * neuron._mw[i] + (1 - this.b1) * gradient;
      neuron._vw[i] = this.b2 * neuron._vw[i] + (1 - this.b2) * Math.pow(gradient, 2);
      neuron.w[i] += (neuron._mw[i] / this.bias1) * 
        this.rate / (Math.sqrt(neuron._vw[i] / this.bias2) + this.epsilon) ;
    }

    neuron._mb = this.b1 * neuron._mb + (1 - this.b1) * de;
    neuron._vb = this.b2 * neuron._vb + (1 - this.b2) * Math.pow(de, 2);
    neuron.b += (neuron._mb / this.bias1) * 
      this.rate / (Math.sqrt(neuron._vb / this.bias2) + this.epsilon);
  }
}

module.exports = Adam;