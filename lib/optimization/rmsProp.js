
const StochasticGradientDescent = require('./stochasticGradientDescent');

class RMSProp extends StochasticGradientDescent {
  constructor(rate, gamma, epsilon) {
    super(rate);
    this.gamma = gamma || 0.9;
    this.epsilon = epsilon || 0.000001;
  }

  prepareNeuron(neuron) {
    neuron._gw = new Array(neuron.w.length);
    neuron._gw.fill(0);
    neuron._gb = 0;
  }

  cleanNeuron(neuron) {
    delete neuron._gw;
    delete neuron._gb;
  }

  /** Adjust weights on single neuron using SGD momentum algorithm.
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
      neuron._gw[i] = this.gamma * neuron._gw[i] + (1 - this.gamma) * Math.pow(gradient, 2);
      neuron.w[i] += gradient * this.rate / Math.sqrt(neuron._gw[i] + this.epsilon) ;
    }

    neuron._gb = this.gamma * neuron._gb + (1 - this.gamma) * Math.pow(de, 2);
    neuron.b += de * this.rate / Math.sqrt(neuron._gb + this.epsilon);
  }
}

module.exports = RMSProp;