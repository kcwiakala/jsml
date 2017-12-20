
const Neuron = require('./neuron');

/** Describes single layer of neurons in artificial neural network.
 * Base type for all other layer specialization.
 */
class Layer {
  
  /** Creates new layer with given number of neurons.
   * 
   * @param {Number} size
   * Number of neurons in the layer. 
   */
  constructor(size) {
    this.type = 'Layer';
    this.size = size;
    this.neurons = new Array(size);
  }

  init(weightInit, biasInit) {
    for(let i in this.neurons) {
      this.neurons[i].init(weightInit, biasInit);
    }
  }

  /** Computes layer output for given input values.
   * 
   * @param {Array} x 
   * Input values array 
   * 
   * @returns {Array}
   * Returns array of outputs produced by neurons.
   */
  output(x) {
    return this.neurons.map(neuron => neuron.output(x));
  }

  /** Adjusts neurons according to input and related
   * error gradient.
   * 
   * @param {Array} x 
   * Array of inputs producing error.
   * @param {Array} de 
   * Array of error gradient produced by layer.
   */
  adjust(x, de) {
    for(let i in this.neurons) {
      this.neurons[i].adjust(x, de[i]);
    }
  }

  errorGradient(error, output) {
    return this.neurons.map((n, i) => n.errorGradient(error[i], output[i]));
  }

  decomposeGradient(errorGradient) {
   return errorGradient; 
  }

  /** Produces debug string of current layer object */
  log() {
    let str = this.type + '(';
    if(this.neurons[0]) {
      str += this.neurons.map(n => n.log()).join(', ');
    } else {
      str += 'uninitialized'
    }
    str += ')';
    return str;
  }
};

module.exports = Layer;