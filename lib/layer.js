
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
    this.name = 'Layer';
    this.size = size;
    this.neurons = new Array(size);
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
    let y = new Array(this.size);
    for(let i in this.neurons) {
      y[i] = this.neurons[i].output(x);
    }
    return y;
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

  /** Produces debug string of current layer object */
  log() {
    let str = this.name + '(';
    if(this.neurons[0]) {
      str += this.neurons.map(n => n.log()).join(', ');
    } else {
      str += 'uninitialized'
    }
    str += ')';
  }
};