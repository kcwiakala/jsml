
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

  /** Initializes weights and bias of all neurons in the layer 
   * with given initialization methods
   * 
   * @param {Function|Number} weightInit 
   * Function or direct value used to initialize input weights
   * @param {Function|Number} biasInit
   * Function or direct value used to initialize neuron bias.
   * Optional parameter, if skipped weightInit is used.
   */
  init(weightInit, biasInit) {
    this.neurons.forEach(neuron => neuron.init(weightInit, biasInit));
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

  /** Activates all neurons in the layer with given input.
   * 
   * @param {Array} x 
   * Input values array
   */
  activate(x) {
    this.neurons.forEach(neuron => neuron.activate(x));
  }

  get state() {
    return this.neurons.map(neuron => neuron.act);
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
    this.neurons.forEach((neuron, i) => neuron.adjust(x, de[i]));
  }

  errorGradient(error) {
    return this.neurons.map((n, i) => n.errorGradient(error[i]));
  }

  /** Translates layer output error to previous layer based
   * on the connection weights. 
   * 
   * @param {Array} error 
   * Error observed on layer output.
   */
  backpropagateError(error) {
    throw new Error('Method splitError should be implemented by concrete layer');
  }

  /** Produces debug string of current layer object */
  get str() {
    return this.neurons[0] ? 
      `${this.type}(${this.neurons.map(n => n.str).join(', ')})` :
      `${this.type}(uninitialized)`;
  }
};

module.exports = Layer;