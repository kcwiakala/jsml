
/** Describes single neuron of artificial neural network.
 * 
 */
class Neuron {
  /** Creates a new Neuron object
   * 
   * @param {Number} inputSize 
   * Number of input connections
   * @param {Function} activation 
   * Activation function used to produce neuron output
   */
  constructor(inputSize, activation) {
    this.activation = activation;
    this.w = new Array(inputSize);
    this.init(0);
  }

  /** Initializes weights and bias with given initialization
   * methods
   * 
   * @param {Function|Number} weightInit 
   * Function or direct value used to initialize input weights
   * @param {Function|Number} biasInit
   * Function or direct value used to initialize neuron bias.
   * Optional parameter, if skipped weightInit is used.
   */
  init(weightInit, biasInit) {
    biasInit = biasInit || weightInit;  
    if(typeof weightInit === 'number') {
      this.w.fill(weightInit);
    } else {
      for(let i = 0; i < this.w.length; ++i) {
        this.w[i] = weightInit();
      }
    }
    this.b = (typeof biasInit === 'number') ? biasInit : biasInit();
  }

  /** Computes neuron output for given input.
   * 
   * @param {Array} x
   * Input array 
   */
  output(x) {
    let s=this.b;
    for(let i in this.w) {
      s += this.w[i] * x[i];
    }
    return this.activation(s);
  }

  /** Adjusts weights in the neuron according to given 
   * error delta of corresponding input.
   * 
   * @param {Array} x
   * Array of input values 
   * @param {Array} de 
   * Error derivative for given input
   */
  adjust(x, de) {
    for(let i in this.w) {
      this.w[i] += de * x[i];
    }
    this.b += de;
  }

  /** Produces debug string of current neuron object. */
  log() {
    let str = 'Neuron[' + this.w.map(w => w.toFixed(3)) + ' ; ' + this.b.toFixed(3) + ']';
    return str; 
  }
}

module.exports = Neuron;