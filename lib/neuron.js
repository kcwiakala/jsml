
/** Describes single neuron of artificial neural network.
 * Base type for all other neuron specializations.
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
    this.type = 'Neuron';
    this.size = inputSize;
    this.activation = activation;
    this.w = new Array(inputSize);
    this.init(0);
    this.change = this.w.concat([0]);
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
      for(let i = 0; i < this.size; ++i) {
        this.w[i] = weightInit();
      }
    }
    this.b = (typeof biasInit === 'number') ? biasInit : biasInit();
  }

  /** Computes neuron output for given input.
   * 
   * @param {Array} x
   * Input values array
   * 
   * @returns {Number} 
   * Returns neuron activation value for given input.
   */
  output(x) {
    let net = this.b;
    for(let i=0; i<this.size; ++i) {
      net += x[i] * this.w[i];
    }
    return this.activation(net);
  }

  /** Activates neuron with given input. Net and activation values
   * are kept as neuron state.
   * 
   * @param {Array} x 
   * Input values array
   * 
   * @returns {Number}
   * Returns neuron activation value for given input.
   */
  activate(x) {
    this.net = this.b;
    for(let i=0; i<this.size; ++i) {
      this.net += x[i] * this.w[i];
    }
    this.act = this.activation(this.net);
    return this.act;
  }

  /** Adjusts weights in the neuron according to given 
   * error delta of corresponding input.
   * 
   * @param {Array} x
   * Array of input values 
   * @param {Array} de 
   * Error derivative for given input
   */
  adjust(x, de, rate, momentum) {
    for(let i=0; i < this.size; ++i) {
      this.change[i] = de * rate * x[i] + momentum * this.change[i];
      this.w[i] += this.change[i];
    }
    this.change[this.size] = de * rate + momentum * this.change[this.size];
    this.b += this.change[this.size];
  }

  errorGradient(e) {
    return e * this.activation.df(0,this.act);
  }

  /** Produces debug string of current neuron object. */
  get str() {
    return `${this.type}:[${this.w.map(w => w.toFixed(3))};${this.b.toFixed(3)}|${this.activation.type}]`; 
  }
}

module.exports = Neuron;