
const Layer = require('./layer');

class InputLayer extends Layer {
  constructor(size) {
    super('InputLayer', 0);
    this.size = size;
  }
  
  output(x) {
    return x;
  }

  get activation() {
    return this.act;
  }

  activate(x) {
    this.act = x.slice();
    return this.act;
  }

  log() {
    return `${this.type}(${this.size})`;
  }

  serialize() {
    return {
      type: this.type, 
      size: this.size
    }
  }
}

module.exports = InputLayer;