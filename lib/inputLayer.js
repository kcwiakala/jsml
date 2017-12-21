
const Layer = require('./layer');

class InputLayer extends Layer {
  constructor(size) {
    super(0);
    this.type = 'InputLayer';
    this.size = size;
  }
  
  output(x) {
    return x;
  }

  get state() {
    return this._state;
  }

  activate(x) {
    this._state = x.slice();
    return this._state;
  }

  log() {
    return `${this.type}(${this.size})`;
  }
}

module.exports = InputLayer;