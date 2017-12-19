
const _ = require('lodash');
const expect = require('chai').expect;

const Perceptron = require('../lib/perceptron');
const activation = require('../lib/activation');
const initializer = require('../lib/initializer');

describe('Perceptron', () => {
  it('Should contain single neuron with heaviside activation and given number of inputs', () => {
    let p = new Perceptron(342);
    expect(p.neuron.activation).to.be.equal(activation.heaviside);
    expect(p.neuron.w).to.have.lengthOf(342);
  });

  it('Should learn from linearly separable sets', () => {
    let p = new Perceptron(2);
    let ls = [
      {x: [0,0], y: [0]},
      {x: [0,1], y: [0]},
      {x: [1,0], y: [1]},
      {x: [1,1], y: [1]},
    ];
    expect(p.learn(ls)).to.be.true;
    expect(p.totalLoss(ls)).to.be.equal(0);
    expect(p.output([-1, 0])).to.be.equal(0);
    expect(p.output([2, 0])).to.be.equal(1);
  });
});