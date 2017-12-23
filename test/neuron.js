
const _ = require('lodash');
const expect = require('chai').expect;

const Neuron = require('../lib/neuron');
const activation = require('../lib/activation');
const initializer = require('../lib/initializer');

describe('Neuron', () => {

  describe('constructor', () => {
    it('Should create weight array for given inputSize', () => {
      let n = new Neuron(19, activation.sigmoid);
      expect(n.w).to.have.length(19);
    });

    it('Should initialize all weights and bias to 0.0', () => {
      let n = new Neuron(8, activation.sigmoid);
      expect(n.b).to.be.equal(0);
      expect(n.w).to.deep.equal([0,0,0,0,0,0,0,0]);
    });

    it('Should create neuron with given activation function', () => {
      let n1 = new Neuron(2, activation.sigmoid);
      expect(n1.activation).to.be.equal(activation.sigmoid);
      let n2 = new Neuron(2, activation.heaviside);
      expect(n2.activation).to.be.equal(activation.heaviside);
    });
  });

  describe('init', () => {
    it('Should initialize neuron weights with given initialization methods', () => {
      let n = new Neuron(1000, activation.identity);
      n.init(initializer.uniform(3,5), 7);
      expect(_.mean(n.w)).to.be.closeTo(4, 0.1);
      expect(_.min(n.w)).to.be.gte(3);
      expect(_.max(n.w)).to.be.lte(5);
      expect(n.b).to.be.equal(7);
    });

    it('Should use weight initializer for bias if no explicit given', () => {
      let n = new Neuron(2, activation.identity);
      n.init(initializer.constant(87));
      expect(n.w).to.be.deep.equal([87,87]);
      expect(n.b).to.be.equal(87);
    });
  });

  describe('output', () => { 
    it('Should produce neuron output for input according to weights', () => {
      let n = new Neuron(3, activation.identity);
      n.w = [1,2,3];
      n.b = 5;
      expect(n.output([5,6,7])).to.be.equal(43); // 1*5 + 2*6 + 3*7 + 5

      n.w = [0,3,-1];
      n.b = 7;
      expect(n.output([1,1,2])).to.be.equal(8); // 0*1 + 3*1 - 1*2 + 7
    });

    it('Should pass input product through activation function', () => {
      let n = new Neuron(1, activation.sigmoid);
      n.w = [1];
      n.b = 0;
      expect(n.output([0])).to.be.equal(0.5);
      expect(n.output([5])).to.be.closeTo(activation.sigmoid(5), 0.01);
      expect(n.output([-2])).to.be.closeTo(activation.sigmoid(-2), 0.01);
    });
  });
});