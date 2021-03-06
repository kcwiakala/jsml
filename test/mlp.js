const expect = require('chai').expect;

const activation = require('../lib/activation');
const initializer = require('../lib/initializer');
const {StochasticGradientDescent, Momentum, AdaGrad, RMSProp, Adam} = require('../lib/optimization');

const MultiLayerPerceptron = require('../lib/multiLayerPerceptron');

const training = require('./etc/training');

describe('MultiLayerPerceptron', () => {
  let mlp = new MultiLayerPerceptron([2,3,1], activation.sigmoid);
  
  describe('training', () => {
    let optimizer = new Momentum(0.5, 0.75);
    
    it('Should be able to learn xor', () => {
      mlp.init(initializer.normal(-0.3,0.1));
      expect(optimizer.batchTrain(mlp, training.xor, 10000, 0.01)).to.be.true;
      expect(mlp.output([0,0])[0]).to.be.lt(0.3);
      expect(mlp.output([1,1])[0]).to.be.lt(0.3);
      expect(mlp.output([0,1])[0]).to.be.gt(0.7);
      expect(mlp.output([1,0])[0]).to.be.gt(0.7);
    });
  
    it('Should be able to learn and', () => {
      mlp.init(initializer.uniform(0.5,1));
      expect(optimizer.batchTrain(mlp, training.and, 10000, 0.001)).to.be.true;
    });
  
    it('Should be able to learn or', () => {
      mlp.init(initializer.uniform(0.5,1));
      expect(optimizer.batchTrain(mlp, training.or, 10000, 0.001)).to.be.true;
    });
  
    it('Should be able to learn sin', () => {
      let mlp = new MultiLayerPerceptron([1,2,1], activation.sigmoid);
      mlp.init(initializer.uniform(0.5,1));
      expect(optimizer.batchTrain(mlp, training.sin, 10000, 0.001, 5)).to.be.true;
      expect(mlp.output([0.5])[0]).to.be.closeTo(Math.sin(0.5), 0.05);
      expect(mlp.output([0.2])[0]).to.be.closeTo(Math.sin(0.2), 0.05);
      expect(mlp.output([0.3])[0]).to.be.closeTo(Math.sin(0.3), 0.05);
    });
  });

  describe('serialize', () => {
    it('Should serialize to plain json object', () => {
      let mlp = new MultiLayerPerceptron([3,4,2], activation.sigmoid);
      let json = mlp.serialize();
      expect(json.type).to.be.equal('MultiLayerPerceptron');
      expect(json.layers).to.have.length(3);
      expect(json.layers[0].type).to.be.equal('InputLayer');
      expect(json.layers[1].type).to.be.equal('FullyConnectedLayer');
      expect(json.layers[2].type).to.be.equal('FullyConnectedLayer');
    });
  })
});
