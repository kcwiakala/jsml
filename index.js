'use strict';

const _ = require('lodash');

function relu(a) {
  return (a > 0) ? a : 0.0;
}

function sigmoid(a) {
  return 1.0 / (1.0 + Math.exp(-a));
}

function step(a) {
  return (a < 0) ? 0.0 : 1.0;
}

function tanh(a) {
  return (1.0 / (1.0 + Math.exp(-2 * a))) - 1.0;
}

function atan(a) {
  return Math.atan(a);
}

function identity(a) {
  return a;
}

function softsign(a) {
  return a / (1 + Math.abs(a));
}

function softmax(a) {
  let output = [];
  let sum = 0;
  for(let i in a) {
    output[i] = Math.exp(a[i]);
    sum += output[i];
  }
  for(let i in output) {
    output[i] /= sum;
  }
  return output;
}

console.log(softmax([0.1, 0.1, 0.1, 1, 0, 0.8]));
console.log(sigmoid(0.8), sigmoid(1), sigmoid(13), sigmoid(0), sigmoid(-20));