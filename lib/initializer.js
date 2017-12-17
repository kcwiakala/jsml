const bm = require('jsboxmuller');

exports.constant = val => () => val;
exports.uniform = (min, max) => () => Math.random() * (max - min) + min;
exports.normal = (mean, sigma) => () => bm(mean, sigma);
