
const SIGMOID = (x) => 1 / (1 + Math.exp(-x));
SIGMOID.df = (x,y) => y * (1 - y);
SIGMOID.type = 'sigmoid';

const RELU = (x) => (x > 0) ? x : 0.01 * x;
RELU.df = (x,y) => 1;
RELU.type = 'relu';

const HEAVISIDE = (x) => (x < 0) ? 0.0 : 1.0;
HEAVISIDE.df = (x,y) => 0;
HEAVISIDE.type = 'heaviside';

const IDENTITY = (x) => x;
IDENTITY.df = (x,y) => 1;
IDENTITY.type = 'identity';

exports.heaviside = HEAVISIDE;
exports.sigmoid = SIGMOID;
exports.identity = IDENTITY;
exports.relu = RELU;