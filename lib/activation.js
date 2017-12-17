

const SIGMOID = (x) => 1 / (1 + Math.exp(-x));
SIGMOID.df = (x,y) => y * (1 + y);

const HEAVISIDE = (x) => (x < 0) ? 0.0 : 1.0;
HEAVISIDE.df = (x,y) => 0;

const IDENTITY = (x) => x;
IDENTITY.df = (x,y) => 1;

exports.heaviside = HEAVISIDE;
exports.sigmoid = SIGMOID;
exports.identity = IDENTITY;