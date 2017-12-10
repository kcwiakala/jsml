
class BoxMuller {
  constructor() {
    this.phase = 0;
    this.z0 = 0.0;
    this.z1 = 0.0;
  }

  generate() {
    while(1) {
      const u = 2 * Math.random() - 1.0;
      const v = 2 * Math.random() - 1.0;
      let s = Math.pow(u, 2.0) + Math.pow(u, 2.0);
      if(s > 0.0 && s < 1.0) {
        s = Math.sqrt(-2.0 * Math.log(s) / s);
        this.z0 = u * s
        this.z1 = v * s;
        break;
      }
    }
  }

  get() {
    this.phase = 1 - this.phase;
    if(this.phase == 0) {
      return this.z1;
    } else {
      this.generate();
      return this.z0;
    }
  }
}

const normalGenerator = new BoxMuller();

exports.constant = val => () => val;

exports.uniform = (min, max) => () => Math.random() * (max - min) + min;

exports.normal = (mean, sigma) => () => normalGenerator.get() * sigma + mean;
